import polars as pl
import numpy as np 
import pandas as pd

def read_data(file_path):
    df = pl.read_csv(file_path)
    df = df.drop("Round-Trip Time [ms]")
    return df

def downsample(df, factor, seed=42):

    # Find User IDs with at least one positive case
    positive_user_ids = df.filter(pl.col("Is Account Takeover") == 1)["User ID"].unique().to_numpy()

    # Find User IDs with only negative cases
    all_user_ids = df["User ID"].unique().to_numpy()
    negative_user_ids = np.setdiff1d(all_user_ids, positive_user_ids)

    # Choose how many negative User IDs to keep (e.g., 10x the number of positive User IDs)
    n_neg_keep = len(positive_user_ids) * factor
    np.random.seed(seed)
    neg_keep_ids = np.random.choice(negative_user_ids, n_neg_keep, replace=False)

    # Combine User IDs to keep
    keep_ids = np.concatenate([positive_user_ids, neg_keep_ids])

    # Filter DataFrame to keep only selected User IDs
    downsampled_df = df.filter(pl.col("User ID").is_in(keep_ids))

    return downsampled_df


def parse_login_timestamp(df):
    return df.with_columns([
        pl.col("Login Timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f")
    ])

def add_time_since_last_login(df):
    df = df.sort(["User ID", "Login Timestamp"])
    df = df.with_columns([
        (pl.col("Login Timestamp") - pl.col("Login Timestamp").shift(1)).over("User ID").alias("Time Since Last Login")
    ])
    df = df.with_columns([
        pl.col("Time Since Last Login").fill_null(pl.duration(milliseconds=0))
    ])
    df = df.with_columns([
        pl.col("Time Since Last Login").dt.total_seconds().alias("Time Since Last Login")
    ])
    df = df.with_columns([
        (pl.col("Time Since Last Login").is_null()).alias("Is First Login")
    ])
    return df

def fill_nulls(df, cols):
    for col in cols:
        df = df.with_columns([
            pl.col(col).fill_null("null")
        ])
    return df

def add_login_hour_and_day(df):
    df = df.with_columns([
        pl.col("Login Timestamp").dt.hour().alias("Login Hour"),
        pl.col("Login Timestamp").dt.weekday().alias("Login Day of Week")
    ])
    return df

def add_ip_and_country_change_flags(df):
    df = df.with_columns([
        (pl.col("IP Address") != pl.col("IP Address").shift(1)).over("User ID").alias("Has IP Changed"),
        (pl.col("Country") != pl.col("Country").shift(1).over("User ID")).alias("Country Changed")
    ])
    df = df.with_columns([
        pl.col("Has IP Changed").fill_null(False),
        pl.col("Country Changed").fill_null(False)
    ])
    return df

def add_login_success_features(df):
    df = df.with_columns([
        pl.col("Login Successful").cast(pl.Boolean)
    ])
    df = df.with_columns([
        pl.col("Login Successful").cum_sum().over("User ID").alias("Success Group")
    ])
    df = df.with_columns([
        ((pl.col("Login Successful") == False).cast(pl.Int32)).sum().over(["User ID", "Success Group"]).alias("Failed Logins Since Last Success")
    ])
    df = df.with_columns([
        pl.col("Failed Logins Since Last Success").fill_null(0)
    ])
    return df

def add_browser_changed_flag(df):
    df = df.with_columns([
        (pl.col("Browser Name and Version") != pl.col("Browser Name and Version").shift(1).over("User ID")).alias("Browser Changed")
    ])
    df = df.with_columns([
        pl.col("Browser Changed").fill_null(False)
    ])
    return df

def engineer_features(df):
    df = parse_login_timestamp(df)
    df = add_time_since_last_login(df)
    df = fill_nulls(df, ["Region", "City", "Device Type"])
    df = add_login_hour_and_day(df)
    df = add_ip_and_country_change_flags(df)
    df = add_login_success_features(df)
    df = add_browser_changed_flag(df)
    # convert to pandas
    df = df.to_pandas()
    return df

# Usage:
# final_logins_df = feature_engineering_pipeline(downsampled_df)

