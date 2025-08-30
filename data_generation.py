import numpy as np


def generate_user_table(df):
	# Make a table of the unique User IDs
	user_table = df.groupby('User ID').agg({
		'Is Account Takeover': 'max'
	})
	# Create 'Any Account Takeover' column
	user_table['Any Account Takeover'] = user_table['Is Account Takeover'] == 1

	# Set random seed for reproducibility
	np.random.seed(42)

	# Generate ages: older for account takeover, younger otherwise
	ages = np.where(
		user_table['Any Account Takeover'],
		np.random.normal(loc=45, scale=8, size=len(user_table)),  # Skew older
		np.random.normal(loc=35, scale=10, size=len(user_table))  # Skew younger
	)
	# Clip ages to reasonable range
	ages = np.clip(ages, 18, 90).astype(int)
	user_table['Age'] = ages

	# Assign random job categories
	job_categories = [
		'Engineer', 'Teacher', 'Doctor', 'Artist', 'Manager',
		'Sales', 'Clerk', 'Technician', 'Consultant', 'Driver'
	]
	user_table['Job Category'] = np.random.choice(job_categories, size=len(user_table))

	return user_table

# Example usage:
# user_table = generate_user_table(df)
# print(user_table)
