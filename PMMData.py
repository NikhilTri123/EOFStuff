import numpy as np

np.set_printoptions(suppress=True)

# Initialize a 2D list to store Oct, Nov, Dec values
oct_nov_dec_values = []

# Read data from the text file
with open('pmmdata.txt', 'r') as file:
    for line in file:
        # Split the line into year and 12 values
        year, *monthly_values = map(float, line.strip().split())

        # Extract Oct, Nov, Dec values (last 3 values in the line)
        oct_nov_dec = monthly_values[-3:]

        # Append Oct, Nov, Dec values to the 2D list
        oct_nov_dec_values.append(oct_nov_dec)

averages = [round(float(np.mean(month_values)), 2) for month_values in oct_nov_dec_values]
averages = np.array([list(range(1948, 2023)), averages]).T
# averages = averages[averages[:, 1].argsort()]

# Print the 2D list containing Oct, Nov, Dec values for each year
print(list(averages[:, 1]))