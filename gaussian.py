import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the Gaussian distribution
mu = 0  # Mean
sigma = 0.1  # Standard deviation

# Generate random values from the Gaussian distribution
random_values = np.random.normal(mu, sigma, 1000)

# Plot the histogram of the random values
plt.hist(random_values, bins=30, density=True)

# Plot the probability density function (PDF) of the Gaussian distribution
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
plt.plot(x, pdf, 'r-', label='Gaussian PDF')

# Set the plot labels and title
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.title('Gaussian Distribution')

# Display the plot
plt.legend()
plt.show()
