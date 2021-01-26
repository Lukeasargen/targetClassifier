import matplotlib.pyplot as plt
import numpy as np


mu = 0.0  # mean
sigma = 0.03 # standard deviation
s = np.random.normal(mu, sigma, 1000)

s = s[s >= 0]  # min threshold values

count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()

