import numpy as np
mu, sigma = 0.4, 0.5 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
s = s[s >= 0]

import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()

