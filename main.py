import numpy as np
import matplotlib.pyplot as plt

plt.figure(dpi=300)
noise = np.random.uniform(0, 100, size=(300, 2))
plt.plot(noise.T[0], noise.T[1], 'b.')

cluster1 = np.random.normal((5, 10), 1, (30, 2))
plt.plot(cluster1.T[0], cluster1.T[1], 'r.')

cluster2 = np.random.normal((30, 50), 5, (30, 2))
plt.plot(cluster2.T[0], cluster2.T[1], 'g.')

cluster3 = np.random.normal((70, 20), 1, (30, 2))
plt.plot(cluster3.T[0], cluster3.T[1], 'y.')

plt.show()
