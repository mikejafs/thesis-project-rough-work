import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i in range(3):
    a = np.random.rand(4)
    plt.plot(a)

plt.show()