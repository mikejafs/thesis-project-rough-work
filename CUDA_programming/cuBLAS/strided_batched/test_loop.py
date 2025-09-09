import numpy as np

arr = np.random.randint(0, 10, size = 20)

print(arr)

unique, inverse = np.unique(arr, return_inverse=True)
print(unique,
      np.bincount(inverse))