#investigating how to use mod division with random numbers

import numpy as np

arr = np.random.rand(10)*100

arr = arr % 10

print(arr)
