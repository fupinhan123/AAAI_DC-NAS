#encoding=utf-8
import numpy as np


a = np.array([
    [1.864, 0.057, 50.2, 50.1, 81.5, 78.8, 82.4, 64.9],
    [1.143, 0.518, 72.3, 72.1, 81.0, 81.2, 65.4, 44.0],
    [1.079, 0.581, 73.9, 73.9, 81.7, 81.7, 84.2, 64.1],
    [1.019, 0.601, 73.9, 74.0, 81.3, 74.0, 84.3, 65.9],
    [0.968, 0.625, 77.1, 77.0, 83.6, 81.2, 84.2, 65.9],
    [0.965, 0.632, 77.4, 77.3, 84.0, 82.8, 84.2, 65.4],
    [0.970, 0.633, 73.9, 73.4, 83.6, 82.8, 84.2, 65.4],
    [0.912, 0.668, 76.4, 75.7, 85.8, 85.9, 89.0, 71.7],
    [0.945, 0.672, 77.5, 77.4, 86.2, 86.6, 88.8, 72.5],
    [0.871, 0.698, 83.0, 82.8, 88.6, 86.0, 87.0, 70.7],
    [0.837, 0.744, 83.5, 83.5, 87.2, 87.7, 89.5, 73.8]
])

b = [0.828, 0.756, 83.6, 83.7, 87.2, 87.9, 89.7, 74.4]
b2 = [0.828, 0.756, 83.6, 83.7, 87.6, 87.9, 89.8, 74.1]
for i in range(a.shape[0]-1):
    print(a[10]-a[i])
print(b - a[10])
print(b2 - a[10])