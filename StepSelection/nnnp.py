import numpy as np
# b =  np.array([[[]]])

a = np.array([[1, 2, 3], [7, 8, 9]])
b = np.expand_dims(a, 0)
print(b)
print(b.shape)
print()
b = np.append(b,[a],axis = 0)
print(b)
print(b.shape)