import numpy as np

A = np.array([[1., 2, 3],
              [5, 4, 2],
              [11, 8, 9],
              [4, 7, 1]])
A = np.insert(A, 0, np.zeros(A.shape[0]), axis=1)

index = np.random.randint(len(A), size=(2))
init_cent = A[index, :]


print(init_cent)

print(A)
temp = np.argmin(A[:, 1::], axis=1)+1
print(temp)
A[:, 0] = temp
print(A)

new = np.average(A[A[:, 0] == 3][:, 1::], axis=0)
print(new)
init_cent[0, 1::] = new
print(init_cent)