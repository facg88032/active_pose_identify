
import numpy as np

# file = open('X_train.txt', 'r')
# X_ = np.array(
#     [elem for elem in [
#         row.split(',') for row in file
#     ]],
#     dtype=np.float32
# )
# file.close()
# blocks = int(len(X_) / 32)
#
# X_ = np.array(np.split(X_, blocks))
#
#
#
#
# Y=np.loadtxt('Y_train.txt')
# print('Y_train')


X_v=np.loadtxt('X_val.txt',delimiter=',')

print('X_v')