# 1.5.1 넘파이 가져오기
import numpy as np


# 1.5.2 넘파이 배열 생성하기
x = np.array([1.0, 2.0, 3.0])
print(x)    # [1. 2. 3.]
print(type(x))  # <class 'numpy.ndarray'>


# 1.5.3 넘파이의 산술 연산
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y)    # [3. 6. 9.]
print(x - y)    # [-1. -2. -3.]
print(x * y)    # [ 2.  8. 18.]
print(x / y)    # [0.5 0.5 0.5]


# 1.5.4 넘피아의 N차원 배열
A = np.array([[1, 2], [3, 4]])
B = np.array([[3, 0], [0, 6]])
print(A.shape)  # (2, 2)
print(A.dtype)  # int64
print(A + B)    # [[4, 2], [3, 10]]
print(A * B)    # [[3, 0], [0, 24]]


# 1.5.5 브로드캐스트
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)    # [[10, 40], [30, 80]]


# 1.5.6 원소 접근
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X[0])     # [51, 55]
print(X[0][1])  # 55
for row in X:
    print(row)  # [51, 55], [14, 19], [0, 4]

X = X.flatten()
print(X)        # [51 55 14 19 0 4]
print(X[np.array([0, 2, 4])])   # 인덱스가 0, 2, 4인 원소 얻기
print(X > 15)   # [True True False True False False]