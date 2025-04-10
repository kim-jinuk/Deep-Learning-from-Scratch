import numpy as np

# 5.6 Affine/Softmax 계층 구현하기
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def cross_entropy_error(y, t):
    delta = 1e-7  # 0일때 -무한대가 되지 않기 위해 작은 값을 더함
    return -np.sum(t * np.log(y + delta))


# 5.6.2 Affine 계층
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout=1):
        dx = np.sum(np.dot(dout, self.W.T), axis=0)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


# 5.6.3 Softmax-with-Loss 계층
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실
        self.y = None     # softmax의 출력
        self.t = None     # 정답 레이블(원-핫 벡터)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)  # 3.5.2, 4.2.2에서 구현
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y - self.t / batch_size

        return dx
    

if __name__ == '__main__':   
    X = np.random.rand(2)     # 입력
    W = np.random.rand(2, 3)  # 가중치
    B = np.random.rand(3)     # 편향

    print(X.shape)  # (2,)
    print(W.shape)  # (2, 3)
    print(B.shape)  # (3,)

    aff = Affine(W, B)
    print(aff.forward(X))   # [1.08305964 1.00785265 0.27244078]
    print(aff.backward())

    swl = SoftmaxWithLoss()
    a = np.array([1, 8, 3])   # 비슷하게 맞춤
    t = np.array([0, 1, 0])
    print(swl.forward(a, t))  # 0.0076206166295
    print(swl.backward())     # [ 0.00090496  0.65907491  0.00668679]

    a = np.array([1, 3, 8])   # 오차가 큼
    print(swl.forward(a, t))  # 5.00760576266
    print(swl.backward())   # [  9.04959183e-04 -3.26646539e-01 9.92408247e-01]
    