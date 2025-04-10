import os
import sys
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

# 4.2.1 평균 제곱 오차
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# 4.2.2 교차 엔트로피 오차
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


if __name__ == "__main__":
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    mse = mean_squared_error(np.array(y), np.array(t))
    cee = cross_entropy_error(np.array(y), np.array(t))
    print(mse)  # 0.0975
    print(cee)  # 0.510825457099

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    mse = mean_squared_error(np.array(y), np.array(t))
    cee = cross_entropy_error(np.array(y), np.array(t))
    print(mse)  # 0.5975
    print(cee)  # 2.30258409299

    # 4.2.3 미니 배치 데이터 추출
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
