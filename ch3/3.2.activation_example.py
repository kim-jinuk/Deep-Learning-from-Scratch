import numpy as np
import matplotlib.pyplot as plt

# 3.2.2 계단 함수 구현하기
def step_function(x):
    return np.array(x > 0, dtype=np.int32)


# 3.2.3 그래프 그리기
def draw_function(x=np.arange(-5.0, 5.0, 0.1), func=step_function):
    y = func(x)
    plt.plot(x, y)
    #plt.ylim(-0.1, 1.1)     # y축 범위 지정
    plt.show()


# 3.2.4 시그모이드 함수 구현하기
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 3.2.7 ReLU 함수 구현하기
def ReLU(x):
    return np.maximum(0, x)


# 3.5.2 소프트맥스 함수 구현하기
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y

if __name__ == "__main__":
    #draw_function()
    #draw_function(np.arange(-5.0, 5.0, 0.1), sigmoid)
    draw_function(np.arange(-5.0, 5.0, 0.1), ReLU)