import numpy as np
import matplotlib.pyplot as plt

# 4.3.1 수치 미분 (Bad case)
# 너무 작은 h 값
# 미분 값을 구할 때, 전방 차분이 아닌 중심 차분을 일반적으로 사용
def numerical_diff_bad(f, x):
    h = 10e-50
    # print(np.float32(1e-50))    # 0.0
    return (f(x + h) - f(x)) / h

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 4.3.2 수치 미분의 예
def function_1(x):
    return 0.01*x**2 + 0.1*x


# 4.3.3 편미분
def function_2(x):
    return x[0]**2 + x[1]**2


# 접선을 구하는 함수
def tangent_line(f, x):
    grad = numerical_diff(f, x)
    b = f(x) - grad * x
    return lambda k: grad * k + b

if __name__ == "__main__":
    x = np.arange(0, 20.0, 0.1)
    y1 = function_1(x)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y1, label="function")
    # plt.show()
    
    temp_function = tangent_line(function_1, 8)
    y2 = temp_function(x)
    plt.plot(x, y2, linestyle='--', label="tangent")
    plt.legend()
    plt.show()