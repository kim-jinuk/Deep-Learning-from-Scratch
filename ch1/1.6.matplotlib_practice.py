import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 1.6.1 sin 함수 그래프 그리기
def draw_sin_func():
    x = np.arange(0, 6, 0.1)    # x 범위
    y = np.sin(x)               # y 값 설정
    plt.plot(x, y)              # x, y 그리기
    plt.show()

# 1.6.2 sin 함수와 cos 함수 동시에 그리기
def draw_sin_cos_func():
    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y1, label="sin")    # sin 함수 라벨 설정
    plt.plot(x, y2, linestyle="--", label="cos")    # cos 함수 선 스타일을 점섬으로 설정
    plt.xlabel("x")     # x축 라벨
    plt.ylabel("y")     # y축 라벨
    plt.title("sin & cos")  # 그래프 제목
    plt.legend()    # 범례 표기
    plt.show()

# 1.6.3 이미지 표시하기
def draw_image(dir='../dataset/lena.png'):
    img = imread(dir)    # 이미지 읽어오기 (모든 데이터는 ../dataset 경로에 있음)
    plt.imshow(img)     # 읽은 이미지 그리기
    plt.show()

if __name__ == "__main__":
    #draw_sin_func()
    #draw_sin_cos_func()
    draw_image()