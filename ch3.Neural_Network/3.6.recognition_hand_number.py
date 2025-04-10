import os
import sys
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def identity_function(x):
    return x

# PIL로 이미지 읽어서 화면에 보여주기
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# 사전 저장된 매개변수 파일 sample_weight.pkl 불러와서 신경망 초기화
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


# 신경망을 통해 입력 영상 추론
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


if __name__ == "__main__":
    # configure
    accuracy_cnt = 0

    # 손글씨 데이터 로드
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

    # x_train 손글씨 이미지 확인
    img = x_train[0].reshape(28, 28)
    # img_show(img)

    # 신경망 초기화
    network = init_network()

    # 신경망 추론
    for i in range(len(x_test)):
        y = predict(network, x_test[i])
        p = np.argmax(y)
        if p == t_test[i]:
            accuracy_cnt += 1

    print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))