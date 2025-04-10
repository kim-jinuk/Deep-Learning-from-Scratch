# 밑바닥부터 시작하는 딥러닝 – 개인 구현

> **From‑Scratch Deep Learning Implementations in Pure Python / NumPy**

![status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)
![license](https://img.shields.io/badge/License-MIT-blue)

본 레포는 사이토 고키(Goki Saito)의 명저 **『밑바닥부터 시작하는 딥러닝(원제: *Deep Learning from Scratch*)』** 을 읽으며 직접 타이핑·실험한 코드를 정리한 공간입니다.  
“공식” 구현이 아닌 **순수 학습용** 저장소이며, 책의 개념을 확인하고 개인적으로 확장·리팩터링한 코드가 포함됩니다.

---

## 📚 목차 & 폴더 구조

```text
.
├── ch01  /  퍼셉트론
├── ch02  /  신경망
├── ch03  /  학습
├── ch04  /  신경망 학습
├── ch05  /  오차역전파
├── ch06  /  CNN
├── ch07  /  DL 실전 팁
├── ch08  /  딥러닝 고급
├── common   /  layers, optim, util
├── dataset  /  mnist.py, spiral.py …
├── requirements.txt
└── README.md  ← you are here

## 저장소 클론
git clone https://github.com/kim-jinuk/Deep-Learning-from-Scratch.git
cd Deep-Learning-from-Scratch

## 요구사항
소스 코드를 실행하려면 아래의 소프트웨어가 설치되어 있어야 합니다.

* 파이썬 3.x
* NumPy
* Matplotlib

## 예제 실행 – Softmax 구현 확인
python ch02/softmax.py
