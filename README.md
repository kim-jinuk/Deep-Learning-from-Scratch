# 밑바닥부터 시작하는 딥러닝 – 개인 학습용 구현

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
└── README.md  ← you are here
```

## 버전
```
Python ≥ 3.9
NumPy ≥ 1.22
(옵션) Matplotlib, tqdm, Pillow
```

# 🚀 빠른 시작
## 1) 저장소 클론
```
git clone https://github.com/kim-jinuk/Deep-Learning-from-Scratch.git
cd Deep-Learning-from-Scratch
```

## 2) 가상환경 (선택)
```
python -m venv venv && source venv/bin/activate
```

## 3) 예제 실행 – Softmax 구현 확인
```
python ch02/softmax.py
```


<br>


# 🤝 기여 (Contributing)
이 레포는 학습 기록이지만 이슈·PR 환영합니다!

포크(fork) → 브랜치 생성 → 수정 → PR


# 📄 라이선스
본 저장소의 코드는 MIT License.

책의 그림·본문 인용은 저작권법에 따라 교육·연구 목적으로만 사용됩니다. 상업적 이용은 금지합니다.

원저자: © Goki Saito / O’Reilly Japan.

# 🙏 감사의 글
사이토 고키 – ‘밑바닥’ 시리즈로 딥러닝 입문 장벽을 허물어 주신 분.

모든 오픈소스 기여자 – NumPy, Matplotlib, Jupyter 커뮤니티.

“밑바닥을 파다 보면 어느새 지구 반대편에 도달한다.”
학습 기록은 언젠가 당신과 누군가의 지구 반대편을 이어줄지도 모릅니다. 🛠️✨
