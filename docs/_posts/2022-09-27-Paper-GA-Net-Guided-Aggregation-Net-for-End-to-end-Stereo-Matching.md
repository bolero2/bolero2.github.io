---
layout: post
title:  "[Paper] GA-Net: Guided Aggregation Net for End-to-end Stereo Matching"
date:   2022-09-27
categories: "Paper Review"
---
# GA-Net: Guided Aggregation Net for End-to-end Stereo Matching

이번에 소개할 논문은 2019년에 CVPR Oral 섹션에서 공개된
**GA-Net** 이라는 딥러닝 모델입니다.

2019년 당시에, RGB 이미지에서 Depth 이미지를 추정하는 **Depth Estimation**에 관한 연구가 활발히 진행되었습니다.

GA-Net은 Stereo 영상(양안)에서 Depth Image를 추정하는 딥러닝 네트워크입니다.

GA-Net은 그들만의 방법으로 Computing cost를 줄이고, 성능을 향상시키는 결과를 도출했습니다.

-----

## 1. Stereo Matching?
* Disparity Estimation
* 각각 다른 시선에서 픽셀들을 매칭시키는 것.  

![](https://velog.velcdn.com/images/bolero2/post/6e9cb1c3-067f-4e21-ab5a-b9e3a22d7516/image.png)

-----

## 2. Contribution

1) 기존의 GC-Net의 경우, 3D stereo matching 연구에서 3D Conv를 사용

2) 이 3D conv는 고비용/높은 복잡도를 요구 → 연산 시 소요되는 시간이 길다.

3) 기존의 SGA(Semi-Global Aggregation)을 다른 방향으로 개선

4) 여기에 추가적으로 LGA(Local Guided Aggregation) 개발

5) **SGA+LGA는 3D Conv를 대체 가능하고, 1/100 수준의 복잡도를 가진다.**

6) 15-20fps로 실시간 평가 및 대조 가능

-----

## 3. Improvement of SGA

1) 기존의 SGA는 직접 조정(tuning)하기 힘든 사용자 설정 파라미터가 있음, 이 파라미터는 네트워크 학습 도중에 가변성이 매우 높다 → unstable 함.

2) SGM으로 픽셀 매칭 시 너무 다른 환경에 영향을 많이 받음.

3) 그래서 SGA를 세 방향으로 개선함.
> 3-1) 사용자 설정 파라미터를 학습 가능하게 변경(learnable)
> 
> 3-2) 첫 번째/외부 최소 선택을 가중치의 합으로 대체함, 기존 연구에서 strides는 max-pooling layer 대체해서 정확도 손실이 없음을 밝혔음.
>
> 3-3) 내부/두 번째 최소 선택(selection)을 최대 선택으로 변경 → 본 모델의 학습하고자 하는 타겟은 매칭 코스트를 최소화하는 것 대신 ground truth의 예상 점수(confidence score와 비슷한 개념)를 최대화하는 것을 중점으로 보기 때문. (다시 말해서, 최대한 잘 맞추는 것에 집중함)

-----

## 4. LGA

→ 영상 내의 얇은 구조물과 객체의 가장자리(엣지)를 처리하기 위함.

-----

## 5. Architecture

* 3개의 SGA 레이어와 1개의 LGA 레이어 확인 가능.

![](https://velog.velcdn.com/images/bolero2/post/931c43c5-e1c5-43e5-86bc-b05eab33e412/image.png)

-----

## 6. Experiment

* 실험 대상 데이터셋은 **Scene Flow Dataset**과 **KITTI Benchmarks** 를 사용함.

![](https://velog.velcdn.com/images/bolero2/post/bcfc97e0-f362-4d42-b902-62bd0950dc4d/image.png)

-----

## 7. SGM vs 3D Convolution

* 빨간 선은 Ground Truth와 매칭 지점.

![](https://velog.velcdn.com/images/bolero2/post/d8ecd03f-8285-4c01-a240-52ad0c8cb910/image.png)

* **결론 : SGA + LGA 방식이 noise를 잘 잡음.**
(_"The SGA layers successfully suppress these noise"_)

-----

## 8. Traditional SGA vs GA-Net's SGA

![](https://velog.velcdn.com/images/bolero2/post/95f13d3b-3a6d-47f8-8f59-e4cab925ffe4/image.png)

-----

**(Paper Review는 제가 스스로 읽고 작성한 글이므로, 주관적인 내용임을 밝힙니다.)**
