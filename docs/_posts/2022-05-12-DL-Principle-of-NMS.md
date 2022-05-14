---
layout: post
title:  "[DL] NMS, Non-Maximum Suppression 파헤치기"
date:   2022-05-12
categories: DeepLearning
description: "Object Detection에서 사용하는 NMS(Non-Maximum Suppression, 비 최대 억제 알고리즘)에 대해 알아보고, 간단한 코드를 작성하여 실습해봅니다."
image: '/img/thumbnail-nms.jpg'
published: true
---

{% include tag.html tag="DeepLearning" %}
{% include tag.html tag="ObjectDetection" %}

## 0. Intro

NMS란? Non-maximum Suppression의 약자로써, **비최대 억제 알고리즘**으로 생각하면 된다.
말 그대로, 최대가 아닌 박스들(=Bounding Box)을 삭제하는 알고리즘이다.

**Object Detection** Task에서 객체에 대한 최종 Bounding Box를 결정지을 때 사용한다.

-----

## 1. Principle

작동 방식은 다음과 같다.

> 1. Prediction할 이미지를 Network에 Forward 시킨다.
> 2. 출력 가능한 모든 박스를 구한다. (NMS 이전이므로, 수십~수백개의 박스 정보가 나온다.)
> 3. 해당 박스 정보들을 **Confidence Score** 순으로 정렬한다.
> 4. 박스 정보들의 **Confidence Score**와, 각 박스 간의 **IoU**를 바탕으로 하여 NMS를 적용한다.
> 5. NMS 적용 후에는, 알맞는 박스라고 여겨지는 값들만 출력된다.

핵심 키워드를 뽑자면, **Confidence Score**와 **IoU**라고 생각한다. (이 두 단어로 구현이 가능함.)

### 1-1. Confidence-Score?

여기서 **Confidence Score(신뢰도, 신뢰 점수)**라고 하는 것은
**네트워크가 정답을 도출해냈을 때, 그 정답에 대해 n%의 확신도를 갖는다는 의미**이다.

예를 들어, A 박스에 대한 Confidence Score가 0.75라면,
네트워크가 생각했을 때 "아, 이 박스를 내가 도출하긴 했는데, 이 박스가 정답일 확률은 **75%** 정도야." 라는 의미이다.

### 1-2. IoU?

IoU는 **Intersection Over Union**의 약자이다.
![iou.png](https://images.velog.io/images/bolero2/post/93cd3e8c-b77f-4de9-a3f4-d7683c4e6838/iou.png)

(Object Detection 을 다뤄본 개발자라면 많이 접해봤을 듯한 그림.)

쉽게 말해서, 두 박스의 **교집합** / 두 박스의 **합집합** 이다.
코드로 보자면 다음과 같다.
```python
def iou(box1, box2):
    def _is_box_intersect(box1, box2):
        if (
            abs(box1[0] - box2[0]) < box1[2] + box2[2]
            and abs(box1[1] - box2[1]) < box1[3] + box2[3]
        ):
            return True
        else:
            return False

    def _get_area(box):			# area of box n.
        return box[2] * box[3]

    def _get_intersection_area(box1, box2):
    # intersection area
        return abs(max(box1[0], box2[0]) - min(box1[0] + box1[2], box2[0] + box2[2])) * abs(
            max(box1[1], box2[1]) - min(box1[1] + box1[3], box2[1] + box2[3])
        )
    def _get_union_area(box1, box2, inter_area=None):
        area_a = _get_area(box1)
        area_b = _get_area(box2)
        if inter_area is None:
            inter_area = _get_intersection_area(box1, box2)

        return float(area_a + area_b - inter_area)    

    # if boxes do not intersect
    if _is_box_intersect(box1, box2) is False:
        return 0
        
    inter_area = _get_intersection_area(box1, box2)
    union = _get_union_area(box1, box2, inter_area=inter_area)
    
    # intersection over union
    iou = inter_area / union
    if iou < 0:
        iou = 0
    assert iou >= 0, f"Measure is wrong! : IoU Value is [{iou}]."
    return iou
```

box1과 box2가 입력으로 왔을 때, 두 박스의 IoU를 계산해주는 코드이다.
box의 좌표 체계는 **[center_x, center_y, width, height]** 이고, **상대 좌표** 이다.
(YOLO의 라벨링 방법이라고 생각하면 된다.)

-----

## 2. Sample Code

코드로 살펴보자.
임의로 박스 정보들을 생성해주고, Confidence Score 값도 넣어주었다.

```python
colorset = [
        (0, 0, 255),		# Red
        (0, 255, 0),		# Green
        (255, 0, 0),		# Blue
        (255, 255, 0),		# Cyan
        (255, 0, 255),		# Magenta
        (0, 255, 255)		# Yellow
]

width, height = 600, 600

boxes = [
    # left sector boxes
    [0.3, 0.3, 0.1, 0.1, 0.9],        # Red
    [0.31, 0.28, 0.14, 0.13, 0.5],    # Green
    [0.28, 0.28, 0.09, 0.11, 0.3],    # Blue

    # right sector boxes
    [0.75, 0.65, 0.2, 0.2, 0.99],     # cyan
    [0.7, 0.63, 0.22, 0.18, 0.35],    # magenta
    [0.75, 0.62, 0.22, 0.22, 0.77],   # yellow
]
```
**박스 정보는 [center_x, center_y, width, height, conf_score] 순으로 구성되어 있고,
절대 좌표이다. (0~1 사이로 스케일링 된 값)**

해당 박스들을 600 * 600의 빈 캔버스에 그려보면, 다음과 같은 박스를 볼 수 있다.

![before_nms.jpg](https://images.velog.io/images/bolero2/post/c166b00b-67ec-4c8c-a644-40a47109769e/before_nms.jpg)

**(박스 그려주는 코드)**
```python
canvas = np.zeros((width, height, 3)).astype('uint8')			# empty canvas
canvas_copy = canvas.copy()						# for after nms

for index, box in enumerate(boxes):
    cv2.rectangle(canvas, (int(width * box[0] - width * box[2]), int(height * box[1] - height * box[3])),
                          (int(width * box[0] + width * box[2]), int(height * box[1] + height * box[3])),
                          colorset[index], 2)
```

임의로 만든 박스들이기 때문에, 우리는 어떤 것이 NMS 통과 후에 남아야 하는지 알 수 있다.
바로 **Red**와 **Cyan** 색상의 박스만 남아야 한다.
_(이유는, 좌/우측 섹터 기준으로 신뢰 점수가 가장 높기 때문이다.)_

그리고, **NMS(boxes, iou_thres=0.4)** 함수를 작성해주었다. (중간에 출력을 디버깅해볼 수 있는 print문이 있다.)
```python
def nms(boxes, iou_thres=0.4):
    elems = np.array(boxes)
    print("\nBefore Arrange")
    print(elems)

    # sorting
    sorted_index = np.argsort(elems[:, -1])[::-1]
    sorted_boxes = elems[sorted_index]

    print("\nAfter Arrange")
    print(sorted_boxes)

    answer = [True for x in range(sorted_boxes.shape[0])]
    print("\nBefore NMS Answer :", answer)

    for i in range(sorted_boxes.shape[0]):
        if answer[i] is False:
            continue
        for j in range(sorted_boxes.shape[0]):
            iou_val = iou(sorted_boxes[i], sorted_boxes[j])
            print(f"{i} vs {j} = iou {round(iou_val, 3)}")
            if iou_val >= iou_thres and int(iou_val) != 1:
                answer[j] = False
                print(f"Index {j} is False.")

    print("\nAfter NMS Answer :", answer)

    return answer, sorted_boxes, sorted_index
```

순서는 **1. 원리**에서 본 것과 동일하다.

```python
sorted_index = np.argsort(elems[:, -1])[::-1]
sorted_boxes = elems[sorted_index]
```
여기 부분에서 Confidence Score 순으로 box 정보들을 정렬하고,

```python
for i in range(sorted_boxes.shape[0]):
    if answer[i] is False:
        continue
    for j in range(sorted_boxes.shape[0]):
        iou_val = iou(sorted_boxes[i], sorted_boxes[j])
        print(f"{i} vs {j} = iou {round(iou_val, 3)}")
        if iou_val >= iou_thres and int(iou_val) != 1:
            answer[j] = False
            print(f"Index {j} is False.")
```
부분에서 실제로 IoU 값을 구하면서, 탈락 여부를 결정한다.

IoU Threshold를 0.4로 설정하였기 때문에,
박스 간의 IoU 값이 0.4 이상인 경우에는, **Confidence Score가 낮은 박스가 탈락한다.**

탈락 여부는 ```answer = [True for x in range(sorted_boxes.shape[0])]``` 로 미리 박스 개수만큼 True를 적어두고,
탈락된 박스 자리에 ```False```를 기록한다.

-----

디버깅 출력물은 다음과 같다.

- ** 정렬 이전의 box 정보들**
```python
Before Arrange
[[0.3  0.3  0.1  0.1  0.9 ]
 [0.31 0.28 0.14 0.13 0.5 ]
 [0.28 0.28 0.09 0.11 0.3 ]
 [0.75 0.65 0.2  0.2  0.99]
 [0.7  0.63 0.22 0.18 0.35]
 [0.75 0.62 0.22 0.22 0.77]]
```

- **정렬 이후의 box 정보들** (confidence score 순서로 잘 정렬되었다.)
```python
After Arrange
[[0.75 0.65 0.2  0.2  0.99]
 [0.3  0.3  0.1  0.1  0.9 ]
 [0.75 0.62 0.22 0.22 0.77]
 [0.31 0.28 0.14 0.13 0.5 ]
 [0.7  0.63 0.22 0.18 0.35]
 [0.28 0.28 0.09 0.11 0.3 ]]
```

- **Answer라는 변수를 두고, 탈락 여부를 위한 리스트를 생성했다.**
```python
Before NMS Answer : [True, True, True, True, True, True]
```

- **NMS 동작 과정, 탈락하면 Answer[index]에 False를 기록한다.**
```python
0 vs 0 = iou 1.0
0 vs 1 = iou 0
0 vs 2 = iou 0.754
Index 2 is False.
0 vs 3 = iou 0
0 vs 4 = iou 0.519
Index 4 is False.
0 vs 5 = iou 0
1 vs 0 = iou 0
1 vs 1 = iou 1.0
1 vs 2 = iou 0
1 vs 3 = iou 0.469
Index 3 is False.
1 vs 4 = iou 0
1 vs 5 = iou 0.463
Index 5 is False.
```
: 원래는 자기 자신과는 비교하면 안되는데, 그냥 비교하게 짰다. (심플하게)
그 부분 처리를 위해 코드에 ```int(iou_val) != 1```인 조건을 추가하였다.
**(iou_val이 1이라는 뜻은, 자기 자신과 비교한 경우이다.)**

이제 보면, 2개의 경우가 있다.
- **IoU가 0.4보다 작을 경우**
- **IoU가 0.4보다 클 경우**

1) 작을 경우는, **살아남는 박스**이다.
2) 클 경우는, **탈락하는 박스**이다.
(0.9 conf의 박스와 0.4 conf의 박스를 비교하여 iou_val == 0.5라고 한다면, 0.4 conf의 박스가 탈락한다.)

- **NMS 이후에 살아남는 박스의 index, Answer 변수에 boolean 값으로 알 수 있다.**
```python
After NMS Answer : [True, True, False, False, False, False]
```

return 되는 값은, Answer 변수와 sorted_boxes, sorted_index를 반환해주었다.
main 문에서 박스를 잘 볼 수 있게 하기 위함이다.

-----

## 3. Result

main 문에서 다음과 같이 NMS 함수를 호출한다.
```python
answer, sorted_boxes, sorted_index = nms(boxes, iou_thres=0.4)
```
**여기서 boxes는 before Arrange 박스들이다.**

그리고, answer과 sorted_boxes, sorted_index 정보를 바탕으로,
새로운 비어있는 canvas에 그려준다.
```python
for index, datum in enumerate(zip(sorted_boxes, sorted_index)):
    sbox, sidx = datum
    if answer[index] is True:
        cv2.rectangle(canvas_copy, (int(width * sbox[0] - width * sbox[2]), int(height * sbox[1] - height * sbox[3])),
                                   (int(width * sbox[0] + width * sbox[2]), int(height * sbox[1] + height * sbox[3])),
                                   colorset[sidx], 2)
```
**(Answer 변수가 _True_ 인 box만 그려준다.)**

그러면, 다음과 같이 Red와 Cyan 색상의 박스만 남은 것을 알 수 있다.
![after_nms.py](https://images.velog.io/images/bolero2/post/509a537d-77a4-41c6-8c59-b5bb0fa6e2a0/after_nms.jpg)

-----

## 4. Full Code
```python
import cv2
import numpy as np


def iou(box1, box2):
    def _is_box_intersect(box1, box2):
        if (
            abs(box1[0] - box2[0]) < box1[2] + box2[2]
            and abs(box1[1] - box2[1]) < box1[3] + box2[3]
        ):
            return True
        else:
            return False

    def _get_area(box):			# area of box n.
        return box[2] * box[3]

    def _get_intersection_area(box1, box2):
    # intersection area
        return abs(max(box1[0], box2[0]) - min(box1[0] + box1[2], box2[0] + box2[2])) * abs(
            max(box1[1], box2[1]) - min(box1[1] + box1[3], box2[1] + box2[3])
        )
    def _get_union_area(box1, box2, inter_area=None):
        area_a = _get_area(box1)
        area_b = _get_area(box2)
        if inter_area is None:
            inter_area = _get_intersection_area(box1, box2)

        return float(area_a + area_b - inter_area)    

    # if boxes do not intersect
    if _is_box_intersect(box1, box2) is False:
        return 0
        
    inter_area = _get_intersection_area(box1, box2)
    union = _get_union_area(box1, box2, inter_area=inter_area)
    
    # intersection over union
    iou = inter_area / union
    if iou < 0:
        iou = 0
    assert iou >= 0, f"Measure is wrong! : IoU Value is [{iou}]."
    return iou


def nms(boxes, iou_thres=0.4):
    elems = np.array(boxes)
    print("\nBefore Arrange")
    print(elems)

    # sorting
    sorted_index = np.argsort(elems[:, -1])[::-1]
    sorted_boxes = elems[sorted_index]

    print("\nAfter Arrange")
    print(sorted_boxes)

    answer = [True for x in range(sorted_boxes.shape[0])]
    print("\nBefore NMS Answer :", answer)

    for i in range(sorted_boxes.shape[0]):
        if answer[i] is False:
            continue
        for j in range(sorted_boxes.shape[0]):
            iou_val = iou(sorted_boxes[i], sorted_boxes[j])
            print(f"{i} vs {j} = iou {round(iou_val, 3)}")
            if iou_val >= iou_thres and int(iou_val) != 1:
                answer[j] = False
                print(f"Index {j} is False.")

    print("\nAfter NMS Answer :", answer)

    return answer, sorted_boxes, sorted_index

if __name__ == "__main__":
    colorset = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (0, 124, 0),
        (0, 0, 124),
    ]

    width, height = 600, 600

    boxes = [
        # left boxes
        [0.3, 0.3, 0.1, 0.1, 0.9],                  # Red
        [0.31, 0.28, 0.14, 0.13, 0.5],              # Green
        [0.28, 0.28, 0.09, 0.11, 0.3],              # Blue

        # right boxes
        [0.75, 0.65, 0.2, 0.2, 0.99],               # cyan
        [0.7, 0.63, 0.22, 0.18, 0.35],              # magenta
        [0.75, 0.62, 0.22, 0.22, 0.77],             # yellow
    ]

    canvas = np.zeros((width, height, 3)).astype('uint8')
    canvas_copy = canvas.copy()

    for index, box in enumerate(boxes):
        cv2.rectangle(canvas, (int(width * box[0] - width * box[2]), int(height * box[1] - height * box[3])),
                              (int(width * box[0] + width * box[2]), int(height * box[1] + height * box[3])),
                              colorset[index], 2)

    answer, sorted_boxes, sorted_index = nms(boxes, iou_thres=0.4)

    for index, datum in enumerate(zip(sorted_boxes, sorted_index)):
        sbox, sidx = datum
        if answer[index] is True:
            cv2.rectangle(canvas_copy, (int(width * sbox[0] - width * sbox[2]), int(height * sbox[1] - height * sbox[3])),
                                       (int(width * sbox[0] + width * sbox[2]), int(height * sbox[1] + height * sbox[3])),
                                       colorset[sidx], 2)

    # print(canvas)
    cv2.imshow("Before NMS", canvas)
    cv2.imshow("After NMS", canvas_copy)

    cv2.imwrite("before_nms.jpg", canvas)
    cv2.imwrite("after_nms.jpg", canvas_copy)
    cv2.waitKey(0)
```