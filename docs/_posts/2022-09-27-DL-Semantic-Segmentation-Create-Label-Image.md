---
layout: post
title:  "[DL] Semantic Segmentation에서 Label Image 생성하기"
date:   2022-09-27
categories: Deep Learning
---
# Semantic Segmentation에서 Label Image 생성하기

semantic segmentation은 간단하게 보자면 _"**Image**" - "**Image**"_ 관계의 학습입니다.  

X 입력에는 보통 jpg든 png든 읽을 수 있는 Image file이 놓이고,  
Y 입력(정답)에는 보통 png 파일이 옵니다.

이번 포스팅에서는 _**어떻게 하면 라벨에 해당되는 Image file(.png)를 만들 수 있는지?**_ 살펴보겠습니다.

-----

## **0. Intro**  

Semantic Segmentation Task를 학습하는데는 2가지 데이터가 필요합니다:
**_(Input) X Data : 학습할 Image dataset
(Input) Y Data : 학습할 Image에 대응되는 Color Map Image file_**

특히, Y Data 의 경우에 “**JPG**” 포맷이 아닌 “**PNG**” 포맷을 사용하게 됩니다.

JPG 포맷은 손실 압축을 사용하기 때문에, 용량이 작다는 장점이 있지만
_**사용자의 눈에 잡히지 않는 특정 부분의 Color 값이 변경된다는 특징**_이 있습니다.

이번 글에서는 Semantic Segmentation의 Label Image를 생성하는 방법과 일반적인 Image Data 와의 차이점을 살펴보도록 하겠습니다.

-----

## **1. About Semantic Segmentation**
시작하기에 앞서, Semantic Segmentation Task가 정확히 어떤 Task인지 알아야 합니다.
Semantic Segmentation Task의 경우, 
**전체 이미지에 대해 각각의 픽셀이 어느 Label(=Category)에 속하는지 분류하는 문제**입니다.

정교한 분류를 해내야 하기 때문에 **Atrous Convolution**과 같은 **Receptive Field(수용 영역, 필터가 한 번에 볼 수 있는 영역)**가 넓은 합성 곱 연산을 주로 사용합니다.

**DeepLab V3+** 코드 중에서, 가장 중요한 Loss Function 구현 부분을 보도록 하겠습니다.
(Github Repository
: https://github.com/jfzhang95/pytorch-deeplab-xception)

```python
import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
```

위의 DeepLab V3+ Repository에 구현되어 있는 Loss Function 입니다.

여기서 우리는 핵심적인 코어 함수가 **nn.CrossEntropyLoss** 임을 알 수 있습니다.
Cross-Entropy Loss Function(이하 **CE Loss**)은 Semantic Segmentation Task(이하 **분할 문제**)이전에 분류 문제(Classification)에서 자주 쓰이는 손실 함수입니다.

그럼 왜 분류 문제와 분할 문제는 둘다 CE Loss를 쓰는 것일까요?

> 1) 분류 문제(Classification) 에서는 이미지 1장 전체에 대한 Label을 분류합니다.
(ex. 이 사진은 고양이 사진입니다!)
> 
> 2) 분할 문제(Semantic Segmentation) 에서는 이미지 1장 내의 1개 픽셀에 대한 Label을 분류합니다.
(ex. 이 픽셀은 고양이에 해당하는 픽셀입니다!)

위에서 서술 하였듯이, 두 문제 모두 분류를 하긴 하지만 Classification 문제의 경우는 이미지 전체를 분류하고,
**Semantic Segmentation 문제의 경우는 1개 픽셀에 대해서만 분류합니다.**

(분류 문제 신경망에서, Batch Size가 1이라고 가정한다면 단순히 
True-Label 1개와 Predicted-Label 1개를 비교하는 것처럼, 
True-Label’s 1–pixel과 Predicted-Label’s 1-pixel을 비교하는 것입니다.
이는 이미지의 width * height 수만큼 반복됩니다.)

1개 픽셀이라고 한다면?
일반적인 Color Image는 **3채널의 RGB** 값이 들어오지만, 
CE Loss를 사용하는 분할 문제 특성 상 **1개 채널의 값**이 들어오는데 이 값이 바로 **Label Value**가 되는 것입니다.

최종적으로, Semantic Segmentation Task의 학습 방식은

> 1. 1개 픽셀 별로 Label 값을 다르게 준다.(Dataset 측면)
> 2. CE Loss를 통해 손실 값을 구한다.
> 3. Optimizer(SGD, Adam etc.)를 사용하여 해당 손실 값을 backward 방향으로 가중치를 갱신한다.

3단계로 볼 수 있습니다.

-----

## 2. cv2.imwrite( ) vs Pascal VOC Annotation
Label Image 제작에 앞서, 실제로

* 일반적인 이미지 저장 함수 cv2.imwrite를 사용하여 저장한 Label Image
* Pascal VOC의 Semantic Segmentation Task의 Annotation Label Image

를 비교해 보았습니다.
(Pascal VOC Dataset : host.robots.ox.ac.uk/pascal/VOC/voc2007)

**_Mac OS — file command in terminal_**
![file command in terminal](https://images.velog.io/images/bolero2/post/bf3c4130-3bc6-4d4e-bdd8-c013ff4cbeb0/command.png)

위 이미지는 2개의 이미지를 Terminal 상에서 file command로 읽어온 결과입니다.

**_2009_001625.png — RAW Image file_**  
![2009_001625.png](https://images.velog.io/images/bolero2/post/166417ed-1203-4b76-a7c9-563b1ef34d5d/bottle1.png)


**2009_001625.png** 파일은 Pascal VOC Dataset에서 가져온 파일이고,
**2009_001625_RGB.png** 파일은 cv2.imwrite( ) 함수로 저장한 파일입니다.

2009_001625.png(VOC IMAGE)|  2009_001625_RGB.png(cv2.imwrite IMAGE)
:-------------------------:|:-------------------------:
![2009_001625.png](https://images.velog.io/images/bolero2/post/5a20712b-14a2-4374-a4b4-72e31dc1683d/bottle1.png)|![2009_001625_RGB.png](https://images.velog.io/images/bolero2/post/5b4e2800-04c5-4cbb-a3d7-de24198f7da1/bottle1_RGB.png)

명령어 결과를 보면, VOC Image는 8-bit colormap 파일이지만 cv2.imwrite로 저장한 Image는 8-bit/color RGB 파일입니다.
사람이 보기엔, 육안상으로는 어떤 차이가 있을까요?


어떤 이미지가 Pascal VOC인지 모를 정도로 너무 유사합니다…
육안상으로도 전혀 차이점이 없습니다. 
차이점이라고는 위에서 언급한 8-bit colormap이냐, 8-bit/color RGB 파일이냐 차이입니다.
우리는 여기서 PNG 포맷의 특성과 Label Image와의 관계에 대해 알아볼 필요가 있습니다.

-----

## 3. PNG Format _and_ Segmentation Label Image
PNG 포맷을 사용하는 이유는 JPG와 같은 손실압축 방식이 아닌,
원본 그대로의 Color 값을 저장합니다.

그리고 아주 중요한 특징이 하나 더 있는데, 바로
**Palette 정보를 넣을 수 있다는 점**입니다.

Palette 정보가 이미지에 들어가게 되면 Image Array는 더 이상 3채널이어야 할 필요가 없어집니다. 이를

> **“Indexed Image”**

라고 부릅니다. 

말 그대로 **“색인화 된 이미지”** 인 것이죠.

Index 정보는 Palette가 되는 것이고, Image Array에는 단순히 1채널의 공간에 색인 정보(Index value)만 넣어주면 됩니다.
추후에 Palette를 바꾸게 되면, Image의 색상 값도 바뀌게 되는 것입니다.

**Semantic Segmentation은 Pixel에 대한 라벨 값을 학습할 때, 이 Index value를 학습하게 됩니다.**

-----

## 4. _How create_... Label Image?
위에서 
_1) 왜 PNG 포맷을 사용해야 하는지,_
_2) Indexed Image란 무엇인지,_
_3) 왜 cv2.imwrite( )와 같은 일반적인 이미지 저장 함수로 저장하면 안되는지_
알아보았습니다.

이제는 Polygon 타입의 Label Image (for Segmentation)를 제작해보겠습니다.
준비물은 다음과 같습니다:

> 1. Color Map과 Palette 정보
> 2. 저장할 Image 정보(file 형식, numpy.ndarray 형식 모두 상관 없습니다.)

-----

**_Color Map을 생성하는 코드입니다._**
```python
import numpy as np
 
def make_colormap(num=256):
   def bit_get(val, idx):
       return (val >> idx) & 1
 
   colormap = np.zeros((num, 3), dtype=int)
   ind = np.arange(num, dtype=int)
 
   for shift in reversed(list(range(8))):
       for channel in range(3):
           colormap[:, channel] |= bit_get(ind, channel) << shift
       ind >>= 3
 
   return colormap
 
cmap = make_colormap(256).tolist()
palette = [value for color in cmap for value in color]
print(cmap, "\n", palette)
```

위 코드는 Color Map을 생성하는 코드입니다.

Pascal VOC에서 해당 방식으로 Color Map을 생성하여 [20개 라벨 + background] 까지 하여 총 21개의 색상 값을 사용합니다.

우리가 저 코드에서 사용할 변수는 **cmap** 과 **palette** 가 있습니다.
**cmap** 은 Image에 색인 정보를 넣어줄 때 사용할 것이고, 
**palette** 는 PNG 포맷으로 저장할 때 넣어줄 팔레트 정보입니다.

해당 코드 동작 결과는 다음과 같습니다:
> [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, … ]

보시다시피 앞에서 3개씩 끊어서 볼 수 있습니다.

예를 들어, 0번 라벨(background)의 경우 RGB 값은 [0, 0, 0]이 될 것이고,
1번 라벨(aeroplane)의 경우 RGB 값은 [128, 0, 0]이 될 것이며, 
2번 라벨(bicycle)의 경우 RGB 값은 [0, 128, 0]이 될 것입니다.

(이는 사용자가 직접 값을 넣어줘도 상관없습니다. 본문에서는 Pascal VOC Dataset을 사용하여 실험하였기 때문에, Pascal VOC Dataset의 Category 정보와 색상 정보를 사용하였습니다.)

-----

Color Map과 Palette 정보를 생성하였다면, 이미지에 색인 정보와 Palette를 함께 넣어주기만 하면 됩니다.

_**Label Image 생성 코드**_
```python
import cv2
import numpy as np
from PIL import Image


# Image data to save = image_data(numpy.ndarray)
label_img = np.array(image_data)
 
# if image array has BGR order
label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
# Create an unsigned-int (8bit) empty numpy.ndarray of the same size (shape)
img_png = np.zeros((label_img.shape[0], label_img.shape[1]), np.uint8)
 
# Assign index to empty ndarray. Finding pixel location using np.where.
# If you don't use np.where, you have to run a double for-loop for each row/column.
for index, val_col in enumerate(cmap):
    img_png[np.where(np.all(label_img == val_col, axis=-1))] = index
 
# Convert ndarray with index into Image object (P mode) of PIL package
img_png = Image.fromarray(img_png).convert('P')
# Palette information injection
img_png.putpalette(palette)
# save image
img_png.save('output.png')
```

위 코드는 실제로 Label Image를 생성하는 코드입니다.

순서는 다음과 같습니다:
> 1. image data를 numpy.ndarray 타입으로 호출합니다.
> 2. BGR 순서라면 RGB 순서로 바꿔줍니다.
> 3. 동일한 크기의 unsigned-int (8bit) ndarray를 생성합니다.
> 4. **각 Color Map의 색상 정보를 찾아서, 색인화 과정을 합니다.**
(이미지의 픽셀을 돌며 비교하는 것이 아닌, Color Map의 1-Row에 해당하는 값을 한번에 바꿔주는 형식으로 합니다.)
> 5. 색인화 된 이미지 행렬을 'P' mode로 저장합니다.(**P** mode는 **Palette** 모드입니다.)
> 6. 위에서 생성한 Palette를 넣어줍니다.
> 7. png format으로 이미지를 저장합니다.

이렇게 하면 Palette 정보가 있는 색인화 된 이미지 파일을 생성할 수 있습니다.

이렇게 생성된 이미지 파일은 바로 Semantic Segmentation의 학습에 사용 가능하며, Predict 함수에서 출력된 결과를 저장할 때도 이러한 방식으로 저장하여 성능 측정이 가능합니다.

-----

## 5. Result
1. Semantic Segmentation의 Label Image 생성 시, 일반적인 이미지 저장 방식으로 저장하면 안됩니다.
2. Color Map과 Palette 정보가 포함된, Indexed Image를 제작해야 합니다.
3. 그 이유는 Segmentation 학습이 CE Loss를 사용하는데, 여기서 1-Pixel에 대해 1개의 값(=label value)을 비교하기 때문입니다.
4. 사용자가 Augmentation 혹은 Annotation 시에, 특정 Color Map을 생성 후에 Pixel의 Label Value에 색인화 시켜주는 작업이 필요합니다. 
(상단 소스코드 참조)