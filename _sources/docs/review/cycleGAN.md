```{admonition} Information
- **Title:** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (ICCV 2017)

- **Reference**
    - Paper: [https://arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593)
    - Code: [TensorFlow CycleGAN tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan?hl=ko)
    - [[논문리뷰] Cycle GAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://velog.io/@sjinu/CycleGAN)
    [CycleGAN을 만든 사람이 한국인이라고? CycleGAN 논문 뜯어보기](https://comlini8-8.tistory.com/9)

- **Author:** KwangSu Mun

- **Last updated on Apr. 12, 2023**
```

# CycleGAN

## Abstract

-   Image-to-image translation(이하 translation)은 한 이미지 도메인을 다른 이미지 도메인으로 변환시키는 computer vision의 한 task.
-   translation은 보통 input과 output이 짝이 지어진 상태에서 학습. 하지만 짝이 지어진 학습 데이터를 얻는 것이 어렵습니다. 따라서 cycleGAN 논문에서는 짝지어진 예시 없이 X라는 domain으로부터 얻은 이미지를 target domain Y로 바꾸는 방법을 제안. 이 연구는 Adversarial loss를 활용해, G(x)로부터 생성된 이미지 데이터의 분포와 Y로부터의 이미지 데이터의 분포가 구분이 불가능하도록 "함수 G:X -> Y"를 학습시키는 것을 목표로 합니다. X --> Y로의 mapping에 제약을 가해서 원하는 이미지를 강제하기 위해 F: Y -> X와 같은 역방향 매핑을 함께 진행하고, F(G(x))가 X와 유사해지도록 강제하는 Cycle consistency loss를 도입했습니다.
-   결과적으로 collection style transfer, object transfiguration, season transfer, photo enhancement 등의 task에서 이미지 pair가 존재하지 않는 상태에서 우수한 결과를 보여줬다고 합니다.


## Introduction

### 참고) Image-to-Image translation이란?

:::{figure-md} 
<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" class="bg-primary mb-1" width="800px"/>

image-to-image translation
:::


Image-to-image translation은 input image를 다른 스타일, 속성, 구조 등을 가진 output image로 변환하는 것입니다. 예를 들어 사진을 그림으로 변환한다거나, 낮에 찍은 사진을 밤에 찍은 것 처럼 변환하는 것을 말합니다. 흔히 translation은 input과 output으로 짝이 지어진 data를 바탕으로 학습이 이루어져 있었는데요. 짝이 지어진 사진 데이터를 얻는 것은 어렵고 값이 비싼 일이 됩니다.

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbhMGUZ%2Fbtr7HimHXN5%2FHvjTh02iCzP5Sgk8UYkKO0%2Fimg.png" class="bg-primary mb-1" width="800px">

paired and unpaired data
:::

 이 논문에서는 input image와 output image가 일대일로 짝지어지지 않은 상태에서 하나의 image 모음의 특성을 캡쳐하고, 이러한 특성을 다른 image 모음으로 변환할 수 있는 방법을 제시합니다.  
GAN은 domain X에 이미지 한 세트, domain Y에 이미지 한 세트가 제공되고, model의 output과, Y가 discriminator에 의해 구별할 수 없도록 G:X->Y를 학습합니다. 하지만, 이게 개별 입력 x와 출력 y가 무조건 유의미하게 쌍을 이룬다는 것을 뜻하지는 않습니다. G가 생성할 수 있는 image에는 무한한 경우의 수가 있기 때문. 종종 mode collapse가 일어나기도 합니다.

### mode collapse란?

:::{figure-md} 
<img src="https://1.bp.blogspot.com/-oDCR5UnEIl4/WZkIId-rYCI/AAAAAAAAAJk/PoLvou4JLNIxn5U-OmPFZ_heyxVQGbMNQCEwYBhgL/s1600/14.png" class="bg-primary mb-1" width="800px">

mode collapsing 출처: http://dl-ai.blogspot.com/2017/08/gan-problems.html
:::

-   어떤 input image든 모두 같은 output image로 매핑하면서 최적화에 실패하는 현상. 이 현상은 generator 입장에서, Discriminator가 이 사진이 진짜 Y인지 가짜인 Y^인지 구별하는 것을 '**속이기만**' 하면 되기 때문에 우리의 목적과 전혀 상관이 없는 데이터를 generator가 만들더라도 문제가 생기지 않아서 발생함
-   참고: [http://dl-ai.blogspot.com/2017/08/gan-problems.html](http://dl-ai.blogspot.com/2017/08/gan-problems.html)

이러한 이슈로 인해 추가 objective function이 필요해 졌습니다. 따라서 translation task는 영어 -> 프랑스어 -> 영어로 번역했을 때 원래 문장에 다시 도달하는 것처럼, X --> Y --> X'로 돌아가는 과정에서 X와 X'가 최대한 같아야 한다는 의미의 cyclic consistency이라는 속성을 이용합니다. 필요한 목적식을 간단하게 정리하면 다음과 같습니다.

-   정방향, 역방향 adversarial Loss(X -> Y & Y -> X)
-   Cycle consistency loss: X ~= F(G(x))


## Related work(관련 연구)

-   GAN
-   Image-to-Image Translation
-   Unpaired Image-to-Image Translation
-   Cycle Consistency
-   Neural Style Transfer

논문과 관련된 기존 연구에 대한 내용이었음. 관련 중요한 개념들은 위 introduction에서 설명했고, 나머지는 cycleGAN 스터디와는 딱히 관련이 없어 보여서 스킵했음.


## Formulation

:::{figure-md} 
<img src="../../pics/cyclegan/fig2.png" class="bg-primary mb-1" width="800px">

cycleGAN 도식화 자료
:::

-   목표: X, Y를 mapping하는 function을 학습하는 것
-   용어 정리

1.  data 분포를 x ~ pdata(x), y ~ pdata(y)로 표시
2.  G : X -> Y, F: Y -> X
3.  Dx, Dy는 discriminator
4.  Dx는 X와 F(y)를 구분, Dy는 y와 G(x)를 구분. 목적식은 총 두개
    -   adversarial loss: 생성된 이미지의 분포를 대상 domain의 data distribution과 일치시키기 위한 것.
    -   cycle consistency loss: 학습된 mapping G와 F가 서로 모순되는 것을 방지하기 위한 것.

### Adversarial loss

G: X --> Y와 Dy에 대한 목적식은 다음과 같음.

:::{figure-md} L_GAN Loss function
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FnvzuE%2Fbtr725OfuJy%2FI1IgwK5PIzXpzINWnJxysK%2Fimg.png" alt="L_GAN Loss function" style="width:800px">

L_GAN Loss function (source: https://arxiv.org/abs/1703.10593)
:::

-   GAN에서 쓰이는 loss function과 동일. 대신에 X -> Y로 갈 때와 Y -> X로 갈 때 총 두개의 수식이 나오며, F:Y->X와 Dx에 대해서도 F, Dx를 넣은, 같은 수식을 사용함.

### Cycle consistency Loss

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FzsgD6%2Fbtr8ay8PEBE%2F3mAKd1YSAiCK4ZXeIg84s1%2Fimg.png" class="bg-primary mb-1" width="600px">

cycle consistency loss result
:::

-   앞서 말했듯, mapping distribution에 제한을 두어 최대한 우리가 원하는 이미지를 생성하기 위해 사용하는 수식으로서, 위와 같음.
-   예비 실험에서 L1 norm을 adversarial loss로 대체해봤는데, 성능 향상을 관찰할 수 없었음.
-   cycle consistency loss를 통해 유도된 결과는 아래 그림에서 볼 수 있었음.

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fmq8pC%2Fbtr724Pl3Q2%2FUSK4TDRaUK860iIdvG0vV0%2Fimg.png" class="bg-primary mb-1" width="600px">

cycle consistency loss function
:::

### full objective - 전체 목적식

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUyaOu%2Fbtr724Pl3Rj%2FigjKaeukv5m8Cbdzulp5jK%2Fimg.png" class="bg-primary mb-1" width="600px">

full objective function
:::

-   이 때 consistency loss 앞에 붙은 가중치 (lambda)는 GAN Loss와의 상대적 중요도에 따라 결정됨.


## Implementation

baseline architecture로서 neural style transfer와 super-resolution에 인상적인 결과를 보여준 논문에서 사용된 구조를 채택함.

-   3개의 convolutions and several residual blocks,
-   fractionally-strided convolution with stride 1/2,
-   feature를 RGB로 매핑하는 one convolution layer.
-   6 blocks for 128 x 128 image // 9 blocks for 256 x 256 및 고해상도 학습 image.
-   instance normalization

### Training details

모델 학습을 안정화시키기 위해 아래와 같은 테크닉을 추가로 적용합니다.

-   GAN의 Loss function에서 nll loss를 least-squared loss로 변경
-   생성된 이미지 중 가장 최근의 50개를 따로 저장해 discriminator가 이를 한꺼번에 분류(모델 진동을 최소화하기 위함)

### least-square loss 추가 설명

참고)

-   [https://velog.io/@sjinu/CycleGAN](https://velog.io/@sjinu/CycleGAN)
-   [https://ysbsb.github.io/gan/2022/02/23/LSGAN.html](https://ysbsb.github.io/gan/2022/02/23/LSGAN.html)

사용 이유: Generator의 업데이트를 위해서(LSGAN을 참고)

-   이해는 못했고, 이런게 있구나 정도로만 알 수 있었음.

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F6JIT8%2Fbtr73nVyIqs%2FKfcPK33U3OY0AjKhjFlUh1%2Fimg.png" class="bg-primary mb-1" width="800px">

출처: https://velog.io/@sjinu/CycleGAN
:::

(원래 Discriminator는 이보다 더 고차원이지만) 간략히 2차원을 표방하면 결정경계를 위와 같이 나타낼 수 있습니다. 윗 쪽이 가짜 영역, 아래 쪽이 진짜 영역입니다 이 때, 아래에 보면 진짜 데이터 샘플과 거리가 먼 가짜 데이터 샘플이 존재합니다. 즉, NLL Loss를 사용한다면, Generator의 입장에서는 이미 Discriminator를 잘 속이고 있기 때문에 학습할 필요가 없습니다. 즉, Vanishing Gradient가 일어나기 때문에, Discriminator를 잘 속인다는 이유만으로, 안 좋은 샘플을 생성하는 것에 대해 패널티를 줄 수가 없게 됩니다. 이 때, LS GAN을 사용한다면 실제 데이터 분포와 가짜 데이터 샘플이 거리가 먼 것에 대해서도 패널티를 주게 됩니다.

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHsUiX%2Fbtr77PQw99h%2F0Er06IYIGYlBGw2rVufXc0%2Fimg.png" class="bg-primary mb-1" width="800px">

출처: https://velog.io/@sjinu/CycleGAN
:::

-   Generator는 Discriminator를 속이는 것을 넘어서, 실제 데이터 분포와 유사한 분포를 가지게끔 해야합니다.

### 기타

-   모든 실험에서 람다를 10으로 설정했다.
-   batch size == 1, 아담을 사용했다.
-   모든 네트워크는 learning rate를 0.0002로 사용했다. 첫 100 에포크 동안에는 같은 ln을 사용했고, 다음 100 에포크마다 0으로 조금식 수렴하게 했다.


## Result

모델 성능 평가를 위해 아래와 같은 세 개의 지표를 사용.

1.  AMT perceptual studies: 참가자들은 실제 사진이미지 vs 가짜 이미지, 또는 지도 이미지 vs 가짜이미지에 노출된 후 진짜라고 생각되는 이미지를 선택하게 함.
2.  FCN Score: 1번 study가 테스트에 있어 매우 좋은 기준임에도 불구하고, 사람을 대상으로 한 실험이 아닌, 양적인 기준을 찾았는데, FCN score임. FCN은 생성된 사진에 대한 레이블 맵을 예측합니다. 이 레이블 맵은 아래에서 설명하는 표준 시맨틱 분할 메트릭을 사용하여 input ground truth label과 비교할 수 있다. "도로 상의 자동차"라는 label에서 사진 이미지를 생성하면, 생성된 이미지에 적용된 FCN이 "도로 상의 자동차"를 감지하면 성공한 것입니다.
3.  사진 --> 라벨링 성능을 평가: pixel당 정확도, class 당 정확도, IoU(Intersection-Over-Union)을 포함하는 cityscapes benchmark의 표준 metric

### Baseline

-   coGAN, SimGAN, pix2pix

### Comparison against baselines

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcZUe4E%2Fbtr8eXUQ6ou%2FikWglP8dEglGUny4dRkMjK%2Fimg.png" class="bg-primary mb-1" width="800px">

Comparison aginst baselines
:::

figure 5, figure 6에서 볼 수 있듯이 어떤 baseline에서도 강력한 결과를 얻을 수 없었음. 반면에 cycleGAN은 fully supervise인 pix2pix와 비슷한 품질의 translation을 생성할 수 있음.

### Human study

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb1Zhnx%2Fbtr8eWhk9ID%2FtauuT1N0W2qxRekj3IAnc1%2Fimg.png" class="bg-primary mb-1" width="600px">

AMT score
:::

표 1은 AMT perceptual realism task에 대한 성능을 나타냄. 여기서 지도에서 항공 사진, 항공 사진에서 지도 모두에서 약 1/4의 참가자를 속일 수 있었음. 그 외 모든 baseline은 참가자를 거의 속일 수 없었다.

### FCN 등

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqzYO1%2Fbtr728xs5iD%2FN5NDNYwUYLnEZfnOVYONM0%2Fimg.png" class="bg-primary mb-1" width="600px">

FCN scores
:::

표 2는 도시 풍경에 대한 label --> photo task의 성능을 평가하고 표 3은 반대 매핑을 평가함. 두 경우 모두 cycleGAN이 baseline들의 성능을 능가한다.

### Analysis of the loss function

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcjQ9QQ%2Fbtr79farEX8%2FkQ6SWARw9QK9jqRqHlZoi1%2Fimg.png" class="bg-primary mb-1" width="600px">

Analysis of loss function
:::

GAN, cycle consistency의 중요성을 보여주는 자료.  
table 4, table 5에서 볼 수 있음. GAN을 없애면 cycle을 제거하는 것처럼 결과가 크게 저하됨. 따라서 두 term 모두 결과에 중요하다고 결론을 내릴 수 있음. 또한 한 방향에서만 cycle loss를 통해 각 메소드를 평가함. GAN + forward cycle만 돌렸을 때와, GAN + backward cycle만 돌렸을 때 이따금씩 학습에 불안정성을 보이고, mode collapse를 유발하는 것을 발견함(특히 제거된 매핑의 방향에 대해서 그런 경향을 보임). 그림 7을 보면 그런 경향을 볼 수 잇었음.

### Image reconstruction quality

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fyy7lt%2Fbtr73PdbuJp%2F5bmDtKSlQJJnd5yKvPgfB1%2Fimg.png" class="bg-primary mb-1" width="600px">

cycle consistency result
:::

그림 4에서 재구성된 이미지의 몇가지 무작위 샘플을 보여줌. 지도 --> 항공 사진과 같이 하나의 도메인이 훨씬 더 다양한 정보를 나타내는 경우에도 재구성된 이미지가 훈련 및 테스트 시간 모두 원래 입력 x에 가까운 경우가 많았음.

### paired dataset에 대한 추가 결과

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbqNrhb%2Fbtr72YaInQa%2Fk8b4K99KrAsD9C0SHINtt1%2Fimg.png" class="bg-primary mb-1" width="600px">

compare with paired dataset
:::

그림 8은 CMP Façade Database의 건축 레이블 <--> 사진, UT Zapoos50K dataset의 edge <--> 신발과 같이 pix2pix에 사용된 다른 paired dataset에 대한 몇 가지 예시 결과를 보여줌. cycleGAN의 이미지 품질은 fully supervised pix2pix에 대의 생성된 것과 비슷하지만 cycleGAN은 paired supervision 없이 학습이 된다.(우리가 짱이다!)


## Applications
- ** 이미지가 너무 많아 이미지는 생략하겠습니다.ㅠ**
- paired data가 없는 상태에서 의 application 예시. traning data에서 transslation이 test data에서 한것보다 더 매력적이다. training and test data에 대한 application은 웹사이트에 있다.

### Collection style transfer


신경 스타일 전달"\[13\]에 대한 최근 작업과 달리, 우리의 방법은 선택한 단일 예술 작품의 스타일을 전달하는 대신 전체 예술 작품 컬렉션의 스타일을 모방하는 방법을 학습합니다. 그래서 '별이 빛나는 밤에'처럼 그리는 것 보다 '반 고흐'를 따라하는 느낌을 따라한다.

### Object transfiguration


Turmukhambetov et al. \[50\] 하나의 객체를 동일한 범주의 다른 객체로 변환하는 부분 공간 모델을 제안하는 반면, 우리의 방법은 시각적으로 유사한 두 범주 사이의 객체 변형에 중점을 둡니다.  
Turning a horse video into a zebra video (by CycleGAN)

### season transfer


### Photo generation from paintings \*\*


그림을 사진으로 바꿀 때, 입력과 출력 간 색 구성을 보존하기 위해 추가적인 loss를 도입하는 것이 유용하다는 것을 발견할 수 있습니다. 특히, Taigman et al. \[49\]의 기술을 채택하여 제너레이터가 대상 도메인의 실제 샘플을 입력으로 제공받을 때 identity mapping 근처에 있도록 정규화합니다. 즉, **Lidentity(G,F) = Ey\_pdata(y)\[∥G(y) − y∥1\] + Ex∼pdata (x) \[∥F (x) − x∥1 \]**입니다.

Lidentity가 없으면, 생성자 G와 F는 굳이 필요하지 않을 때 입력 이미지의 색조를 자유롭게 변경할 수 있습니다. 예를 들어, Monet의 그림과 Flickr 사진 간의 매핑을 학습할 때, 생성자는 종종 낮에 그린 그림을 일몰 시간에 찍은 사진에 매핑합니다. 왜냐하면 적대적 손실과 사이클 일관성 손실 아래에서 이러한 매핑이 동등하게 유효할 수 있기 때문입니다. 이러한 identity mapping 손실의 효과는 그림 9에서 보여집니다. figure 12, figure 9는 학습 데이터셋에 포함되어 있는 그림, 하지만 다른 set은 오직 test set으로부터 그려진 그림. training set이 paired datqa를 포함하고 있지 않아서, 학습 세트 그림에 대한 타당한 translation을 찾는 것은 쉬운 일이 아니다. 실제로, Monet이 새 그림을 그릴 수 없기 때문에, 보지 않은 test set 그림에 대한 generalization은 not pressing problem

### Photo enhancement

우리는 우리의 방법이 얕은 깊이의 초점을 가진 사진을 생성하는 데 사용될 수 있음을 보여줍니다. 우리는 Flickr에서 다운로드한 꽃 사진을 기반으로 모델을 훈련합니다. 소스 도메인은 스마트폰으로 찍힌 꽃 사진으로 구성되어 있으며, 보통 작은 조리개로 인해 깊은 DoF(초점 깊이)를 가지고 있습니다. 대상은 조리개가 큰 DSLR로 촬영된 사진을 포함합니다. 우리 모델은 스마트폰으로 촬영된 사진으로부터 더 얕은 깊이의 초점을 가진 사진을 성공적으로 생성합니다.

> : shallow depth of field: 얕은 초점. 초점이 맞은 대상과 배경이 흐릿하게 보이는 효과. 인물 사진 / 작품 사진에 활용. 구목하고자 하는 대상을 강조하기 위해 활용.  
> 따라서 source domain은 스마트폰의 **작은 조리개로 깊은 초점** \--> target은 **조리개가 커서 얕은 초점**.

### Comparison with Gatys


## Limitations and Discusssion

:::{figure-md} 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdJc1k5%2Fbtr76zUPUWj%2F27Mk0oQ5VanEHANWWmaseK%2Fimg.png" class="bg-primary mb-1" width="800px">

Limitation and Discussion
:::

이 방법은 많은 경우에 흥미로운 결과를 얻을 수 있지만, 결과는 결과가 균일하게 좋은 것은 아니었습니다.

1.  (해석) 개<->고양이 task와 같은 경우는 input image에서 최소한의 변화만 주어, 사람이 보았을 때 실제로 변화가 안되는 경우도 있었고, 형체가 애매해진 경우도 있음. 이런걸 보았을 때, 세부적인 구조(geometry? 라는 표현을 보아), 눈, 코, 입에 대한 정확한 구조를 구현하는데 한계가 있어 보임.
2.  말<--> 얼룩말 예제의 경우, 말은 사람이 타는 모습이 많았는데, 얼룩말의 경우는 사람이 타는 사진이 없다보니, 사람 뿐만 아니라 배경도 얼룩 그림을 그림을 그리거나, 단순히 얼룩말에서 노랗게 칠한 경우가 생김.
3.  때때로 photo --> image task에서 나무와 건물의 label을 바꾸는 경우도 있었음.  
    이러한 모호성을 해결하려면 weak semantic supervision이 필요할 수도 있을 것 같음.

마무리: 그럼에도 불구하고 많은 경우 완전히 짝지어지지 않은 데이터가 풍부하게 제공되며, 이를 활용해야 합니다. 이 논문은 이러한 "unsupervised" setting에서 가능한 것의 한계를 늘리는데 기여합니다.
