```{admonition} Information
- **Title:** SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations

- **Reference**
    - Paper: [https://arxiv.org/pdf/2108.01073.pdf](https://arxiv.org/pdf/2108.01073.pdf)

- **Author:** Seunghwan Ji

- **Last updated on Oct. 03, 2023**
```

# SDEdit

## Abstract

- 최근 이미지 생성 분야에서의 놀라운 진화 속도가 계속 되어오고있다. (GAN, Diffusion etc..)
- 이 중 이미지에 random noise를 추가해 denoising 과정을 학습하는 Diffusion을 통해 high quality의 이미지를 생성할 수 있다.
- 또, 생성되는 이미지를 사용자가 원하는 방향으로 이끌어내려는 연구 분야도 활발히 진행되고있다 (a.k.a Editing)
- 하지만, GAN 또는 Diffusion을 포함한 방식으로의 Editing에는 몇가지 단점이 있고, SDEdit은 그런 문제점을 해결해나아갔다는 점을 논문의 핵심 Contribution으로 제시하였다.

## 1. Introduction

- Abstract에서 말한 Editing이란, 유저가 생성하고자 하는 Guide를 제시하면 모델은 해당 Guide를 기반으로 이미지를 생성해내는 Vision Task를 의미한다.
- 이때 두가지의 평가요소가 있는데
    1. faithful : 유저의 Guide를 얼마나 잘 따르는지
    2. realistic : 생성된 이미지가 얼마나 real한지
- 기존의 연구방식은 크게 두가지로 나뉜다.
    1. GAN(Generative Adversarial Network) 기반
    2. Diffusion 기반
- 이 중 기존에 SOTA를 이룬 GAN 방식을 살펴보면 다시 크게 두가지로 나뉜다.
    1. conditional GAN
        - 특징 : 원본 이미지에서 Edit된 Pair 이미지를 직접 학습
        - 단점 : Pair Dataset이 반드시 필요하고, Condition마다 재학습을 요구
    2. GAN Inversion
        - 특징 : 이미지를 Latent space로 Inversion한 후, Latent vactor를 조작해(manipulate) Edited image를 생성
        - 단점 : 새로운 loss function이 정의되어야하고, condition마다 재학습을 요구
- 그에 반해 SDEdit은
    1. Pair Dataset이 필요하지 않다.
    2. 추가적인 loss function과 재학습이 모두 필요하지 않다.
    3. 단 한개의 pretrained weight로 모든 condition의 이미지를 생성할 수 있다.
    

## 2. Related Works

### 2.1. Score Based Generated Model
:::{figure-md} 
<img src="../../pics/SDEdit/img0.png" alt="SDEdit_00" class="bg-primary mb-1" width="700px">

Image 1
:::

- Key Idea
    - *“Real 이미지들은 실제 데이터 확률 분포에서 높은 값을 유지할 것이다. 따라서, 이미지를 분포가 높은곳으로 update 해나가면 좋은 퀄리티의 이미지를 생성하는 모델을 얻어낼 수 있다.”*
- 이 때, score는 확률 밀도 함수의 순간 기울기(미분값)로 정의한다.

### 2.2. Score Based Generated Diffusion Model (SDE, SMLD)
:::{figure-md} 
<img src="../../pics/SDEdit/img1.png" alt="SDEdit_01" class="bg-primary mb-1" width="700px">

Image 2
:::


- 위에서 제시한 Score Based Generated Model에 Diffusion 방식을 적용한 모델
- Forward Process 과정에서 이미지에 noise를 주입하는데, 이 때 Stochastic Differential Equation 수식을 이용해 noise를 주입한다.
- 또다른 Diffusion 모델인 (Probability based) DDPM과의 차이는 Forward, Reverse process에서 정의하는 equation의 차이 정도이다.
- paper : [https://arxiv.org/abs/1907.05600](https://arxiv.org/abs/1907.05600)

## 3. Methods

1. Pre-Setup
    - Guide image의 Level을 정의한다.
        :::{figure-md} 
        <img src="../../pics/SDEdit/img2.png" alt="SDEdit_02" class="bg-primary mb-1" width="700px">
        
        Image 3
        :::
        
        1. low-level guide : real 이미지위에 image patch를 추가
        2. mid-level guide : real 이미지위에 stroke를 추가
        3. high-level guide : 단순히 coarse한 stroke의 이미지
2. Procedure
    - DDPM과 달리 SDE의 경우, 완전히 noise화된 이미지 즉, random noise로부터 denoising을 진행할 필요가 없다.
    - 즉, 적절한 $t_{0} \in [0,1]$를 지정한 후 denoising process가 가능하다.
        
        :::{figure-md} 
        <img src="../../pics/SDEdit/img3.png" alt="SDEdit_03" class="bg-primary mb-1" width="700px">
        
        Image 4
        :::
    - 이 때, 적절한 $t_{0}$를 정의해야하는데,
        1. $t_{0}$ = 1 (i.e. random noise)이면, realistic하지만, faithful 하지않은 이미지
        2. $t_{0}$ = 0 이면, faithful하지만, artistic한 이미지
        
        를 얻게된다.
        
        :::{figure-md} 
        <img src="../../pics/SDEdit/img4.png" alt="SDEdit_04" class="bg-primary mb-1" width="700px">
        
        Image 5
        :::
    - 아래는 SDEdit의 적용 과정이다.
        
        :::{figure-md} 
        <img src="../../pics/SDEdit/img5.png" alt="SDEdit_05" class="bg-primary mb-1" width="700px">
        
        Image 6
        :::

## 4. Experiments

- Score
    - Metric
        - realistic : Kid score (lower is better)
        - faithful : $L_{2}$ score (lower is better)
        - 그 외 종합적인 평가 지표로 survey를 통한 수치를 제시하였다.
            
            :::{figure-md} 
            <img src="../../pics/SDEdit/img6.png" alt="SDEdit_06" class="bg-primary mb-1" width="700px">
            
            Image 7
            :::
    - 기존의 GAN 방식들과 비교했을 때 Kid, $L_{2}$ score 모두 더 좋은 수치를 보이는 것을 확인할 수 있다.
- Comparison with GAN (styleGAN-ADA + Inversion)
    
    :::{figure-md} 
    <img src="../../pics/SDEdit/img7.png" alt="SDEdit_07" class="bg-primary mb-1" width="700px">
    
    Image 8
    :::
    - SDEdit이 GAN Based model보다 더 자연스럽고(realistic), 유저의 guide를 잘 따르는(faithful)것을 확인할 수 있다.
- Comparison with original blending technique
    
    :::{figure-md} 
    <img src="../../pics/SDEdit/img8.png" alt="SDEdit_08" class="bg-primary mb-1" width="700px">
    
    Image 9
    :::

    :::{figure-md} 
    <img src="../../pics/SDEdit/img9.png" alt="SDEdit_09" class="bg-primary mb-1" width="700px">
    
    Image 10
    :::
    - 기존의 전통적인 방식의 몇가지 blending 기법과 비교해도 더 좋은 성능과 수치를 보이는 것을 확인할 수 있다.
