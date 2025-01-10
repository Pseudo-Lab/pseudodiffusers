```{admonition} Information
- **Title:** CAT3D: Create Anything in 3D with Multi-View Diffusion Models (NIPS 2024)

- **Reference**
    - Paper: [https://arxiv.org/abs/2405.10314](https://arxiv.org/abs/2405.10314)    
    - Code: -
    - Project Page : [https://cat3d.github.io/](https://cat3d.github.io/)

- **Presentor:** Geonhak Song

- **Last updated on December.18 , 2023**
```

# CAT3D

<img src="../../pics\CAT3D\cat3d_fig_1.jpg" alt="cat3d_fig1" class="bg-primary mb-1" width="700px">

## 1. Introduction

3D content의 수요는 증가하지만 3D content는 여전히 부족함.

이를 위해서는 복잡하고 특수한 tool 사용해야 하며 상당한 시간과 노고가 들어감.

NeRF, Instant-NGP, Gaussian Splatting과 같은 최신 기술들은 2D 이미지를 기반으로 어떠한 viewpoint에서 render하여 3D를 생성해내는 방법을 제안함.

그러나 여전히 고품질 장면 생성을 위해서는 수 천장을 찍어야 하는 labor-intensive process 필요.

불충분한 장면 시점은 부정확한 기하학적 외관적 결과를 초래하며 신뢰 높은 3D 복원의 어려움을 제공

단일 이미지, text로 3D 생성 방법을 제안하는 방법 또한 제공되고 있지만, 여전히 품질, 효율성, 일반화에서 어려움이 있음.

본 논문은 **CAT3D**를 통해 입력된 적은 수의 뷰(few-view input)만으로 3D 일관성을 유지하며 고품질의 3D 콘텐츠 생성.

Novel-view synthesis가 가능한 Multi-View Diffusion Model 훈련 진행을 통해 다수의 3D consistent image를 생성할 수 있는 효과적인 sampling strategy 제안.

any view point / few-view에서도 3D 생성 가능 

1분 미만의 생성

## 2. Related Work

**2D priors**

**2D priors with camera conditioning**

**Multi-view priors**

**Video Priors**

**Feed-forward methods**

## 3. Method

<img src="../../pics\CAT3D\cat3d_fig_3.jpg" alt="cat3d_fig3" class="bg-primary mb-1" width="700px">

두 단계 접근법 사용

Step 1. multi-view diffusion model을 활용하여 한 개 혹은 그 이상의 입력 view와 일관된 다수의 새로운 view 생성.

Step 2. 생성된 view를 활용해 robust 3D reconstruction pipeline 실행

### 3.1 Multi-View Diffusion Model

- 3D 장면의 한 개 혹은 그 이상의 view를 입력으로 받아, 카메라 위치에 따라 다수의 출력 이미지를 생성하는 multi-view diffusion model 훈련(“a view” is a paired image and its camera pose)

- 입력 : $M$개의 조건부 views ($I_{cond}$ : images, $p_{cond}$ : camera parameters) & $N$개의 target camera parameters $p_{tgt}$
- 출력 : $N$개의 target images $I_{tgt}$
- 위 조건을 만족하는 joint distribution 모델 학습

<img src="../../pics\CAT3D\cat3d_eq_1.jpg" alt="cat3d_eq_1" class="bg-primary mb-1" width="700px">

**Model architecture**

- 모델 아키텍처는 video latent diffusion model과 유사하지만, time embedding 대신 각 이미지에 대해 camera pose embedding 사용

학습 방법

- conditional and target image 집합이 주어졌을 때, 모델은 각 이미지를 VAE latent representation를 통한 encoding
- 이후 conditional signal이 주어졌을 때 latent representation의 joint distribution를 추정하는 diffusion model training

- T2I LDM initialize & resolution 512×512×3
- Backbone : 2D diffusion model
    - 추가 layer는 다중 입력 이미지 latent와 연결된 상태

<img src="../../pics\CAT3D\cat3d_mvdream.jpg" alt="cat3d_mvdream" class="bg-primary mb-1" width="700px">

[MVDream](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/mv-dream)에서와 같이, 3D self-attention layer (2D in space and 1D across images) 사용

이때, 최소한의 추가 모델 파라미터 사용을 위해 2D self-attention을 직접적으로 inflate하는 방식으로  3D self-attention을 latent에 연결

<img src="../../pics\CAT3D\cat3d_ReconfFusion_fig.jpg" alt="cat3d_ReconfFusion_fig" class="bg-primary mb-1" width="700px">

3D self-attention layer를 통한 input view conditioning은 ReconFusion에서 사용한 PixelNeRF나 CLIP image embedding 필요성 제거

<img src="../../pics\CAT3D\cat3d_fig_7.jpg" alt="cat3d_fig_7" class="bg-primary mb-1" width="700px">

빠른 training, sampling을 위한 FlashAttention 사용 & LDM weight fine-tuning

pretrained image diffusion model에서 더 높은 차원의 데이터를 포착하는 multi-view diffusion model로 전환할 때, noise schedule을 high noise level로 이동하는 것이 중요. 

log SNR(signal-to-noise ratio)를 log N만큼 이동, N: target 이미지 수

- [Simple diffusion: End-to-end diffusion for high resolution images (23.01)](https://arxiv.org/abs/2301.11093) 참조

target image와 conditioning image 구분을 위해 **Binary mask**를 channel dim을 따라 latent에 concat 

다양한 3D Generation 설정 처리를 위해 8개의 conditioning & target view(N+M=8)를 모델링할 수 있는 하나의 모델을 학습

training 중 조건부 view 1 또는 3 무작위 선택 

자세한 내용 : Appendix B. Details of Multi-View Diffusion Model

**Camera conditioning**

camera pose conditioning을 위한 동일 height, width의 a camera ray representation (“raymap”) 사용. 이후 각 공간 위치에서 ray origin, direction로 encoding

ray는 광선은 **첫 번째 조건부 이미지의 camera pose 기준**으로 계산되므로, pose representation은 3D world 좌표계의 rigid transformation에 대해 불변성을 가짐.

- **첫 번째 카메라를 기준으로 모든 위치를 상대적으로 정의**하여, 모델이 3D 세계 좌표계의 이동이나 회전 같은 전역적인 변환에 영향을 받지 않도록 한다는 뜻입니다.
- 이는 3D 재구성의 안정성과 일관성을 높이는 데 중요한 역할

각 이미지의 raymap은 해당 이미지의 latent에 channel-wise concat

### 3.2 Generating Novel Views

목표 : 입력 view 집합이 주어졌을 때, 일관된 view에 대한 대규모 집합을 생성하는 것
필요 1) **샘플링할 카메라 위치 집합**을 결정
필요 2) 일관된 view 생성이 가능한 a sampling strategy

**Camera trajectories**

3D scene reconstruction에서의 문제는 장면을 완전히 포괄하기 위해 필요한 view가 복잡하고 장면의 콘텐츠에 따라 달라질 수 있다는 점

경험적으로 다양한 유형의 장면에 대해 **적절한 카메라 경로**를 설계하는 것이 설득력 있는 few-view 3D reconstruction 달성하는 데 매우 중요함 발견

카메라 경로는 재구성 문제를 완전히 제약할 수 있도록 충분히 철저하고 밀도가 높아야 할 뿐 아니라장면 내 물체를 관통하거나 부자연스러운 각도에서 장면 콘텐츠가 제공되지 않아야함.

In summary, we explore four types of camera paths based on the characteristic of a scene:

(1) orbital paths of different scales and heights around the center scene

- 중심 장면 주위를 도는 다양한 크기와 높이의 궤도형 경로

(2) forward-facing circle paths of different scales and offsets

- 다양한 크기와 오프셋을 가진 전방을 향한 원형 경로

(3) spline paths of different offsets

- 다양한 오프셋을 가진 스플라인 경로

<img src="../../pics\CAT3D\cat3d_image.png" alt="cat3d_image" class="bg-primary mb-1">


(4) spiral trajectories along a cylindrical path, moving into and out of the scene

- 실린더 경로를 따라 장면으로 들어가고 나오는 나선형 경로

Appendix C 참조

<img src="../../pics\CAT3D\cat3d_fig_8.jpg" alt="cat3d_fig_8" class="bg-primary mb-1" >


**Generating a large set of synthetic views**

다중 시점 확산 모델을 Novel view synthesis 에 적용할 때의 문제점

- 소수의 유한한 입출력 view(총 8개)로 훈련되었다는 점

(?)
출력 view의 개수를 늘리기 위해, 

- target viewpoints을 더 작은 그룹으로 clustering
- 조건부 view가 주어진 상태에서 각 그룹을 독립적으로 생성

single-image conditioning에서는 auto-regressive sampling 전략 채택

- 장면을 커버하는 7개의 **anchor views** 집합을 생성(ZeroNVS과 유사하며, k-means++[69]에서 제안된 탐욕 초기화(greedy initialization)를 사용)
- 관찰된 view와 앵커 view가 주어진 상태에서 나머지 view 그룹을 병렬로 생성.

이 방식은 앵커 view 간의 long-range consistency과 근접 view 간의 local similarity을 유지하면서도, 대규모 합성 view 집합을 효율적으로 생성할 수 있도록 합니다.

single-image setting : 80 views / few-view setting : 480~960 views 생성

**Conditioning larger sets of input views and non-square images**

Reconfusion에서 제안된 방법과 같이 가까운 M개의 view condition 선택

sampling동안 단순하게 multi-view diffusion architecture의 sequence length를 늘리는 방법보다 nearest view conditioning과 grouped sampling strategy가 성능이 더 좋았음.

넓은 종횡비의 이미지를 처리하기 위해, 정사각형으로 자른 입력 view에서 나온 정사각형 샘플과 padding된 정사각형 입력 뷰에서 잘라낸 샘플 결합.

### 3.3 Robust 3D reconstruction

생성된 뷰는 일반적으로 완벽한 3D 일관성을 갖추고 있지않음.

때문에 일관적이지 않은 입력 view에 대한 robustness 향상을 위한 NeRF 훈련 방법 수정 진행

Zip-NeRF 기반 **Photometric Reconstruction Loss, Distortion Loss, Interlevel Loss, Normalized L2 Weight Regularizer**

추가로 rendered image와 input image 간 **perceptual loss (LPIPS)** 사용
LPIPS는 렌더링된 이미지와 관찰된 이미지 간의 **고차원적 의미적 유사성**을 강조하며, 저차원 고주파 세부사항의 불일치 가능성은 무시
**가장 가까운 관찰된 뷰와의 거리**를 기준으로 생성된 뷰의 손실에 가중치를 부여

Weight

- 훈련 초기에는 균일하게 적용
- 관찰된 view와 더 가까운 view간의 reconstruction loss에 더 강한 패널티를 부여하는 weighting func로 변환되도록 점진적으로 강화

Appendix D 참조

## 4. Experiments

4 datasets with camera pose annotations: Objaverse, CO3D, RealEstate10k, MVImgNet 

4.1 Few-View 3D Reconstruction

<img src="../../pics\CAT3D\cat3d_table_1.jpg" alt="cat3d_table_1" class="bg-primary mb-1" width="700px">
![table 1.jpg](table_1.jpg)

4.2 single image evaluation

<img src="../../pics\CAT3D\cat3d_fig_5.jpg" alt="cat3d_fig_5" class="bg-primary mb-1" width="700px">
![fig 5.jpg](fig_5.jpg)

<img src="../../pics\CAT3D\cat3d_table_2.jpg" alt="cat3d_table_2" class="bg-primary mb-1" width="700px">
![table 2.jpg](table_2.jpg)

4.3 Ablation

<img src="../../pics\CAT3D\cat3d_table_3.jpg" alt="cat3d_table_3" class="bg-primary mb-1" width="700px">
![table 3.jpg](table_3.jpg)

<img src="../../pics\CAT3D\cat3d_fig_6.jpg" alt="cat3d_fig_6" class="bg-primary mb-1" width="700px">
![fig 6.jpg](fig_6.jpg)

LPIPS loss is crucial for achieving high-quality texture and geometry,

increasing the number of generated views from 80 (single elliptical orbit) to 720 (nine orbits) improved central object geometry but sometimes introduced background blur, probably due to inconsistencies in generated content