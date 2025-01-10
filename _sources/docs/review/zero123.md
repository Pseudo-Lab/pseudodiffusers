# zero 1-to-3 : Zero-shot One Image to 3D Object

``` {admonition} Information
- **Title:** zero 1-to-3 : Zero-shot One Image to 3D Object

- **Reference**
    - Paper: [<https://arxiv.org/abs/2303.11328>](<https://arxiv.org/abs/2303.11328>)
    - Code: [<https://github.com/cvlab-columbia/zero123>](<https://github.com/cvlab-columbia/zero123>)

- **Author:** Jeongin Lee

- **Last updated on Jan. 10, 2025**
```

# Abstract : Zero-shot One Image to 3D Object

---

:::{figure-md} 
 <img src="../../pics/zero123/image.png" alt="tag" class="bg-primary mb-1">
:::

1. **단일  RGB 이미지**를 입력으로  **Object 의 카메라 viewpoint 를 변경**하여 이미지를 합성하는 프레임워크 
2. **제한된 셋팅** 하에서 다중 view 합성 수행을 위해 **large-scale diffusion model** 의 Geometric priors 활용
    1. 조건부 diffusion 모델은 특정 카메라 시점 변형 하에 동일 객체의 새로운 이미지들을 생성할 수있는 **상대적인 카메라 시점의 조정(control)** 을 학습하기 위해 **3D 합성 데이터셋** 사용
    2. 합성 데이터셋에서 훈련되었음에도 wild images , OOD(out-of-distribution) 데이터 등에 대해 강력한 zero-shot 일반화 능력 보유 
3. viewpoint-conditioned diffusion 접근법은 단일 이미지로부터의 3D 재구성을 위해서도 사용 가능

# 1. Introduction

---

1. **인간의 3D 인지 능력**
    - 사람은 단일 시점의 이미지에서도 물체의 3D 형태와 외형을 상상 가능
    - 대칭과 같은 기하학적 prior에 의존하기도 하지만 물리적 제약을 넘는 일반화가 가능
    - 현실에 존재하지 않는 객체의 3D 형태까지 예측할 수 있는 능력은 평생동안의 시각적 경험에서 축적된 사전지식(prior)를 기반으로 함.
2. **기존 3D 재구성 접근 방식의 한계**
    - 대부분의 3D 이미지 재구성 접근 방식은 **closed-world setting**에 제한
        - 고비용 3D annotation(CAD model) 과 카테고리별 prior에 의존
        - CO3D 등 대규모 데이터셋으로 open-world 재구성 연구가 진전되었으나 
        여전히 stereo view나 카메라 포즈 정보들과 같은 학습을 위한 기하적 정보가 필요
        - large-scale diffusion 모델의 성공을 가능하게 한 인터넷 규모의 text-image 데이터셋들과 비교할 때 아직 3D 데이터의 수집 규모는 미미한 상태
3. **diffusion 모델의 인터넷 규모 사전 학습과 한계** 
    - 인터넷 규모의 사전학습은 large-scale diffusion model 에 풍부한 semantic prior 를 부여
    - 하지만, 여전히 기하학적 정보 캡처 능력은 부족
4. **Zero 1-to-3 를 제안** 
    - **Zero 1-to-3 :**
        - Stable Diffusion과 같은 large-scale diffusion 모델이 단일 RGB 이미지를 입력으로,
        
        <aside>
        1. **zero-shot novel view synthesis**
        2. **3D shape reconstruction**
        </aside>
        
              를 수행하기 위해 **카메라 view point(시점)을 조작을 컨트롤하는 매커니즘 학습**
        
    - 주어진 단일 RGB 이미지 입력하에 이러한 TASK를 수행 하는 것은 매우 많은 제약이 존재 
    → Diffusion 모델을 기반으로 다양한 시점에서 방대한 객체 이미지를 생성하여 이용
        - Diffusion 모델은 카메라 포즈 없이 2D 이미지에 대한 학습만 수행되었으므로,
         **파인튜닝을 통해 상대적인 카메라 회전 및 이동을 컨트롤**하여 새로운 시점의 이미지를 생성하는 접근법 제시
        - 파인튜닝을 통해 선택한 다른 카메라 시점에서의 임의의 이미지를 생성 가능

# 2. Related Work

1. **3D generative models**
    - 생성 모델의 동향
        - 대규모 이미지-텍스트 데이터셋과 결합된 생성 모델의 발전으로 고품질의 다양한 장면과 객체를 합성 가능해짐.
        - 특히 diffusion 모델의 경우 denoising objective를 통해 확장 가능한 이미지 생성 학습에 효과적
        - 이를 3D 도메인으로 확장하려면 많은 3D annotation 데이터를 필요로 하므로, 이에 대한 대안으로 **대규모 2D diffusion 모델을 3D task 에 전이하는 방식**이 연구되고 있음.
    - **NeRF과 DreamFields의 역할**
        - **NeRF**는 고품질 장면 인코딩에 탁월하여 단일 장면 재구성에 주로 사용되며, 관찰되지 않은 각도에서 새로운 view 를 예측
        - **DreamFields**는 NeRF를 3D 생성 시스템의 주요 요소로 활용하여, 텍스트 입력에서 고품질 3D 객체와 장면을 생성할 수 있게 함.
    - **Zero-1-to-3 의 접근법**
        - 새로운 시점 합성을 **뷰포인트로 조건화된 image-to-image 변환** 작업으로 보고 **diffusion** 모델을 활용
        - 학습된 모델은 **3D distillation** 과 결합해 단일 이미지로부터 3D Shape reconstruction 수행
        - 합성 데이터셋으로 시점 조작을 학습하여 wild 이미지에 대해 zero-shot 일반화를  입증
    - **선행 연구와의 비교**
        - 선행연구는 **Zero-1-to-3** 와 유사한 파이프라인을 채택하였으나, 제로샷 일반화 능력을 입증하지 못함
        - 이 외에도 다양한 접근방식이 language-guided 와 text-inversion 을 활용하여 image-to-3D 생성 태스크를 위한 유사한 방법론들을 제안
        - 하지만 본 방법론은 합성 데이터셋을 통해 view point 컨트롤을 학습하고, 제로샷 일반화 능력을 보여줬다는 데에서 차이가 있음.
    
2. **Single-view object reconstruction**
    - **단일 뷰에서 3D 객체 재구성**:
        - 단일 뷰에서 3D 객체를 재구성하는 것은 매우 어려운 문제
        - 이를 해결하기 위한 강력한 사전 지식(**prior**) 필요
    - [**기존 접근 방식**:](https://velog.io/@dldydldy75/3D-Understanding)
        
        :::{figure-md} 
        <img src="../../pics/zero123/image 1.png" alt="tag" class="bg-primary mb-1">
        :::
        
        1. **데이터 수집 형태에 따른 전역적인(global) 특성 기반의 조건화 모델** 
            - 기존 연구 중 일부는 3D 데이터를 **메쉬, 복셀, 또는 포인트 클라우드** 형식으로 수집해 이를 기반으로 조건에 대한 이미지 인코사전 지식을 형성
            - 사용된 3D 데이터의 종류에 큰 제약 존재
            - 데이터에 따른 조건의 타입에 대한 **global nature** 로 인해 일반화 능력이 부족
            - 새로운 관점에서 재구성하기 위해 추가적인 포즈 추정 단계가 필요
        2. **국소적으로(locally) 조건화된 모델**
            - 이미지의 국소적 특성을 직접 사용하여 장면 재구성을 시도
            - 교차 도메인 일반화 능력이 더 좋지만, 일반적으로 가까운 뷰 재구성에 제한됨.
        
        <aside>
        
        - global nature : 객체의 형태, 기하학적 구조, 데이터 종류, 색상 분포 등
        - local nature : 저수준 정보로 색상 픽셀값, 특정 패턴, 텍스쳐 등을 의미
        </aside>
        
        - **MCC (Multiview Compressive Coding for 3D Reconstruction)**
            - **RGB-D** 뷰를 사용하여 3D 재구성을 위한 general-purpose representation 을 학습하며, 대규모 객체 중심 비디오 데이터셋으로 훈련됨
        - **Zero-1-to-3**
            - 사전 훈련된 Stable Diffusion 모델에서 직접 풍부한 기하학적 정보를 추출할 수 있음을 보여주며, 이는 **추가적인 깊이 정보 없이**도 가능함을 보임

# 3. Methods

- **목표 :** 주어진 단일 RGB 이미지 $x \in \mathbb{R}^{H \times W \times 3}$ 를  입력으로, 다른 카메라 시점에서의 객체 이미지를 합성
- **카메라 변환**
    - $R \in \mathbb{R}^{3 \times 3}$ : relative camera rotation
    - $T \in \mathbb{R}^{3}$ : relative camera translation
- **모델 학습**
    - 함수  $f$ 를 학습하여 새로운 이미지를 합성
        
        $$
        \hat{x}_{R,T} = f(x, R, T)
        $$
        
        - $*x$ : 주어진 단일 RGB 이미지 $(x \in \mathbb{R}^{H \times W \times 3})$*
        - $*\hat{x}_{R,T}$ :*  합성 이미지
        - $\hat{x}_{R,T}$ 가 실제지만 관측되지 않은 새로운 view $x_{R,T}$ 와 지각적으로 유사하도록 추정

- 단일 RGB 이미지에서 새로운 시점으로의 합성은 많은 제약사항이 존재 (단일 이미지로 많은 정보를 예측해야 하므로)
- **large-scale Diffusion(Stable Diffusion)을 활용하여 태스크 수행**
    - 프롬프트로부터 다양한 이미지를 생성할 때 갖는 뛰어난 zero-shot 능력을 활용
- **large-scale Diffusion에서 3D 정보를 추출하는 능력을 저해하는 요인 두가지**
    1. **뷰포인트 간의 명확한 대응 관계의 부족** : 대규모 생성 모델들은 다양한 객체와 시점에서 훈련되었지만, 서로 다른 시점들 간의 연관성을 명시 으로 인코딩하지 않음.
    2. **인터넷 규모 데이터셋에 반영된 뷰포인트 편향** : 생성 모델들은 인터넷에서 반영된 시점 편향을 물려받아, 특정한 자세 및 시점에서의 이미지를 생성하는 경향 존재

    :::{figure-md} 
    <img src="../../pics/zero123/image 2.png" alt="tag" class="bg-primary mb-1">
    :::

    

## 3.1. Learning to Control Camera Viewpoint

- **목표**: Diffusion 모델을 통해, 촬영된 이미지에서 카메라의 외부 매개변수를 제어할 수 있는 메커니즘을 학습하여 새로운 시점의 이미지를 합성하는 것.
- **데이터셋**: 이미지 쌍과 상대적 카메라 외부 매개변수로 구성된 데이터셋 ${ (x, x_{(R,T)}, R, T) }$
- **접근법 [Figure3]**
    
    :::{figure-md} 
    <img src="../../pics/zero123/image 3.png" alt="tag" class="bg-primary mb-1">
    :::
    
- 사전 훈련된 diffusion 모델을 미세조정하여 나머지 표현을 손상시키지 않고 카메라 파라미터를 제어하도록 학습.
- **Latent Diffusion Architecture 를 이용**
    - **[참고] LDM**
    :::{figure-md} 
    <img src="../../pics/zero123/image 4.png" alt="tag" class="bg-primary mb-1">
    :::
    - Encoder($\mathcal{E}$), Denoiser(U-Net, $\epsilon_\theta$), Decoder($\mathcal{D}$)로 구성
    - **The Objective**
        
        $$
        \text{min}_{\theta} \mathbb{E}_{z \sim \mathcal{E}(x), t, \epsilon \sim \mathcal{N}(0, 1)} \left\| \epsilon - \epsilon_{\theta}(z_t, t, c(x, R, T)) \right\|^2_2
        $$
        
        - $z \sim \mathcal{E}(x)$:  $x$ 에 대한 Encoder의 출력
        - $t ∼ [1, 1000]$ : Diffusion time step
        - $\epsilon \sim \mathcal{N}(0, 1)$: 가우시안 노이즈 샘플.
        - $c(x, R, T)$ : input view 와 상대적 camera extrinsic 임베딩
    - $\epsilon_\theta$ **가 훈련된 후, 추론 모델 $f$는 $c(x, R, T)$ 를 조건으로 가우시안 noise image로부터 반복적인 denoising 을 통해 이미지를 생성**
    
- **본 방법론의 효과**
    - 사전 훈련모델에 대한 이러한 방식의 파인튜닝을 통해 모델은 카메라 viewpoints 통제를 위한 일반적인 메커니즘을 학습 가능
        
        → fine tuning 데이터셋에서 보이는 객체 외부의 정보를 외삽 
        
    - fine tuning 을 통해 제어 기능이 “**부착**”될 수 있고 diffusion의 포토리얼리스틱 이미지 생성 능력을 유지
    - 이러한 **compositionality**(구성 가능성)는 모델에서 제로샷 기능을 확립하며, 
    최종모델은 3D assets 이 부족하고 fine-tuning 집합에 존재하지 않는 객체 클래스에 대한 새로운 뷰를 합성 가능

## 3.2. View-Conditioned Diffusion

- 단일이미지로부터 3D 객체를 재구성하기 위해 저수준 인식(깊이, 음영, 텍스처 등)과 고수준 이해(형태, 기능, 구조 등)  모두를 필요로 함.

**→ 하이브리드 조건부 메커니즘 채택**:  두 가지 스트림을 결합 

1. **1st Stream**
    - 입력 이미지를 CLIP(Contrastive Language–Image Pretraining) 임베딩으로 변환
    - 이 CLIP 임베딩에 카메라의 상대적인 회전($R$)과 변환($T$)를 연결하여 “posed CLIP” 임베딩인  $c(x, R, T)$을 생성
    - 이 임베딩을 통해 고수준의 의미 정보를 제공하는 **cross-attention**을 U-Net의 디노이징 과정에 적용
2. **2nd Stream**
    - input image 와 denoising 중의 이미지를 channel-concatenated
    - 모델이 합성되는 중인 객체의 identity 와 details 을 유지하도록 함.
- **classifier-free guidance** 적용을 위해 이미지와 posed CLIP 임베딩을 무작위로 영(0) 벡터로 설정하고, 추론 중 조건부 정보를 스케일링하는 메커니즘

## 3.3 3D Reconstruction

- 많은 응용에서, 객체의 새로운 뷰를 합성하는 것만으로는 충분하지 않으며, 객체의 외관과 기하 구조를 모두 캡처하는 전체 3D 재구성이 필요
- **Score Jacobian Chaining(SJC)**을 채택하여, 텍스트-이미지 확산 모델의 사전 정보를 사용하여 3D 표현을 최적화
        
    - diffusion모델의 확률적 특성으로 인해, 그래디언트 업데이트가 매우 불확실하게 변함.
    - DreamFusion을 기반으로 한 **SJC**에서 사용된 기술로, **classifier-free guidance 값을 기존보다  상당히 증가 → 각 샘플의 다양성 감소 & reconstruction 의 fiderlity 향상**
    - **SJC** : Dream Fusion 에서 NeRF로 만들어지는 이미지는 pretrained된 Diffusion Model을 학습시킬 때, 없었던 확률 분포이기 때문에 OOD (Out of Distribution) 문제가 발생 [참고](https://arxiv.org/abs/2208.01618)
- **3D reconstruction with Zero-1-to-3 [Figure4]**
    1. **SJC 와 유사하게 임의의 뷰포인트를 샘플링**하고, 볼륨 렌더링을 수행
    2. 볼륨 렌더링으로 생성된 이미지에 가우시안 노이즈를 주입
    3. input 이미지  $x$, CLIP 임베딩  $c(x, R, T)$ 및 타임스텝 $t$ 를 조건부로  Denoising U-Net, $\epsilon_\theta$을 사용하여  non-noisy input $x_π$에 대한 스코어를 근사
    
    $$
    \nabla \mathcal{L}_{SJC} = \nabla _{I_{\pi}} \log p_{\sqrt{2}\epsilon}(x_{\pi})
    $$
    
    - $\nabla \mathcal{L}_{SJC}$ : PAAS 스코어
    
    :::{figure-md} 
    <img src="../../pics/zero123/image 7.png" alt="tag" class="bg-primary mb-1">
    :::
    
- 입력 뷰와의 **MSE** 손실로 최적화
- NeRF representation 규제를 위한 추가 loss term
    - **Depth smoothness loss** to every sampled viewpoint
    - **near-view consistency loss** : 근접 뷰 간의 외관 변화(appearance change)를 규제, 가까운 뷰 간의 일관성을 유지

## 3.4. Dataset

- **Dataset** :  Objaverse, 800K+ 3D models  created by 100K+ artists
- **데이터 구성**
    - ShapeNet과 같이 명시적인 클래스 레이블이 없지만, 다양한 고품질 3D 모델을 포함
    - 모델은 정교한 기하학적 세부사항과 재질 속성을 갖춘 많은 모델을 포함
- **Camera Extrinsics matrices 샘플링**
    - 각 객체에 대해, 객체 중심을 가리키는 **12**개 카메라 외부 매트릭스 $\mathcal{M}_{\rceil}$를 무작위로 샘플링
    - 각 객체에서 12개 뷰를 ray 트레이싱 엔진을 사용해 렌더링합니다.
- **Training**
    - 각 객체에 대해 이미지 쌍$(x, x_{R,T})$을 형성하기 위해 2개 뷰를 샘플링
    - 이 때 두 시점간의 맵핑을 정의하는 relative viewpoint transformation$(R, T)$ 는 두 시점 각각의 extrinsic matrices 를 통해 쉽게 유도 가능
- **카메라 외부 파라미터 (Camera Extrinsics)**
    - **3D 장면 내에서 카메라의 위치와 방향**을 정의하는 요소
    - 카메라의 시점과 실제 세계 좌표계(**World Coordinate system**)를 연결하여, 특정 객체가 3D 공간에서 정확히 어디에 위치하는지를 파악
    - **Camera Extrinsics 구성 요소**
        - **회전(Rotation)**: 카메라가 세계 좌표계에 대해 어떤 방향으로 회전되어 있는지를 나타냄
        - **변환(Translation)**: 카메라가 세계 좌표계의 특정 위치에 어느 좌표로 이동해 있는지를 나타냄
        - [참조 링크](https://jhtobigs.oopy.io/3dcoordinate)

# 4.Experiments

- 평가 대상 : **model’s performance  zero-shot novel view synthesis & 3D reconstruction**
- Objaverse 데이터셋 외의 데이터와 이미지를 사용하였으므로, 제로샷 결과로 간주
- 모델 성능을 합성 객체와 장면들을 다양한 복잡함 수준에서 정량적으로 결과 비교
- 다양한 자연 이미지(일상적인 객체 사진부터 그림까지)를 사용하여 질적 결과를 보고

## 4.1. Tasks

- 단일 RGB 이미지를 입력으로, 밀접하게 연관된 두가지 태스크를 수행 & zero-shot 적용
1. **Novel View Synthesis**
    - 단일 뷰에서 객체의 깊이, 텍스처 및 형태를 학습하도록 요구되는 오랜 3D problem
    - 입력 정보가 극히 제한적일 때, 모델은 prior 를 활용한 새로운 synthesis method 가 필요
    - 최근 방법들 : CLIP consistency 목적함수를 사용, implicit neural fields 최적화에 의존
    - **본 연구의 방법론**
        - **3D reconstruction** 과 **Novel View Synthesis** 간에 **orthogonal** (독립적)
        - **3D reconstruction** 과 **Novel View Synthesis**의 순서를 반대로 전환하여 여전히 입력 이미지에 묘사된 객체의 정체성을 유지
        - Self-occlusion로 인한 **aleatoric uncertainty**을 확률적 생성 모델을 사용해 모델링
        - 대규모 Diffusion 모델로 학습된 semantic, geometric priors  들을 효율적으로 활용
2. **3D reconstruction**
    - **SJC(Symmetric Jacobian Chaining) 및 DreamFusion과 같은 확률적 3D 재구성 프레임워크를 채택하여 최적의 3D 표현을 생성**
    - **parameterize**
        - 3D 표현을 **voxel radiance field**으로 파라미터화
        - density 필드에서 **Marching Cubes** 알고리즘을 사용하여 3D 메쉬를 추출합
            - **Marchinng Cube** 알고리즘 : 3D 스칼라 필드에서 등치선(iso-surface)을 추출하기 위해 사용되는 알고리즘
    - **3D reconstruction 에 대한 view-conditioned diffusion model 활용**
        - diffusion 모델이 학습한 풍부한 2D 외양 prior를 3D 기하학으로 전환 가능한 경로 제공

## 4.2. Baselines

- 본 방법론이 다루는 범위와 일관되도록 아래 두가지 모두에 해당되는 방법론들만 비교
    - **zero-shot setting**
    - **input : single-view RGB image**
1. **Novel View Synthesis** 
    1. **DietNeRF**: viewpoints 전반에 걸쳐 CLIP mage-to-image consistency loss 로 NeRF 규제
    2. **Image Variations (IV)**
        - 텍스트 프롬프트가 아닌 이미지 조건을 받기 위해 Stable Diffusion 을 파인튜닝한 모델로, Stable Diffusion 을 활용한 semantic 최근접 이웃 탐색 엔진으로 간주될 수 있음.
    3. **SJC (SJC-I) :** 
        - diffusion-based text-to-3D 모델인 SJC 를 선택, 이때 텍스트 프롬프트 조건을 이미지 조건으로 대체한 모델을 사용하며 이를 SJC-I 로 명명
2. **3D Reconstruction** 
    - **Multiview Compressive Coding (MCC)**
        - 신경 필드 기반 접근 방식으로, RGB-D 관측치를 바탕으로 3D 표현을 완성
        - CO3Dv2 데이터셋에서 훈련
    - **Point-E**
        - 색칠된 포인트 클라우드 위에 구축된 Diffusion모델
        - OpenAI의 내부 3D 데이터셋에서 훈련되어 더 큰 데이터셋을 사용
    - **MCC**와 **Point**-**E** 외에도, **SJC**-**I**와 같은 다른 기법들과도 비교
    - 특정 데이터셋에서 낮은 수준의 정보와 높은 수준의 이해를 요구하는 3D reconstruction 수행
    
    ---
    
    <aside>
    ☑️ **MCC(다중 뷰 압축 코딩)의 깊이 추정**
    - **MCC**는 입력으로 깊이 정보가 필요, **MiDaS**라는 모델을 사용하여 깊이를 추정
    - **MiDaS** : 상대적인 불일치(disparity) 맵을 생성하여, 이를 절대적인 pseudo-metric 깊이로 변환 (전체 테스트셋에서 합리적으로 보이는 standard scale 과 shift 값을 가정하고 변환)
    </aside>
    

## 4.3. Benchmarks and Metrics

- **데이터셋 평가**
    - Google Scanned Objects (GSO): 고품질 스캔된 가정용 아이템 데이터셋
    - RTMV: 20개의 랜덤 객체로 구성된 복잡한 장면 데이터셋
    - 모든 실험에서 각 데이터셋의 ground truth 3D data 를 사용하여 3D reconstruction을 평가
- **Novel view synthesis evaluation metrics**
    - **이미지의 유사성을 평가 → PSNR, SSIM, LPIPS, FID**
- **3D reconstruction evaluation metrics**
    - **Chamfer Distance, Volumetric IoU (Intersection over Union)**

## 4.4. Novel View Synthesis Results

:::{figure-md} 
<img src="../../pics/zero123/image 12.png" alt="tag" class="bg-primary mb-1">
:::
- **Table 2** : RTMV에서의 새로운 뷰 합성 결과.  RTMV의 장면은 Objaverse 훈련 데이터와 분포가 다르지만, 우리의 모델은 여전히 기준선을 상당한 차이로 능가.

:::{figure-md} 
<img src="../../pics/zero123/image 13.png" alt="tag" class="bg-primary mb-1">
:::
- **Figure 5 : Novel view synthesis on Google Scanned Object** 
왼쪽에 표시된 입력 뷰는 두 개의 무작위 샘플링된 새로운 뷰를 합성하는 데 사용됨. 해당하는 실제 뷰는 오른쪽에 표시되어 있음. 기준 방법들과 비교할 때, 우리가 합성한 새로운 뷰는 실제와 매우 일치하는 풍부한 텍스트 및 기하학적 세부 사항을 포함하고 있으며, 반면 기준 방법들은 고주파 세부 사항의 유의미한 손실을 보임.

:::{figure-md} 
<img src="../../pics/zero123/image 14.png" alt="tag" class="bg-primary mb-1">
:::
- **Figure 6 Novel view synthesis on RTMV**
왼쪽에 표시된 입력 뷰는 두 개의 무작위로 샘플링된 새로운 뷰를 합성하는 데 사용됨. 오른쪽에는 해당하는 실제 뷰가 표시됨. 우리가 합성한 뷰는 큰 카메라 시점 변화가 있을 때조차도 높은 충실도를 유지하며, 대부분의 다른 방법들은 품질이 급격히 저하됨.
****
- **Point-E**
    - Point-E 모델은 다른 기준선(baselines)들보다 더 뛰어난 성능을 발휘하며, 우수한 제로샷(Zero-shot) 일반화 능력을 보임
    - 그러나 생성된 포인트 클라우드의 크기가 작아 Point-E가 새로운 뷰 합성(novel view synthesis)에서의 적용 가능성을 제한함.

:::{figure-md} 
<img src="../../pics/zero123/image 15.png" alt="tag" class="bg-primary mb-1">
:::

- **Figure 7 Novel view synthesis on in-the-wild images.**
1, 3, 4 행은 iPhone으로 촬영한 이미지에 대한 결과를 보여주며, 2nd 행은 인터넷에서 다운로드한 이미지에 대한 결과를 제시. 본 방법은 서로 다른 표면 재료와 기하학을 가진 객체에 대해 로버스트함.
- **샘플의 다양성**
    - diffusion 모델이 NeRF보다 이러한 기본적인 불확실성을 포착하는 데 더 적합한 아키텍처
    - 입력 이미지가 2D이기 때문에 항상 객체의 부분적인 뷰만을 나타내고 많은 부분이 관찰되지 않으므로, diffusion 을 통해 다양한 시점에서 샘플들을 랜덤으로 생성
    
:::{figure-md} 
<img src="../../pics/zero123/image 16.png" alt="tag" class="bg-primary mb-1">
:::
    
- Figure 8: 입력 뷰를 고정하고, 다양한 시점에서 새로운 샘플을 랜덤으로 생성하여 시각화
이러한 다양한 결과는 입력 뷰에서 놓친 기하학적 및 외관 정보를 반영

## 4.5. 3D Reconstruction Results

:::{figure-md} 
<img src="../../pics/zero123/image 17.png" alt="tag" class="bg-primary mb-1">
:::

- 실제 ground truth 와 유사한 고충실도의 3D 메쉬를 reconstruct

:::{figure-md} 
<img src="../../pics/zero123/image 18.png" alt="tag" class="bg-primary mb-1">
:::
:::{figure-md} 
<img src="../../pics/zero123/image 19.png" alt="tag" class="bg-primary mb-1">
:::

- **MCC (Multiview Compressive Coding)**:
    - 입력 뷰에서 보이는 표면에 대한 좋은 추정을 제공하지만, 종종 물체 뒷면의 기하학을 올바르게 추론하지 못함.
- **SJC-I** : 의미 있는 기하학을 재구성하는 데 실패
- **Point-E**: 인상적인 제로샷 일반화 능력을 보여주며, 물체 기하학에 대한 합리적인 추정을 예측
    - 그러나, 4,096 포인트로 구성된 비균일한 희소 포인트 클라우드만 생성 가능하고, 이로 인해 재구성된 표면에 구멍이 생김.
    - 좋은 CD(Chamfer Distance) 점수를 얻지만, 부피 IoU(Intersection over Union)에서는 부족한 평가척도 결과를 보임.
- **제안된 방법**:
    - 학습된 다중 보기 우선 순위를 활용하고 NeRF 스타일 표현의 장점을 결합하여 CD와 부피 IoU 모두에서 이전 작업들보다 개선됨(표 3 및 4에서 확인).

## 4.6. Text to Image to 3D-

:::{figure-md} 
<img src="../../pics/zero123/image 20.png" alt="tag" class="bg-primary mb-1">
:::

- 실제 환경에서 촬영된 이미지 외에도, Dall-E-2와 같은 txt2img 모델이 생성한 이미지에 대해서도 테스트

# 5. Discussion

- **Zero-1-to-3** 방식은 단일 이미지에서 새로운 시점을 생성하고 3D 재구성을 수행하는 제로샷 방식
- 사전 학습된 Stable Diffusion 모델의 **풍부한 의미적, 기하적 prior 를** 활용하여 우수한 성능을 달성
    - 이러한 정보 추출을 위해 **Stable Diffusion 모델이** **카메라 시점 제어를 학습**하도록 미세 조정하여, 벤치마크에서 최첨단 성능을 입증
    

## 5.1. Future Work

### From objects to scenes

- 본 접근법은 평범한 배경의 단일 객체 데이터셋으로 훈련
- RTMV 데이터셋에서 여러 객체가 있는 장면에 대한 강한 일반화를 입증했지만, GSO에서의 분포 내 샘플에 비해 품질이 여전히 저하
- 복잡한 배경이 있는 장면으로의 일반화는 앞으로의 주요 과제

# 6. Appendix

### C. Finetuning Stable Diffusion

- 렌더링된 데이터셋을 사용하여 새로운 뷰 합성을 수행하기 위해 사전 훈련된 스테이블 디퓨전 모델을 미세 조정
- 원래의 스테이블 디퓨전 네트워크는 다중 모드 텍스트 임베딩에 대해 조건화되지 않기 때문에, 조건 정보를 이미지에서 사용할 수 있도록 원래의 스테이블 디퓨전 아키텍처를 조정하고 미세 조정해야함.
- 이미지 CLIP 임베딩(차원 768)과 포즈 벡터(차원 4)를 연결하고, 디퓨전 모델 아키텍처와의 호환성을 확보하기 위해 또 다른 완전 연결 층(772 → 768)을 초기화
- 이 층의 학습률은 다른 층보다 10배 크게 조정
- 나머지 네트워크 아키텍처는 원래 스테이블 디퓨전과 동일하게 유지

### C.1 훈련 세부사항

- **8×A100-80GB 머신**에서 7일 동안 모델을 미세 조정

### C.2 추론 세부사항

- 새로운 뷰를 생성하기 위해 Zero-1-to-3는 RTX A6000 GPU에서 단 2초 소요
- 이전 작업에서는 일반적으로 NeRF를 훈련하여 새로운 뷰를 렌더링하는 데 상당한 시간 소요
- 비교적으로, 본 접근 방식은 3D 재구성과 새로운 뷰 합성의 순서를 반전시켜 새로운 뷰 합성 과정을 신속하고 불확실성 하에서 다양성을 포함하도록 함

### D. 3D Reconstruction

- 이미지에서 전체 3D 재구성을 실행하는 데 RTX A6000 GPU에서 약 30분이 소요