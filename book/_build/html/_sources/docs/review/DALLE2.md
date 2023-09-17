``` {admonition} Information
- **Title:** Hierarchical Text-Conditional Image Generation with CLIP Latents (arXiv 2022)

- **Reference**
    - Paper: [https://arxiv.org/pdf/2204.06125v1.pdf](https://arxiv.org/pdf/2204.06125v1.pdf)
    
- **Author:** SeonHoon Kim

- **Last updated on Sep. 18, 2023**
```

# DALLE2

DALLE2 는 2022년에 공개되어 세상을 놀라게 했습니다.
이미지 생성 능력도 뛰어났고, 이미지를 사용자 입맛에 맞게 조작할 수 있게 되었죠.

DALLE2 의 이름은 왜 DALL-E 일까요?
DALLE2 의 DALLE 는 초현실주의 화가 Salvador Dali 와 WALL-E 의 합성어입니다.
DALLE2 로 생성해낸 결과물이 과연 어떻길래 세상을 놀라게 했을까요?

- **DALL-E 2 결과물**

:::{figure-md} 
<img src="../../pics/DALLE2/img_00.png" alt="img_00" class="bg-primary mb-1" width="700px">
vibrant portrait of Salvador Dali with a robotic half face from DALLE2
:::

:::{figure-md} 
<img src="../../pics/DALLE2/img_01.png" alt="img_01" class="bg-primary mb-1" width="700px">
Salvador Dali 의 생전 모습
:::

위 그림은 DALLE2 에게 "vibrant portrait of Salvador Dali with a robotic half face" 를 prompt 로 주고 생성해낸 이미지입니다.
실제 Salvador dali 의 모습이 보이네요.
게다가 Salvador dali 의 초현실주의적 그림체가 반영된 것 같기도 합니다.
놀라운 이미지입니다.

아래의 corgi 그림은 어떤가요 ?
:::{figure-md} 
<img src="../../pics/DALLE2/img_02.png" alt="img_02" class="bg-primary mb-1" width="700px">
a corgi's head depicted as an explosion of a nebula from DALLE2
:::

corgi 의 모습을 성운의 폭발로 묘사해달라고 했을 때 생성된 그림입니다.
아래의 그림은, 실제 NASA 에서 촬영한 초신성 폭발의 잔해입니다.

정말 그럴듯하지 않나요?

:::{figure-md} 
<img src="../../pics/DALLE2/img_03.png" alt="img_03" class="bg-primary mb-1" width="700px">
This mosaic image, one of the largest ever taken by NASA's Hubble Space Telescope of the Crab Nebula, is a six-light-year-wide expanding remnant of a star's supernova explosion.
:::

- **학습 목표 및 주의사항**
    - 해당 paper 요약은 [DALL-E 2 paper](https://cdn.openai.com/papers/dall-e-2.pdf), [OpenAI blog](https://openai.com/dall-e-2), [AssemblyAI Youtube](https://www.youtube.com/watch?v=F1X4fHzF4mQ&t=360s&ab_channel=AssemblyAI), [Eden Meyer Youtube](https://www.youtube.com/watch?v=gmfI3B6pQTo&t=83s&ab_channel=EdanMeyer) 를 참고했습니다.
    - DALL-E 2 는 CLIP 과 Diffusion Model 을 통합시켰습니다. (최초는 x)
        - CLIP 은, 이미지와 text 를 학습한 multi-modal 모델입니다.
            - The fundamental principles of training CLIP are quite simple:
                1. First, all images and their associated captions are passed through their respective encoders, mapping all objects into an *m-*dimensional space.
                2. Then, the cosine similarity of each *(image, text)* pair is computed.
                3. The training objective is to simultaneously **maximize the cosine similarity** between N **correct** encoded image/caption ****pairs and **minimize the cosine similarity** between N - N **incorrect** encoded image/caption pairs.
                
    - 하지만 CLIP 을 사용하는 것이 정답은 아닙니다.
    DALL-E 2 는 22년 5월, 
    CLIP 을 사용하지 않은 IMAGEN 에게 SOTA 를 내주었습니다.
    
- **아키텍쳐 찍먹하기**
    
    특정 이미지 내의 Semantics 와 style 을 모두 포착해낼 수 있는
    CLIP 의 이미지 표현 능력을 끌어올리기 위해서, 
    저자들은 CLIP 과 Diffusion 모델을 통합한 Two-stage model 을 제안합니다.
    이것이 바로 DALLE2 인데요. 저자들은 이 모델을 unCLIP 이라고 부릅니다.
    
:::{figure-md} 
<img src="../../pics/DALLE2/img_06.png" alt="img_06" class="bg-primary mb-1" width="700px">
A high level overview of the architecture.
:::
    
DALLE2 paper 의 그림은 좀 복잡해보이니,
Assembly AI 의 Youtube 에서 제공하는 좀 더 단순화된 그림을 살펴볼게요.

:::{figure-md} 
<img src="../../pics/DALLE2/img_07.png" alt="img_07" class="bg-primary mb-1" width="700px">
A high level overview of the architecture from AssemblyAI youtube.
:::
[https://www.youtube.com/watch?v=F1X4fHzF4mQ&t=360s&ab_channel=AssemblyAI](https://www.youtube.com/watch?v=F1X4fHzF4mQ&t=360s&ab_channel=AssemblyAI)

    - **Prior** : 텍스트 캡션을 받아서, 상응하는 CLIP image embedding 을 생성합니다.
        - Autogregressive prior 와 Diffusion prior 를 비교하는 실험 수행했습니다.
        - Diffusion prior 가 computationally efficient 하고, 고품질 이미지 생성합니다. 따라서 후반부에는 Diffusion prior 만 사용해서 실험합니다.
    - **Decoder** : CLIP image embedding 을 받아서, 이미지를 생성합니다.
        - Diffusion 모델만 사용했습니다.
- **왜 CLIP 이랑 Diffusion 을 사용했을까요**
    - **CLIP**
        - CLIP 이 images representation 을 학습하는데 에 큰 성공을 거두고 있었습니다.
        - CLIP embeddings 는 image distribution shift 에 robust 했습니다.
        - CLIP embeddings 는 zero-shot capabilities 가 뛰어났습니다.
        - 다양한 vision & language tasks 에 fine-tuned 되어 SOTA 를 달성해냈습니다.
    - **Diffusion**
        - Diffusion 은 image 와 video generation taks 에서 SOTA 를 갱신하는 중이었죠.
        - non-deterministic 하게 만들 수 있습니다.
        이러한 Decoder 덕분에, CLIP image embedding 과 같은 
        **image representation 에 존재하지 않는 non-essential 한 details** 는 **변주하면서,** 
        **image representation 의 semantics 와 style 은 유지**할 수 있죠.
            
:::{figure-md} 
<img src="../../pics/DALLE2/img_08.png" alt="img_08" class="bg-primary mb-1" width="700px">
Variations of an input image by encoding with CLIP and then decoding with a diffusion model.
:::

- **아키텍쳐 설명 좀 자세히 해줘**
    
    :::{figure-md} 
    <img src="../../pics/DALLE2/img_09.png" alt="img_09" class="bg-primary mb-1" width="700px">
    A high level overview of the architecture from AssemblyAI youtube.
    :::    
    [https://www.youtube.com/watch?v=F1X4fHzF4mQ&t=360s&ab_channel=AssemblyAI](https://www.youtube.com/watch?v=F1X4fHzF4mQ&t=360s&ab_channel=AssemblyAI)
    
    - **Prior**
        - **input**
            - Caption 그 자체의 embedding vector 입니다.
            - **CLIP text embedding** 입니다.
        - **output**
            - **Generated CLIP Image embedding** 입니다.
        - **설명**
            - 사실 Prior 은 CLIP text embedding 만 조건으로 받는 것이 아니라 
            Caption 자체도 받습니다. (물론 embedding vector 로 받을 것) 
            CLIP text embedding 과, 그 Caption 은 서로 1대1 대응되기 때문에, 
            Duel-conditioning 이 문제될 것은 없다고 저자들은 변론합니다.
            - 샘플 퀄리티를 높이기 위해서, 
            2개의 CLIP image embeddings 를 생성한 후 
            주어진 CLIP text embedding 과 
            더 높은 dot product 를 갖는 CLIP image embedding 을 사용했다고 합니다.
    - **Decoder**
        - **Input**
            - CLIP text embedding
            - Generated CLIP Image embedding
        - **Output**
            - Generated Image
        - **설명**
            - modified GLIDE model 을 Decoder 로 사용했습니다.
            → 따라서, **projected CLIP text embeddings 를 아키텍쳐**에 통합시킬 수 있다고 주장합니다.
            
            어떻게 통합시키냐하면,

            1. GLIDE timestep embedding 에 추가하고, 
            2. 4개의 extra context tokens 을 만들어서 GLIDE text encoder 의 output sequence 에 concat 하는거죠.
            
            이 방법으로 **CLIP image embeddings 를 받아서, 원본 영상을 생성하는 것** 입니다.
                
            :::{figure-md} 
            <img src="../../pics/DALLE2/img_10.png" alt="img_10" class="bg-primary mb-1" width="700px">
            GLIDE training process
            :::    
                
            - GLIDE 를 수정해 사용함으로써, GLIDE 가 가지고 있던 text-conditional photorealistic image generation capabilities 를 활용할 수 있다고 주장합니다.

- **그렇다면 왜 Prior 가 필요할까요?**
    1. **To obtain a full generative model of images**, we combine the CLIP image embedding decoder with a prior model, which generates possible CLIP image embeddings from a given text caption
    2. **아래 세 가지 아키텍쳐를 비교하는 실험 수행**
    (1) GLIDE 모델처럼, text 의 token embeddings 만 조건으로 주어 실험
    (2) 추가적으로, CLIP text embeddings 를 조건으로 주어 실험
    (3) 추가적으로, CLIP image embeddings 를 생성해내는 Prior 를 갖추고 실험
    
    결과 (3) 이 가장 훌륭했습니다. 특히 image diversity 가 뛰어났습니다.
        
    :::{figure-md} 
    <img src="../../pics/DALLE2/img_11.png" alt="img_11" class="bg-primary mb-1" width="700px">
    3가지 경우의 아키텍쳐에 따른 실험 결과 from AssemblyAI youtube.
    :::
    :::{figure-md} 
    <img src="../../pics/DALLE2/img_12.png" alt="img_12" class="bg-primary mb-1" width="700px">
    Samples using different conditioning signals for the same decoder.
    :::
                
        - 하지만.. **95% 의 학습 시간 동안, (3) 방식으로 학습한 Decoder 를, (1) 과 (2) 방식에 그대로 적용해 실험했습니다.** 따라서 공정한 실험이라고 보긴 어려울 것 같습니다.
        - 또한.. **Decoder 를 True CLIP Image embeddings 와 Generated CLIP Image embeddings 로 각각 학습시켰을 때의 성능 비교 실험은 없습니다.**
        - 개인적으로 저는 이러한 결과들을 보고, Prior 를 반드시 써야하는 근거에 대한 설득력이 조금 떨어진다고 생각했습니다.
- **왜 CLIP 을 써야할가요?**
    1. CLIP 은 어떤 객체를 묘사한 텍스트와, 그 객체의 시각적 발현 사이의 의미론적 관계를 학습했습니다. 따라서 저자들은 이러한 CLIP 의 능력이 Text-to-Image task 에서 매우 중요하다고 주장합니다.
    2. **CLIP 을 활용한 덕분에 이미지를 Manipulation 할 수 있습니다.**
        
        ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2012.png)
        
- **그래서? 이 모델 좋아?**
    - **Evaluation**
        - 주어진 Caption 에 대한 GLIDE 의 생성물과 unCLIP 의 생성물을 
        사람들에게 제시하고,
        **Photorealism, Caption Similarity, Diversity** 에 대해서 **평가**하도록 함.
        1. GLIDE 에 비해서 **Photorealism, Caption Similarity,** 은 Comparable 하다.
        ~~(안 좋다)~~ 
        2. 하지만, **Diversity** 는 훨씬 뛰어나다.
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2013.png)
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2014.png)
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2015.png)
            
    - **Image Manipulations**
        - Bipartite Representation
            - unCLIP 구조 덕분에, 주어진 이미지 x 를 (z_i, x_T) 와 같은 bipartite latent representation 으로 인코딩 할 수 있음
            - 이 latent space 를 활용해서, Image manipulation 을 수행할 수 있다.
            - x_T 는 DDIM inversion 을 z_i 가 condition 된 x 에 적용해 얻으며, 
            Decoder 가 x 를 복원하는데 필요한 잔여 정보들을 지님
        
        1. **Variations**
        
        ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%208.png)
        
        - Non-essential details 를 변주하기 위해서, 
        bipartite representation 에 DDIM with η > 0 for sampling decoder 를 적용한다.
        - η = 0 일 때, decoder 는 deterministic 해지고 x 자체를 복원해낸다.
        - η 가 커질수록, sampling steps 에는 stochasticity 가 생기고, 원본 이미지 x 근처에서 perceptually “centereed” 된 variations 를 만들어낼 것이다.
        - η 를 키우면, 우리는 CLIP image embedding 에 어떤 정보가 존재하고 어떤 정보가 유실되었는지 탐색 가능
        **→ CLIP latent space 를 탐색해낼 수 있다 !**
        
        1. **Interpolations**
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2016.png)
            
            - input image 두 장의 CLIP image embeddings 를 interpolation 해서 
            Decoder 에 준다면, interpolated image 를 생성할 수 있다.
            
        2. **Text Diffs**
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2017.png)
            
            - **어떤 이미지와 그 캡션이 주어져있을 때,
            그 이미지를 우리가 원하는 target text prompt 에 맞게
            조작할 수도 있음**
            - **Method**
            **z_t0 = current CLIP text embedding
            z_t = target CLIP text embedding**
                
                ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2018.png)
                
            - 주어진 이미지의 **CLIP image embdding z_i** 를 바로 이 **text diff vector 와 interpolate 해서 Decoding** 하면 이미지가 조작된다.
    - **Robustness against typographic attaks**
        - **typographic attacks** : 이미지 내 사물 위에, 글씨가 쓰여 있는 경우이다.
        - Multimodal 로 학습한 CLIP 은 텍스트에 있는 정보를 더 많이 활용해 
        사물을 판단하는 경향이 있음
            1. unCLIP 의 Decoder 모델에 “iPod” 텍스트 종이가 붙은 사과를 보고 분류를 수행해보았다. 
            2. 역시, “Granny Smith” 의 예측 확률을 거의 0 에 가깝다고 판단했다.
            3. 그럼에도 불구하고, 사과의 사진으로 recover 해낸다.
                
                ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2019.png)
                
        
- **이 모델, 단점은 없어?**
    1. **객체(cubes)와 그들의 속성(colors) 을 매칭시키는 능력이 떨어짐**
        
        ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2020.png)
        
    
    1. **텍스트를 일관성있게 생성하는 능력이 떨어짐**
        
        ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2021.png)
        
    2. **복잡한 상황에서 디테일을 묘사하는 능력이 떨어짐**
        
        ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2022.png)
        
    
- **Method - Training**
    - unCLIP 모델의 아키텍쳐에 대한 수학적 justify 를 하고 있음
    - Training 데이터셋의 이미지를 x 라 한다.
    - 그에 상응하는 text captions 을 y 라 한다.
    - 각각에 대한 embeddings 인 Z_i, Z_t 를 기존의 CLIP 으로  생성.
        - image **x —CLIP Image encoder—> Z_i** image embeddings
        - text caption **y —CLIP text encoder—> Z_t** text embeddings
    
    - 저자의 주장
        - unCLIP 으로, text caption y 로부터 image x 를 샘플링할 수 있다.
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2023.png)
            
        - ***The first equality holds because z_i is a deterministic function of x.***
        - ***The second equality holds because of the chain rule.***
    
    - **내 부가 설명**
        - z_t 도 y 의 deterministic function 이므로, 다음과 같이 쓸 수 있다.
            
            $$
            P(x|y) = P(x, z_i|y, z_t) = P(x|z_i, y, z_t)P(z_i|y, z_t)
            $$
            
        - Prior 를 사용해 Z_t 로부터 Z_i 를 샘플링하고,
        Decoder 를 사용해 x 를 샘플링함으로써
        True conditional distribution 인 P(x|y) 샘플링이 가능해짐
    
- **DALL-E 2 Bias**
    
    [https://github.com/openai/dalle-2-preview/blob/main/system-card.md](https://github.com/openai/dalle-2-preview/blob/main/system-card.md)
    
    - **현재 OpenAI 가 DALL-E 2 의 Safety 를 위해 하고 있는 노력**
        1. 학습 데이터에서 violent, hate, or adult images 를 제거함으로써
        이러한 이미지들에 DALL-E 2 가 노출되는 시간을 최소화함
        2. Safety policies 를 위반한 text prompts 혹은 생성된 images 를 자정하는 시스템 보유
        3. 신뢰할 수 있는 전문가들과 DALL-E 2 에 대한 사전 검토를 진행했음
    - **DALL-EVAL : 이미지 생성형 AI 의 Bias 를 평가하는 방법 소개**
        
        [https://arxiv.org/pdf/2202.04053.pdf](https://arxiv.org/pdf/2202.04053.pdf)
        
        [https://github.com/j-min/DallEval](https://github.com/j-min/DallEval)
        
        - **Contribution**
            - Text to Image Generation models 의 시각적 추론 능력 3가지를 평가하기 위한 데이터셋 제공
            - 최근의 모델들이 object recognition skill 은 상대적으로 뛰어나지만, object counting 및 spaial relation 이해 능력은 떨어짐을 발견
            - Text to Image Generation models 에 존재하는 **gender 및 skin tone biases 를 측정하는 evaluation metrics 와 그 분석 결과를 제시한 최초의 논문**
                - web 에서 image-text pairs 를 학습한 현재 T2I 생성 모델들은 특정한 gender/skin tone biases 를 학습했음을 보여줌
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2024.png)
            
        - **Social Bias Evaluation 측정 방법 (Sec. 4.2.1 참고)**
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2025.png)
            
            1. **Diagnostic prompts 로 이미지들을 생성한다.**
                1. ex) “a person who works as a nurse”
                2. 총 252개의 Diagnostic prompts 제공
            2. **Gender, Skin tone, Attributes 를 생성된 이미지로부터 탐지**
                1. using automated detection models and verify the reliability of detection models with human evaluation
                2. **Gender**
                    1. BLIP-2 라는 모델에 생성된 영상을 주면서 영상 내 사람의 성별을 맞추게 함. BLIP-2 의 답변을 기반으로 Gender Bias 측정
                3. **Skin tone**
                    1. 신경망으로 facial landmark 를 추출하고, illumination 을 측정
                4. **Attributes**
                    1. BLIP-2 라는 모델에 생성된 영상을 주면서 영상 내 사람의 복장을 맞추게 함. BLIP-2 의 답변을 기반으로 Attributes Bias 측정
            3. 탐지된 Gender, Skin tone, Attributes 가 unbiased uniform distribution 으로부터 얼마나 skewed 되어있는지 측정한다.
        - **실험 결과**
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2026.png)
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2027.png)
            
            ![Untitled](DALL-E%202%20paper%2064d873e56fd8435b98a6713546cc2405/Untitled%2028.png)



이번 시간에는 Google Research 에서 소개하는 Imagen 모델 기반의 text-guided image inpainting 모델 Imagen Editor 와 text-guided impainting 의 평가기법 EditBench 에 대해 알아볼 예정입니다. 

Text-guided image inpainting 에서 기존에는 mask 영역을 random 하게 지정하여 학습을 진행했습니다. 이는 입력된 text prompt 와 무관한 영역을 masking 하게 됨으로써 모델이 prompt 를 참조하지 않고 오로지 image content 만으로 학습하게 되는 현상이 발생합니다. Imagen Editor 는 이를 해결하기 위해 Object Masking 기법을 소개합니다. Prompt 에 해당하는 객체 전체를 masking 함으로써 모델이 text prompt 를 더 참조할 수 있도록 유도하는 것이 목표입니다. SSD MobileNet v2 모델을 Object Detector 로 사용함으로써 모델 성능이 크게 개선되는 부분을 확인할 수 있었다고 합니다.  

:::{figure-md} 
<img src="../../pics/DALLE2/img_01.png" alt="img_01" class="bg-primary mb-1" width="700px">

Effect of Object Masking
:::

Imagen Editor 에서 또 다른 특징은 Imagen 모델 기반의 cascaded diffusion model architecture 를 지니고 있다는 점입니다. 이때, SR3, Palette, GLIDE 와 유사하게 이미지와 mask 가 Encoder 를 거친 후, diffusion latent 와 concatenate 하면서 conditioning input 으로 들어가게 되며, 모두 1024x1024 해상도를 가진다고 합니다. 따라서, base diffusion 64x64 모델 그리고 64x64 → 256x256 super resolution 모델에 입력 시, downsampling 작업 후 모델 input 으로 입력합니다. 또한, conditioning 이미지와 mask 없을 시 Imagen 모델을 사용하는 것과 동일한 효과를 내기 위해, 새로 추가되는 input channel weights 는 0으로 초기화해서 학습을 진행했다고 소개합니다. 

:::{figure-md} 
<img src="../../pics/DALLE2/img_02.png" alt="img_02" class="bg-primary mb-1" width="700px">

Imagen Editor Architecture
:::

Imagen 에서 소개되었던 Classifier-Free Guidance 를 동일하게 사용하고, 이때 guidance weight 를 1부터 30 까지 범위 내에서 변화시키는 oscillating guidance 기법을 적용함으로써 생성된 이미지 퀄리티 및 text-image alignment 가 상승되는 효과를 볼 수 있었다고 합니다. 

논문에서는 Imagen Editor 와 같은 text-guided image inpainting 모델들을 평가할 수 있는 새로운 benchmark EditBench 를 제시합니다. 240개의 (image, mask) 쌍으로 데이터셋이 구축되어있고, 각 쌍마다 3가지의 prompt 로 생성된 이미지로 사람이 모델 성능을 측정하게 됩니다. Automatic Evaluation Metric 으로는 CLIPScore, 그리고 CLIP-R-Prec 를 사용했습니다.

EditBench 이미지 데이터셋의 절반은 open source 로 공개된 computer vision 데이터셋으로부터 수집되었고, 나머지 절반은 text-to-image 모델로 생성해서 구축했습니다. 이때, *attribute-object-scene* 의 요소들을 모두 갖추도록 이미지들을 수집 및 생성했습니다. 

- Attributes (material, color, shape, size, count)
- Objects (common, rare, text rendering)
- Scenes (indoor, outdoor, realistic, paintings)

예를 들어서, ‘a=metal|o=cat|s=outdoor’ 요소들을 포함하는 문구를 ‘a metal cat standing in the middle of a farm field’ 처럼 생성하는 것입니다. 앞써 언급한 3가지 prompt 는 해당사진처럼 *Mask-Simple*, *Mask-Rich*, 그리고 *Full* 로 정의합니다. 

:::{figure-md} 
<img src="../../pics/DALLE2/img_03.png" alt="img_03" class="bg-primary mb-1" width="600px">

EditBench example
:::

데이터셋 구축시, mask 크기도 다양하게 설정하여 mask 크기에 따른 모델 성능도 확인할 수 있었습니다. 성능을 측정해본 결과, Object masking 으로 학습한 모델이 random masking 으로 학습한 모델보다 small/medium masks 에서 성능적으로 월등히 좋다는 것을 확인할 수 있습니다. 

:::{figure-md} 
<img src="../../pics/DALLE2/img_04.png" alt="img_04" class="bg-primary mb-1" width="500px">

Human Evaluations on EditBench
:::

또한, object-rendering 에 비해 text-rendering 성능이 저하되는 부분을 확인할 수 있고, material/color/size 속성보다 count/size 속성에 더 취약한 부분도 확인할 수 있었습니다. 

:::{figure-md} 
<img src="../../pics/DALLE2/img_05.png" alt="img_05" class="bg-primary mb-1" width="500px">

Imagen Editor failure cases by attribute
:::

마지막으로, 동일한 prompt 에 대해 Stable Diffusion, DALL-E2, Imagen Editor 모델로 inpainting 한 결과를 비교한 예시 사진입니다.

:::{figure-md} 
<img src="../../pics/DALLE2/img_06.png" alt="img_06" class="bg-primary mb-1" width="500px">

Example model outputs for Mask-Simple vs MaskRich prompts
:::
