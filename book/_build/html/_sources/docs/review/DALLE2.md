``` {admonition} Information
- **Title:** Hierarchical Text-Conditional Image Generation with CLIP Latents (arXiv 2022)

- **Reference**
    - Paper: [https://arxiv.org/pdf/2204.06125v1.pdf](https://arxiv.org/pdf/2204.06125v1.pdf)
    
- **Author:** SeonHoon Kim

- **Last updated on Sep. 18, 2023**
```

# DALLE2

DALLE2 는 2022년에 공개되어 세상을 놀라게 했습니다.<br>
이미지 생성 능력도 뛰어났고, 이미지를 사용자 입맛에 맞게 조작할 수 있게 되었죠.

DALLE2 의 이름은 왜 DALL-E 일까요?<br>
DALLE2 의 DALLE 는 초현실주의 화가 Salvador Dali 와 WALL-E 의 합성어입니다.<br>
DALLE2 로 생성해낸 결과물이 과연 어떻길래 세상을 놀라게 했을까요?

<br>

- **DALL-E 2 결과물**

    :::{figure-md} 
    <img src="../../pics/DALLE2/img_01.png" alt="img_01" class="bg-primary mb-1" width="350px">

    Salvador Dali 의 생전 모습
    :::

    :::{figure-md} 
    <img src="../../pics/DALLE2/img_00.png" alt="img_00" class="bg-primary mb-1" width="350px">

    vibrant portrait of Salvador Dali with a robotic half face from DALLE2
    :::

    위 그림은 DALLE2 가 생성해낸 "vibrant portrait of Salvador Dali with a robotic half face" 이미지입니다.<br>
    실제 Salvador dali 의 모습이 보이네요.<br>
    게다가 Salvador dali 의 초현실주의적 그림체가 반영된 것 같기도 합니다.<br>
    놀라운 이미지입니다.

    아래의 corgi 그림은 어떤가요 ?
    :::{figure-md} 
    <img src="../../pics/DALLE2/img_02.png" alt="img_02" class="bg-primary mb-1" width="350px">

    a corgi's head depicted as an explosion of a nebula from DALLE2
    :::

    corgi 의 모습을 성운의 폭발로 묘사해달라고 했을 때 생성된 그림입니다.<br>
    아래의 그림은, 실제 NASA 에서 촬영한 초신성 폭발의 잔해입니다.

    정말 그럴듯하지 않나요?

    :::{figure-md} 
    <img src="../../pics/DALLE2/img_03.png" alt="img_03" class="bg-primary mb-1" width="350px">

    This mosaic image, one of the largest ever taken by NASA's Hubble Space Telescope of the Crab Nebula, is a six-light-year-wide expanding remnant of a star's supernova explosion.
    :::

<br>

- **학습 목표 및 주의사항**
    - 본 포스팅에서는 DALLE2 paper 의 내용을 비선형적으로 살펴봅니다.<br>
    마치 오픈월드 게임처럼 말이죠.<br>
    핵심이 되는 질문들을 던지며, DALLE2 의 아키텍쳐를 파헤쳐 볼 겁니다.
    - 본 포스팅은 [DALL-E 2 paper](https://cdn.openai.com/papers/dall-e-2.pdf), [OpenAI blog](https://openai.com/dall-e-2), [AssemblyAI Youtube](https://www.youtube.com/watch?v=F1X4fHzF4mQ&t=360s&ab_channel=AssemblyAI), [Eden Meyer Youtube](https://www.youtube.com/watch?v=gmfI3B6pQTo&t=83s&ab_channel=EdanMeyer) 를 참고했습니다.
    - 본격적으로 학습하기 전에 알아야할 것은, CLIP 모델입니다.
        - CLIP 은, 이미지와 text 를 학습한 multi-modal 모델입니다.
            - The fundamental principles of training CLIP are quite simple:
                1. First, all images and their associated captions are passed through their respective encoders, mapping all objects into an m-dimensional space.
                2. Then, the cosine similarity of each *(image, text)* pair is computed.
                3. The training objective is to simultaneously **maximize the cosine similarity** between N **correct** encoded image/caption pairs and **minimize the cosine similarity** between N - N **incorrect** encoded image/caption pairs.
    - DALL-E 2 는 CLIP 과 Diffusion Model 을 통합시켰습니다. (최초는 x)           
    - 하지만 CLIP 을 사용하는 것이 정답은 아닙니다.<br>
    DALL-E 2 는 22년 5월, CLIP 을 사용하지 않은 IMAGEN 에게 SOTA 를 내주었습니다.

<br>

- **아키텍쳐 찍먹하기**
    
    특정 이미지 내의 Semantics 와 style 을 모두 포착해낼 수 있는 CLIP 의 이미지 표현 능력을 끌어올리기 위해서,<br>
    저자들은 CLIP 과 Diffusion 모델을 통합한 Two-stage model 을 제안합니다.<br>
    이것이 바로 DALLE2 인데요.<br>
    저자들은 이 모델을 unCLIP 이라고 부릅니다.
    
    :::{figure-md} 
    <img src="../../pics/DALLE2/img_06.png" alt="img_06" class="bg-primary mb-1" width="700px">

    A high level overview of the architecture.
    :::
    
    DALLE2 paper 의 그림은 좀 복잡해보이니,<br>
    Assembly AI 의 Youtube 에서 제공하는 좀 더 단순화된 그림을 살펴볼게요.

    :::{figure-md} 
    <img src="../../pics/DALLE2/img_07.png" alt="img_07" class="bg-primary mb-1" width="700px">

    A high level overview of the architecture from AssemblyAI youtube.
    :::
    [https://www.youtube.com/watch?v=F1X4fHzF4mQ&t=360s&ab_channel=AssemblyAI](https://www.youtube.com/watch?v=F1X4fHzF4mQ&t=360s&ab_channel=AssemblyAI)

    Prior 와 Decoder 가 DALLE2 의 핵심이 되는 모델인 것 같네요.

    - **Prior** : 텍스트 캡션을 받아서, 상응하는 CLIP image embedding 을 생성합니다.
        - 본 논문에서는 Autogregressive prior 와 Diffusion prior 를 비교하는 실험 수행했습니다.
        - Diffusion prior 가 computationally efficient 하고, 고품질 이미지 생성합니다.<br>
        따라서 후반부에는 Diffusion prior 만 사용해서 실험합니다.
    - **Decoder** : CLIP image embedding 을 받아서, 이미지를 생성합니다.
        - Diffusion 모델만 사용했습니다.

<br>

- **왜 CLIP 이랑 Diffusion 을 사용했을까요?**
    - **CLIP**
        - CLIP 이 images representation 을 학습하는데 에 큰 성공을 거두고 있었습니다.
        - CLIP embeddings 는 image distribution shift 에 robust 했습니다.
        - CLIP embeddings 는 zero-shot capabilities 가 뛰어났습니다.
        - 다양한 vision & language tasks 에 fine-tuned 되어 SOTA 를 달성해냈습니다.
    - **Diffusion**
        - Diffusion 은 image 와 video generation taks 에서 SOTA 를 갱신하는 중이었죠.
        - non-deterministic 하게 만들 수 있습니다.<br>
        이러한 Decoder 덕분에, CLIP image embedding 과 같은 <br>
        **image representation 에 존재하지 않는 non-essential 한 details** 는 **변주하면서,** <br>
        **image representation 의 semantics 와 style 은 유지**할 수 있죠.
            
        :::{figure-md} 
        <img src="../../pics/DALLE2/img_08.png" alt="img_08" class="bg-primary mb-1" width="700px">

        Variations of an input image by encoding with CLIP and then decoding with a diffusion model.
        :::

        위 왼쪽의 그림처럼, Salvador dali 의 그림에서 중요한 objects 들은 보존됩니다.<br>
        하지만 그들이 표현되는 방식이나 전체적인 그림의 style 은 조금씩 바뀝니다.<br>
        그럼에도, Salvador dali 특유의 초현실주의적 화풍은 유지되는 것 같네요.<br>
        Diffusion Decoder 덕분에, **Non-essential details** 는<br>
        마치 **변주곡처럼 매번 새롭게 연주**해낼 수 있는겁니다.


<br>

- **아키텍쳐 파헤치기**
    
    :::{figure-md} 
    <img src="../../pics/DALLE2/img_09.png" alt="img_09" class="bg-primary mb-1" width="700px">

    A high level overview of the architecture from AssemblyAI youtube.
    :::    
    [https://www.youtube.com/watch?v=F1X4fHzF4mQ&t=360s&ab_channel=AssemblyAI](https://www.youtube.com/watch?v=F1X4fHzF4mQ&t=360s&ab_channel=AssemblyAI)
    <br>
    
    이번에는 DALLE2 의 아키텍쳐를 좀 더 자세히 살펴보죠.

    - **Prior**
        - **input**
            - Caption 그 자체의 embedding vector 입니다.
            - **CLIP text embedding** 입니다.
        - **output**
            - **Generated CLIP Image embedding** 입니다.
        - **설명**
            - 사실 Prior 은 CLIP text embedding 만 조건으로 받는 것이 아니라 Caption 자체도 받습니다.<br>
            (물론 embedding vector 로 받겠죠)<br>
            CLIP text embedding 과, 그 Caption 은 서로 1대1 대응되기 때문에,<br>
            Duel-conditioning 이 문제될 것은 없다고 저자들은 변론합니다.
            - 샘플 퀄리티를 높이기 위해서 2개의 CLIP image embeddings 를 생성한 후 <br>
            주어진 CLIP text embedding 과 더 높은 dot product 를 갖는 CLIP image embedding 을 사용했다고 합니다.
    - **Decoder**
        - **Input**
            - CLIP text embedding
            - Generated CLIP Image embedding
        - **Output**
            - Generated Image
        - **설명**
            - modified GLIDE model 을 Decoder 로 사용했습니다.<br>
            → 따라서, **projected CLIP text embeddings 를 아키텍쳐**에 통합시킬 수 있다고 주장합니다.
            <br>
            어떻게 통합시키냐하면,

            1. GLIDE timestep embedding 에 추가하고,
            2. 4개의 extra context tokens 을 만들어서 GLIDE text encoder 의 output sequence 에 concat 하는거죠.
            <br>
            이 방법으로 **CLIP image embeddings 를 받아서, 원본 영상을 생성하는 것** 입니다.
                
            :::{figure-md} 
            <img src="../../pics/DALLE2/img_10.png" alt="img_10" class="bg-primary mb-1" width="700px">

            GLIDE training process
            :::    
                
            - GLIDE 를 수정해 사용함으로써 GLIDE 가 가지고 있던<br>
            text-conditional photorealistic image generation capabilities 를 활용할 수 있다고 주장합니다.

<br>

- **그렇다면 왜 Prior 가 필요할까요?**
    1. **To obtain a full generative model of images**, <br>
    we combine the CLIP image embedding decoder with a prior model, <br>
    which generates possible CLIP image embeddings from a given text caption <br>

    라고 하지만.. 딱히 와닿지는 않습니다.<br>
    하지만 아직 실망하긴 이릅니다.<br>
    Prior 의 유무에 따라, 생성된 이미지의 품질을 비교하는 실험을 수행했다고 합니다.<br>
    한번 살펴볼까요?

    2. **아래 세 가지 아키텍쳐를 비교하는 실험 수행**<br>
    (1) GLIDE 모델처럼, text 의 token embeddings 만 조건으로 주어 실험<br>
    (2) 추가적으로, CLIP text embeddings 를 조건으로 주어 실험<br>
    (3) 추가적으로, CLIP image embeddings 를 생성해내는 Prior 를 갖추고 실험<br>
    <br>
    실험 결과, (3) 이 가장 훌륭했습니다.<br>
    특히 image diversity 가 뛰어났습니다.

    :::{figure-md} 
    <img src="../../pics/DALLE2/img_11.png" alt="img_11" class="bg-primary mb-1" width="700px">

    3가지 경우의 아키텍쳐에 따른 실험 결과 from AssemblyAI youtube.
    :::

    :::{figure-md} 
    <img src="../../pics/DALLE2/img_12.png" alt="img_12" class="bg-primary mb-1" width="700px">

    Samples using different conditioning signals for the same decoder.
    :::

    그렇지만, 의문이 말끔히 해소되지는 않습니다.
    왜냐하면..

    - **95% 의 학습 시간 동안, (3) 방식으로 학습한 Decoder 를,**<br>
    **(1) 과 (2) 방식에 그대로 적용해 실험했습니다.** <br>
    따라서 공정한 실험이라고 보긴 어려울 것 같습니다.
    - **Decoder 를, True CLIP Image embeddings 와 Generated CLIP Image embeddings 로**<br>
    **각각 학습시켰을 때의 성능 비교 실험은 없습니다.**
    <br>
    개인적으로 저는 이러한 결과들을 보고,<br>
    Prior 를 반드시 써야하는 근거에 대한 설득력이 떨어진다고 생각했습니다.

<br>

- **왜 CLIP 을 써야할까요?**
    1. CLIP 은 어떤 객체를 묘사한 텍스트와, 그 객체의 시각적 발현 사이의 의미론적 관계를 학습했습니다. <br>
    따라서 저자들은 이러한 CLIP 의 능력이 Text-to-Image task 에서 매우 중요하다고 주장합니다.
    2. **CLIP 을 활용한 덕분에 이미지를 Manipulation 할 수 있습니다.**
        
    :::{figure-md} 
    <img src="../../pics/DALLE2/img_13.png" alt="img_13" class="bg-primary mb-1" width="700px">

    Text diffs applied to images by interpolating between their CLIP image embeddings and a normalised difference of the CLIP text embeddings produced from the two descriptions.
    :::

    어떻게 이미지를 Manipulation 하는지는 곧 자세히 살펴보겠습니다.

<br>

- **그래서 이 모델은 뭐가 좋은가요?**
    - **Evaluation 결과, Diversity 가 뛰어났습니다.**
        - 모델을 평가하기 위해서,<br>
        주어진 Caption 에 대한 GLIDE 의 생성물과 unCLIP 의 생성물을 사람들에게 제시하고,<br>
        **Photorealism, Caption Similarity, Diversity** 에 대해서 **점수를 매기도록** 했습니다.<br>
        
            
        :::{figure-md} 
        <img src="../../pics/DALLE2/img_14.png" alt="img_14" class="bg-primary mb-1" width="700px">

        Samples when increasing guidance scale for both unCLIP and GLIDE.
        :::

        :::{figure-md} 
        <img src="../../pics/DALLE2/img_15.png" alt="img_15" class="bg-primary mb-1" width="700px">

        Comparison of unCLIP and GLIDE for different evaluations.
        :::

        :::{figure-md} 
        <img src="../../pics/DALLE2/img_16.png" alt="img_16" class="bg-primary mb-1" width="700px">

        FID versus guidance scale for unCLIP and GLIDE.
        :::

        결론은 다음과 같습니다.
        1. GLIDE 에 비해서 **Photorealism, Caption Similarity,** 은 Comparable 했습니다.<br>
        (안 좋다.)
        2. 하지만, **Diversity** 는 훨씬 뛰어났습니다.

    <br>

    - **Image Manipulations 가 가능합니다.**
        - Bipartite Representation
            - unCLIP 구조 덕분에, <br>
            주어진 이미지 x 를 (z_i, x_T) 와 같은 bipartite latent representation 로 인코딩 가능합니다.
            - 이 latent space 를 활용해서, Image manipulation 을 수행할 수 있습니다.
            - x_T 는 DDIM inversion 을 z_i 가 condition 된 x 에 적용해 얻으며,<br> 
            Decoder 가 x 를 복원하는데 필요한 잔여 정보들을 지닙니다.
        
        <br>

        1. **Variations**

            :::{figure-md} 
            <img src="../../pics/DALLE2/img_17.png" alt="img_17" class="bg-primary mb-1" width="700px">

            Variations of an input image by encoding with CLIP and then decoding with a diffusion model.
            :::
            
            - Non-essential details 를 변주하기 위해서,<br> 
            bipartite representation 에 DDIM with η > 0 for sampling decoder 를 적용합니다.
            - η = 0 일 때, decoder 는 deterministic 해지고 x 자체를 복원해냅니다.
            - η 가 커질수록, sampling steps 에는 stochasticity 가 생기고,<br> 
            원본 이미지 x 근처에서 perceptually “centereed” 된 variations 를 만들어낼 것입니다.
            - η 를 키우면, 우리는 CLIP image embedding 에 어떤 정보가 존재하고 어떤 정보가 유실되었는지 탐색 가능합니다.<br> 
            **→ 즉, CLIP latent space 를 탐색해낼 수 있는거죠 !**
        
        <br>

        2. **Interpolations**
            
            :::{figure-md} 
            <img src="../../pics/DALLE2/img_18.png" alt="img_18" class="bg-primary mb-1" width="700px">

            Variations between two images by interpolating their CLIP image embedding and then decoding with a diffusion model.
            :::
            
            - 이런 것도 됩니다.<br>
            input image 두 장의 CLIP image embeddings 를 interpolation 해서 Decoder 에 준다면,<br>
            interpolated image 를 생성할 수 있습니다.
        
        <br>

        3. **Text Diffs**
            
            :::{figure-md} 
            <img src="../../pics/DALLE2/img_19.png" alt="img_19" class="bg-primary mb-1" width="700px">

            Text diffs applied to images by interpolating between their CLIP image embeddings and a normalised difference of the CLIP text embeddings produced from the two descriptions.
            :::
            
            - **어떤 이미지와 그 캡션이 주어져있을 때,<br> 
            그 이미지를 우리가 원하는 target text prompt 에 맞게 조작할 수도 있습니다.**
            - **Method**
                - **z_t0 = current CLIP text embedding** 이고,
                - **z_t = target CLIP text embedding** 이라면,
                
                :::{figure-md} 
                <img src="../../pics/DALLE2/img_19_2.png" alt="img_19_2" class="bg-primary mb-1" width="400px">

                text diff method
                :::
                
            - 주어진 이미지의 **CLIP image embdding z_i** 를 <br>
            바로 이 **text diff vector 와 interpolate 해서 Decoding** 하면 이미지가 조작됩니다.
    
    <br>
    
    - **typographic attaks 에 대해서, Robust 합니다.**
        - **typographic attacks** : 이미지 내 사물 위에, 글씨가 쓰여 있는 경우입니다.
        - Multimodal 로 학습한 CLIP 은 텍스트에 있는 정보를 더 많이 활용해<br> 
        사물을 판단하는 경향이 있습니다.
            1. unCLIP 의 Decoder 모델에 “iPod” 텍스트 종이가 붙은 사과를 보고 분류를 수행해보았습니다. 
            2. 역시, “Granny Smith” 의 예측 확률을 거의 0 에 가깝다고 판단했습니다.
            3. 그럼에도 불구하고, 사과의 사진으로 recover 해냅니다.    
            :::{figure-md} 
            <img src="../../pics/DALLE2/img_20.png" alt="img_20" class="bg-primary mb-1" width="700px">

            Variations of images featuring typographic attacks
            :::
            
            이처럼 DALLE2 는 typographic attacks 에 더욱 robust 합니다.

<br>

- **이 모델, 단점은 없나요?**
    
    <br>

    1. **객체(cubes)와 그들의 속성(colors) 을 매칭시키는 능력이 떨어집니다.**
        
    :::{figure-md} 
    <img src="../../pics/DALLE2/img_21.png" alt="img_21" class="bg-primary mb-1" width="700px">

    Samples from unCLIP and GLIDE for the prompt “a red cube on top of a blue cube”.
    :::

    위 그림처럼, 파란 큐브 위에 빨간 큐브를 그려달라고 했을 때, <br>
    DALLE2 는 **아래의 큐브와 위의 큐브에 각각 어떤 색상 (attributes) 를 부여해야할지** 헷갈려합니다.
    
    <br>

    2. **텍스트를 일관성있게 생성하는 능력이 떨어집니다**
        
    :::{figure-md} 
    <img src="../../pics/DALLE2/img_22.png" alt="img_22" class="bg-primary mb-1" width="700px">

    Samples from unCLIP for the prompt, “A sign that says deep learning.”
    :::
    
    물론 이것은 DALLE2 만의 문제는 아닙니다.<br>
    많은 text-to-image models 가 어려워하는 문제입니다.
    
    <br>
    
    3. **복잡한 상황에서 디테일을 묘사하는 능력이 떨어집니다**
        
    :::{figure-md} 
    <img src="../../pics/DALLE2/img_23.png" alt="img_23" class="bg-primary mb-1" width="700px">

    unCLIP samples show low levels of detail for some complex scenes.
    :::
    
    복잡한 네온 사인들의 디테일들이 좀 떨어지는 것을 확인하실 수 있습니다.

<br>

- **Method - Training**
    - 본 논문의 Method 에서는, unCLIP 모델의 아키텍쳐에 대한 수학적 justify 를 하고 있습니다.
    - Training 데이터셋의 이미지를 x 라 합시다.
    - 그에 상응하는 text captions 을 y 라 합시다.
    - 각각에 대한 embeddings 인 Z_i, Z_t 를 기존의 CLIP 으로  생성합니다.
        - image **x —CLIP Image encoder—> Z_i** image embeddings
        - text caption **y —CLIP text encoder—> Z_t** text embeddings
    
    <br>

    - 저자의 주장
        - unCLIP 으로, text caption y 로부터 image x 를 샘플링할 수 있다고 합니다.
            
            :::{figure-md} 
            <img src="../../pics/DALLE2/img_24.png" alt="img_24" class="bg-primary mb-1" width="700px">

            P(x|y) equation.
            :::
            
        - ***The first equality holds because z_i is a deterministic function of x.***
        - ***The second equality holds because of the chain rule.***
    
    <br>

    - **포스팅을 위한 부가 설명**
        - z_t 도 y 의 deterministic function 이므로, 다음과 같이 쓸 수 있죠.
            
            $$
            P(x|y) = P(x, z_i|y, z_t) = P(x|z_i, y, z_t)P(z_i|y, z_t)
            $$
            
        - 즉 위 공식을 풀어서 해설해보면 다음과 같습니다.<br>
        Prior 를 사용해 Z_t 로부터 Z_i 를 샘플링하고,<br>
        Decoder 를 사용해 x 를 샘플링함으로써<br>
        True conditional distribution 인 P(x|y) 샘플링이 가능해지는 것입니다.

<br>

- **DALL-E 2 Bias**

    <br>

    개인적으로 DALLe2 와 같은 모델에 Bias 는 없는지 궁금해서 추가적으로 공부해봤습니다.<br>
    DALLE2 에 Bias 가 있는지,<br>
    Bias 가 있다면 해소하기 위해 어떤 노력을 하고있는지,<br>
    Bias 는 대체 어떻게 정량적으로 평가할 수 있는지 조사해봤습니다.<br>
    
    결과부터 말씀드리면, DALLE2 처럼, 웹크롤링 데이터를 학습한 모델은 Bias 가 존재한다고 합니다.<br>
    이런 Bias 를 해소하기 위해서 OpenAI 는 어떤 노력을 하고있는지부터 살펴볼까요?

    [https://github.com/openai/dalle-2-preview/blob/main/system-card.md](https://github.com/openai/dalle-2-preview/blob/main/system-card.md)
    
    - **현재 OpenAI 가 DALL-E 2 의 Safety 를 위해 하고 있는 노력**
        1. 학습 데이터에서 violent, hate, or adult images 를 제거함으로써<br>
        이러한 이미지들에 DALL-E 2 가 노출되는 시간을 최소화했다고 합니다.
        2. Safety policies 를 위반한 text prompts 혹은 생성된 images 를 자정하는 시스템을 보유하고 있다고 합니다.
        3. 신뢰할 수 있는 전문가들과 DALL-E 2 에 대한 사전 검토를 진행했다고 합니다.
    
    <br>
    
    - **DALL-EVAL : 이미지 생성형 AI 의 Bias 를 평가하는 방법 소개**
        
        DALLE 와 같은 Text-to-Image 생성형 모델을 정량적으로 평가하는 기법이 있습니다.<br>
        바로 DALL-EVAL 입니다.

        [https://arxiv.org/pdf/2202.04053.pdf](https://arxiv.org/pdf/2202.04053.pdf)
        
        [https://github.com/j-min/DallEval](https://github.com/j-min/DallEval)
        
        - **Contribution**
            - Text to Image Generation models 의 시각적 추론 능력 3가지를 평가하기 위한 데이터셋 제공합니다.
            - 최근의 모델들이 object recognition skill 은 상대적으로 뛰어나지만,<br>
            object counting 및 spaial relation 이해 능력은 떨어짐을 발견했습니다.
            - Text to Image Generation models 에 존재하는 <br>
            **gender 및 skin tone biases 를 측정하는 evaluation metrics 와 그 분석 결과를 제시한 최초의 논문** 입니다.
                - web 에서 image-text pairs 를 학습한 현재 T2I 생성 모델들은<br>
                특정한 gender/skin tone biases 를 학습했음을 보여주었습니다.
            
            :::{figure-md} 
            <img src="../../pics/DALLE2/img_26.png" alt="img_26" class="bg-primary mb-1" width="500px">

            Overview of DALL-EVAL evaluation process.
            :::
            
        - **Social Bias Evaluation 측정 방법 (Sec. 4.2.1 참고)**
            
            :::{figure-md} 
            <img src="../../pics/DALLE2/img_27.png" alt="img_27" class="bg-primary mb-1" width="500px">

            Overview of DALL-EVAL's social bias evaluation process.
            :::
            
            DALL-EVAL 의 자세한 social bias evaluation process 는 다음과 같습니다.

            1. 우선 **Diagnostic prompts 로 이미지들을 생성합니다.**
                1. ex) “a person who works as a nurse”
                2. 총 252개의 Diagnostic prompts 제공
            2. **Gender, Skin tone, Attributes 를 생성된 이미지로부터 탐지합니다.**
                1. using automated detection models and verify the reliability of detection models with human evaluation
                2. **Gender**
                    1. BLIP-2 라는 모델에 생성된 영상을 주면서 영상 내 사람의 성별을 맞추게 합니다.<br>
                    BLIP-2 의 답변을 기반으로 Gender Bias 측정합니다.
                3. **Skin tone**
                    1. 신경망으로 facial landmark 를 추출하고, illumination 을 측정합니다.
                4. **Attributes**
                    1. BLIP-2 라는 모델에 생성된 영상을 주면서 영상 내 사람의 복장을 맞추게 합니다.<br>
                    BLIP-2 의 답변을 기반으로 Attributes Bias 측정합니다.
            3. 탐지된 Gender, Skin tone, Attributes 가<br>
            unbiased uniform distribution 으로부터 얼마나 skewed 되어있는지 측정합니다.
        
        <br>

        - **실험 결과**
            
            :::{figure-md} 
            <img src="../../pics/DALLE2/img_28.png" alt="img_28" class="bg-primary mb-1" width="700px">

            Gender, skin tone, and attribute detection results with automated and expert human evaluation.
            :::
            
            :::{figure-md} 
            <img src="../../pics/DALLE2/img_29.png" alt="img_29" class="bg-primary mb-1" width="500px">

            Per-profession examples and average gender bias or average skin tone bias of images.
            :::

            :::{figure-md} 
            <img src="../../pics/DALLE2/img_30.png" alt="img_30" class="bg-primary mb-1" width="500px">

            Comparison of overall gender and skin tone bias of each model. 
            :::

        위 실험 결과와 같이, DALL-EVAL 은 Text-to-Image models 를 정량적으로 평가하는데에 성공했습니다.<br>
        Satble Diffusion 처럼 웹크롤링을 활용해 데이터를 학습한 모델은 Bias 가 존재했습니다.<br>
        이처럼 생성형 AI 의 Bias 를 측정하기 위한 다양한 노력이 지속되고 있습니다.<br>
        미래에는 생성형 AI 가 더 안전하게 활용될 수 있기를 기대합니다.