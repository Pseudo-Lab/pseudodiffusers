```{admonition} Information
- **Title:** High-Resolution Image Synthesis with Latent Diffusion Models (CVPR 2022)

- **Reference**
    - Paper: [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
    - Code: [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)

- **Author:** Namkyeong Cho

- **Last updated on May. 31, 2023**
```

# Latent Diffusion Model

오늘 알아볼 모델은 Latent Diffusion Model입니다.
기존에 다뤘던 Diffusion Model과 유사하게 동작하는 생성 모델입니다. 이 논문에서는 컴퓨터 자원의 소모를 줄이면서 Diffusion Model과 유사한 성능을 얻는것이 그 목표입니다.

Latent Diffusion Model은 전반적으로 아래와 같은 구조를 가집니다.

:::{figure-md} 
<img src="../../pics/Latent_Diffusion_Model/Unet.png"  class="bg-primary mb-1" width="700px">

Structure of Latent Diffusion Model
:::
$x \in \mathbb{R}^{H\times W \times 3}$이 input으로 주어졌을때 이를 encoder $\mathcal{E}$를 통해서 $z=\mathcal{E}(x) \in \mathbb{R}^{h\times w\times c }$로 인코딩 하고 $\hat{x}=\mathcal{D}(z)$
로 디코딩을 한다. 이 논문에서 $f=H/h=W/w=2^m$, $m\in \mathbb{N}$이 되도록 여러 $m$에 대해서 테스트를 진행하였다. 또한 Latent space에서 분산이 커지지 않도록 KL divergence와 vector quantization(VQ)을 활용하였다.
이미지외 텍스트나, sematic map과 같이 추가적인 정보는 $\tau_\theta$를 통해서 전달을 하였고, 

$$  Q=W^{(i)}_Q \phi_i(z_i), K=W^{(i)}_K \phi_i(z_i), V=W^{(i)}_V \phi_i(z_i) $$

로 정의되고 $\phi_i(z_i)$는 $U$-Net 중간의 representation, $W^{i}_V, W^{i}_K, W^{i}_Q$는 학습 가능한 projection matrix이다. 
$Q, K, V$ 는 attention의 query, key, value에 해당하며 

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d}})\cdot V
$$

로 연산이 진행된다. 학습을 위한 loss 함수는 다음과 같이표현된다.

$$
\mathcal{L}_{LDM} = \mathbb{E}_{\mathcal{E}(x), 
\epsilon \sim \mathcal{N}(0,1),t} \left[ \|\epsilon-\epsilon_{\theta}(z_t,t) \|_{2}^{2}\right].
$$

여기서 주목할만한 부분은 기존 Diffusion Model에서 

$$
\mathcal{L}_{DM} = \mathbb{E}_{x, 
\epsilon \sim \mathcal{N}(0,1),t} \left[ \|\epsilon-\epsilon_{\theta}(x_t,t) \|_{2}^{2}\right].
$$

와 같은 loss function으로 학습을 진행시키는데 $x_t$를 $z_t$로 바꾸면서 연산의 양을 줄였다는 점이다.


# Experiments

해당 논문에서는 다양한 task에 대해서 실험을 진행하였는데, 그중 일부만 소개하도록 하겠다.
아래의 그림은 다양한 dataset에서 뽑은 샘플과 text to image sample들입니다.

:::{figure-md} 
<img src="../../pics/Latent_Diffusion_Model/experiment1.png" class="bg-primary mb-1" width="700px">

Sample images
:::


:::{figure-md} 
<img src="../../pics/Latent_Diffusion_Model/text_to_image.png"  class="bg-primary mb-1" width="700px">

text to image on LAION
:::

실험을 통해서 나온 결과 $m=2,3,4$ 혹은 $f=4, 8, 16$인 경우 적절한 FID 점수와 효율성을 보여주었습니다.

:::{figure-md} 
<img src="../../pics/Latent_Diffusion_Model/trade_off.png" class="bg-primary mb-1" width="700px">

text to image on LAION
:::

Layout이 주어졌을 때, 이를 기반으로 image를 생성하는 layout-to-image의 샘플 결과입니다. 
:::{figure-md} 
<img src="../../pics/Latent_Diffusion_Model/layout_to_image.png" class="bg-primary mb-1" width="700px">

layout-to-image
:::



