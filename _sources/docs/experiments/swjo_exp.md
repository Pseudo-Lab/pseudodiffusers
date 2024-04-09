``` {admonition} Information
- **Title:** Training DreamBooth on Naver Webtoon Face Dataset 
    
- **Author:** Sangwoo Jo

- **Last updated on Jul. 09, 2023**
```

# Training DreamBooth on Naver Webtoon Face Dataset 

## Introduction

이번 포스팅에서는 DreamBooth 를 직접 학습해보고 실험한 결과들을 공유할려고 합니다. 

우선적으로 학습데이터는 [https://github.com/bryandlee/naver-webtoon-data](https://github.com/bryandlee/naver-webtoon-data) 에 공개된 YOLOv5 모델 및 Waifu2x 후처리 기법을 활용하여 프리드로우에 등장하는 인물 사진들을 수집했습니다. 논문에서는 3-5 장으로 fine-tuning 이 가능하다고 제시되어있지만, 인물 사진 같은 경우 더 많은 데이터로 학습하면 성능이 더 좋아져서 15-20 장의 이미지로 학습하였습니다. 학습한 이미지들 예시입니다. 

:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_01.png" alt="swjo_exp_01" class="bg-primary mb-1" width="700px">

Training Data
:::

DreamBooth 를 실험하면서 대표적으로 instance prompt, guidance scale, negative prompt, 그리고 마지막으로 prior preservation loss 를 반영하는 정도를 조절하는 prior_loss_weight 를 바꿔가면서 학습해보았습니다. 사전학습된 text-to-image 모델로 처음에는 *hakurei/waifu-diffusion* 모델을 시도해봤지만 결과가 만족스럽지 못해 *runwayml/stable-diffusion-v1-5* 모델로 fine-tuning 작업을 진행했습니다. 

## Ablation Studies

### Prior Preservation Loss

Prior Preservation Loss 를 제외한 동일한 configuration 으로 모델 학습한 결과입니다. 

```
# with prior-preservation loss
MODEL_NAME = “runwayml/stable-diffusion-v1-5”
instance_prompt = "A photo of sks girl"
class_prompt = "A photo of a girl"

python3 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --sample_batch_size=4 \
  --max_train_steps=800 \
  --save_interval=100 \
  --save_sample_prompt="A photo of sks girl" \
  --concepts_list="concepts_list.json"
```

```
# w/o prior-preservation loss
MODEL_NAME = “runwayml/stable-diffusion-v1-5”
instance_prompt = "A photo of sks girl"
class_prompt = "A photo of a girl"

python3 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=0.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --sample_batch_size=4 \
  --max_train_steps=800 \
  --save_interval=100 \
  --save_sample_prompt="A photo of sks girl" \
  --concepts_list="concepts_list.json"
```

아래 그림처럼 동일한 inference prompt 를 입력했을 때, prior preservation loss 를 제외함으로써 input images 에 더 가까운 웹툰 사진들을 생성할 수 있었습니다. 또한, 핑크색 머리를 한 이민지 캐릭터를 어느 정도 잘 생성하는 부분도 확인할 수 있습니다.

- **Inference Prompt: "A photo of *sks* girl with pink hair” (with prior-preservation loss)**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_02.png" alt="swjo_exp_02" class="bg-primary mb-1" width="700px">

With Prior Preservation Loss
:::
    
- **Inference Prompt: " A photo of *sks* girl with pink hair” (w/o prior-preservation loss)**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_03.png" alt="swjo_exp_03" class="bg-primary mb-1" width="700px">

Without Prior Preservation Loss
:::

### Negative Prompt

Negative Prompt 에 대한 Ablation Study 도 진행했습니다. 캐릭터의 부자연스러운 부분이나 저해상도 이미지들을 생성하는 경우들이 종종 발생했는데, negative prompt 를 통해 더 좋은 퀄리티의 웹툰 캐릭터를 생성할 수 있었습니다.  

- **Inference Prompt: " A photo of *sks* girl with pink hair” (w/o negative prompt)**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_03.png" alt="swjo_exp_03" class="bg-primary mb-1" width="700px">

Without Negative Prompt
:::

- **Inference Prompt: " A photo of *sks* girl with pink hair”**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_04.png" alt="swjo_exp_04" class="bg-primary mb-1" width="700px">

With Negative Prompt
:::
    
### Instance Prompt / Guidance Scale

DreamBooth 논문에서 제시한 instance prompt 외에 “A photo of a girl in the style of *sks*” 라는 prompt 로 학습을 시도해보기도 했습니다. *sks* 라는 unique identifier 에 특정 여자 캐릭터에 대한 정보뿐만 아니라 프리드로우 그림체 자체를 담아내기 위한 목적이였습니다. 

```
# different instance prompt with prior-preservation loss
MODEL_NAME = “runwayml/stable-diffusion-v1-5”
instance_prompt = "A photo of a girl in the style of sks"
class_prompt = "A photo of a girl"

python3 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --sample_batch_size=4 \
  --max_train_steps=800 \
  --save_interval=100 \
  --save_sample_prompt="A photo of sks girl" \
  --concepts_list="concepts_list.json"
```

```
# different instance prompt w/o prior-preservation loss
MODEL_NAME = “runwayml/stable-diffusion-v1-5”
instance_prompt = "A photo of a girl in the style of sks"
class_prompt = "A photo of a girl"

python3 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=0.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --sample_batch_size=4 \
  --max_train_steps=800 \
  --save_interval=100 \
  --save_sample_prompt="A photo of sks girl" \
  --concepts_list="concepts_list.json"
```

Inference 시, 프리드로우의 그림체가 반영된 남자가 생성되도록 prompt 를 “A photo of a boy in the style of *sks*” 로 입력했을때의 결과입니다. DreamBooth 혹은 사전학습된 text-to-image 모델을 프리드로우 작가님의 웹툰 장면들로 전체적으로 학습하게 된다면 더 다양한 inference 결과들을 볼 수 있을 것 같습니다. 

- **Inference Prompt: “A photo of a boy in the style of *sks*” (num_inference_steps = 24 / with prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_05.png" alt="swjo_exp_05" class="bg-primary mb-1" width="700px">

Instance Prompt
:::
    
Inference step 을 늘려가면서 추론된 인물 이미지의 퀄리티가 상승하는 부분도 확인할 수 있었습니다. 또한, guidance scale 에 대한 실험도 진행했는데 guidance scale 이 작을수록 prompt 와 무관한 random 한 이미지들을 생성하게 됩니다. 최종적으로 num_inference steps 와 guidance scale 의 값은 각각 100 과 7.5 로 설정하였습니다. 

- **Inference Prompt: “A photo of a boy in the style of *sks*” (num_inference_steps=100 / with prior-preservation loss)**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_06.png" alt="swjo_exp_06" class="bg-primary mb-1" width="700px">

Increasing Number of Inference Steps 
:::
    
- **Inference Prompt: “A photo of a boy in the style of *sks*” (num_inference_steps = 100 / with prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
:::{figure-md}
<img src="../../pics/swjo_exp/swjo_exp_07.png" alt="swjo_exp_07" class="bg-primary mb-1" width="700px">

Increasing Number of Inference Steps / Negative Prompt
:::

- **Inference Prompt: “A photo of a boy in the style of *sks*” (num_inference_steps = 100 / with prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
    **+ guidance_scale = 4**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_08.png" alt="swjo_exp_08" class="bg-primary mb-1" width="700px">

Guidance Scale
:::
    
동일한 inference prompt 로 prior-preservation loss 를 제외해본 결과, 생성된 남자의 머리카락이 더 길어지고 더 여성스러운 생김새를 가지는 놀라운 사실도 발견했습니다. 

- **Inference Prompt: “A photo of a boy in the style of *sks*” (num_inference_steps = 100 / w/o prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_09.png" alt="swjo_exp_09" class="bg-primary mb-1" width="700px">

Without Prior Preservation Loss
:::
    
## Appendix

그 외 다양한 inference prompt 에 따른 재미있는 실험결과들을 공유합니다. 아직 손의 모양을 text-to-image 모델이 생성하지 못하는 부분도 재차 확인할 수 있었습니다. 

- **Inference Prompt: “A photo of a boy climbing up the mountain in the style of *sks*” (num_inference_steps = 100 / w/o prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_10.png" alt="swjo_exp_10" class="bg-primary mb-1" width="700px">

Appendix 1
:::
    
- **Inference Prompt: “A painting of a boy in the style of *sks*” (num_inference_steps = 100 / w/o prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_11.png" alt="swjo_exp_11" class="bg-primary mb-1" width="700px">

Appendix 2
:::
    
- **Inference Prompt: “A hand drawing of a boy in the style of *sks*” (num_inference_steps = 100 / w/o prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
:::{figure-md} 
<img src="../../pics/swjo_exp/swjo_exp_12.png" alt="swjo_exp_12" class="bg-primary mb-1" width="700px">

Appendix 3
:::

마지막으로 하단의 좌측과 우측 사진은 각각 “A photo of *sks* girl” 그리고 “A photo of a girl in the style of *sks*” 이라는 prompt 로 DreamBooth 모델을 각각 학습한 후, 나비를 생성하라는 동일한 prompt 로 추론해본 결과입니다. *sks* 가 수식하는 명사가 girl 이 아닌 style 이도록 prompt 를 수정함으로써, butterfly 사진을 생성할때 조금이나마 더 프리드로우 웹툰의 그림체를 반영할 수 있었던 부분도 확인할 수 있었습니다. 

- **Inference Prompt: “A photo of a butterfly in the style of *sks*” (num_inference_steps = 100 / with prior-preservation loss)**
    
:::{figure-md}
<img src="../../pics/swjo_exp/swjo_exp_13.png" alt="swjo_exp_13" class="bg-primary mb-1" width="700px">

Appendix 4
:::
