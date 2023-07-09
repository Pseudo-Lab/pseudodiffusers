# 제목 없음

**Introduction** 

이번 포스팅에서는 DreamBooth 를 직접 학습해보고 실험한 결과들을 공유할려고 합니다. 

우선적으로 학습데이터는 [https://github.com/bryandlee/naver-webtoon-data](https://github.com/bryandlee/naver-webtoon-data) 에 공개된 YOLOv5 모델 및 Waifu2x 후처리 기법을 활용하여 프리드로우에 등장하는 인물 사진들을 수집했습니다. 논문에서는 3-5 장으로 fine-tuning 이 가능하다고 제시되어있지만, 인물 사진 같은 경우 더 많은 데이터로 학습하면 성능이 더 좋아져서 15-20 장의 이미지로 학습하였습니다. 학습한 이미지들 예시입니다. 

![0184.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0184.png)

![0174.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0174.png)

![0170.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0170.png)

![0159.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0159.png)

DreamBooth 를 실험하면서 대표적으로 instance prompt, guidance scale, negative prompt, 그리고 마지막으로 prior preservation loss 를 반영하는 정도를 조절하는 prior_loss_weight 를 바꿔가면서 학습해보았습니다. 사전학습된 text-to-image 모델로 처음에는 *hakurei/waifu-diffusion* 모델을 시도해봤지만 결과가 만족스럽지 못해 *runwayml/stable-diffusion-v1-5* 모델로 fine-tuning 작업을 진행했습니다. 

**Ablation Studies** 

**Prior Preservation Loss** 

Prior Preservation Loss 를 제외한 동일한 configuration 으로 모델 학습한 결과입니다. 

```python
# with prior-preservation loss
MODEL_NAME = “runwayml/stable-diffusion-v1-5”
instance_prompt = "A photo of sks girl"
class_prompt = "A photo of a girl"

!python3 train_dreambooth.py \
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

```python
# w/o prior-preservation loss
MODEL_NAME = “runwayml/stable-diffusion-v1-5”
instance_prompt = "A photo of sks girl"
class_prompt = "A photo of a girl"

!python3 train_dreambooth.py \
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
    
    
    ![0.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0.png)
    
    ![3.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3.png)
    
    ![1.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1.png)
    
    ![2.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2.png)
    
- **Inference Prompt: " A photo of *sks* girl with pink hair” (w/o prior-preservation loss)**
    
    
    ![0 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0_(1).png)
    
    ![3 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3_(1).png)
    
    ![1 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1_(1).png)
    
    ![2 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2_(1).png)
    

**Negative Prompt** 

Negative Prompt 에 대한 Ablation Study 도 진행했습니다. 캐릭터의 부자연스러운 부분이나 저해상도 이미지들을 생성하는 경우들이 종종 발생했는데, negative prompt 를 통해 더 좋은 퀄리티의 웹툰 캐릭터를 생성할 수 있었습니다.  

- **Inference Prompt: " A photo of *sks* girl with pink hair” (w/o negative prompt)**
    
    
    ![0 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0_(1).png)
    
    ![3 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3_(1).png)
    
    ![1 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1_(1).png)
    
    ![2 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2_(1).png)
    
- **Inference Prompt: " A photo of *sks* girl with pink hair”**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
    ![0 (2).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0_(2).png)
    
    ![1 (2).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1_(2).png)
    
    ![2 (2).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2_(2).png)
    
    ![3 (2).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3_(2).png)
    

**Instance Prompt / Guidance Scale**

DreamBooth 논문에서 제시한 instance prompt 외에 “A photo of a girl in the style of *sks*” 라는 prompt 로 학습을 시도해보기도 했습니다. *sks* 라는 unique identifier 에 특정 여자 캐릭터에 대한 정보뿐만 아니라 프리드로우 그림체 자체를 담아내기 위한 목적이였습니다. 

```python
# different instance prompt with prior-preservation loss
****MODEL_NAME = “runwayml/stable-diffusion-v1-5”
instance_prompt = "A photo of a girl in the style of sks"
class_prompt = "A photo of a girl"

!python3 train_dreambooth.py \
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

```python
# different instance prompt w/o ****prior-preservation loss
****MODEL_NAME = “runwayml/stable-diffusion-v1-5”
instance_prompt = "A photo of a girl in the style of sks"
class_prompt = "A photo of a girl"

!python3 train_dreambooth.py \
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
    
    ![3 (3).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3_(3).png)
    
    ![2 (3).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2_(3).png)
    
    ![1 (3).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1_(3).png)
    
    ![0 (3).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0_(3).png)
    

Inference step 을 늘려가면서 추론된 인물 이미지의 퀄리티가 상승하는 부분도 확인할 수 있었습니다. 또한, guidance scale 에 대한 실험도 진행했는데 guidance scale 이 작을수록 prompt 와 무관한 random 한 이미지들을 생성하게 됩니다. 최종적으로 num_inference steps 와 guidance scale 의 값은 각각 100 과 7.5 로 설정하였습니다. 

- **Inference Prompt: “A photo of a boy in the style of *sks*” (num_inference_steps=100 / with prior-preservation loss)**
    
    
    ![0 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0_(1)%201.png)
    
    ![3 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3_(1)%201.png)
    
    ![2 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2_(1)%201.png)
    
    ![1 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1_(1)%201.png)
    
- **Inference Prompt: “A photo of a boy in the style of *sks*” (num_inference_steps = 100 / with prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
    ![1.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1%201.png)
    
    ![0.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0%201.png)
    
    ![3.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3%201.png)
    
    ![2.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2%201.png)
    

- **Inference Prompt: “A photo of a boy in the style of *sks*” (num_inference_steps = 100 / with prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
    **+ guidance_scale = 4**
    
    ![0.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0%202.png)
    
    ![2.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2%202.png)
    
    ![3.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3%202.png)
    
    ![1.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1%202.png)
    

동일한 inference prompt 로 prior-preservation loss 를 제외해본 결과, 생성된 남자의 머리카락이 더 길어지고 더 여성스러운 생김새를 가지는 놀라운 사실도 발견했습니다. 

- **Inference Prompt: “A photo of a boy in the style of *sks*” (num_inference_steps = 100 / w/o prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
    ![3.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3%203.png)
    
    ![2.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2%203.png)
    
    ![1.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1%203.png)
    
    ![0.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0%203.png)
    

**Appendix**

그 외 다양한 inference prompt 에 따른 재미있는 실험결과들을 공유합니다. 아직 손의 모양을 text-to-image 모델이 생성하지 못하는 부분도 재차 확인할 수 있었습니다. 

- **Inference Prompt: “A photo of a boy climbing up the mountain in the style of *sks*” (num_inference_steps = 100 / w/o prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
    ![3 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3_(1)%202.png)
    
    ![2 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2_(1)%202.png)
    
    ![1 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1_(1)%202.png)
    
    ![0 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0_(1)%202.png)
    

- **Inference Prompt: “A painting of a boy in the style of *sks*” (num_inference_steps = 100 / w/o prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
    ![3 (2).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3_(2)%201.png)
    
    ![2 (2).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2_(2)%201.png)
    
    ![1 (2).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1_(2)%201.png)
    
    ![0 (2).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0_(2)%201.png)
    
- **Inference Prompt: “A hand drawing of a boy in the style of *sks*” (num_inference_steps = 100 / w/o prior-preservation loss)**
    
    **+** **Negative Prompt: “ugly, disfigured, deformed, low resolution”**
    
    ![2 (4).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2_(4).png)
    
    ![1 (3).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/1_(3)%201.png)
    
    ![3 (3).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3_(3)%201.png)
    
    ![2 (3).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/2_(3)%201.png)
    
    ![0 (3).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/0_(3)%201.png)
    

마지막으로 하단의 좌측과 우측 사진은 각각 “A photo of *sks* girl” 그리고 “A photo of a girl in the style of *sks*” 이라는 prompt 로 DreamBooth 모델을 각각 학습한 후, 나비를 생성하라는 동일한 prompt 로 추론해본 결과입니다. *sks* 가 수식하는 명사가 girl 이 아닌 style 이도록 prompt 를 수정함으로써, butterfly 사진을 생성할때 조금이나마 더 프리드로우 웹툰의 그림체를 반영할 수 있었던 부분도 확인할 수 있었습니다. 

- **Inference Prompt: “A photo of a butterfly in the style of *sks*” (num_inference_steps = 100 / with prior-preservation loss)**
    
    
    ![3 (1).png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3_(1)%203.png)
    
    ![3.png](%E1%84%8C%E1%85%A6%E1%84%86%E1%85%A9%E1%86%A8%20%E1%84%8B%E1%85%A5%E1%86%B9%E1%84%8B%E1%85%B3%E1%86%B7%20601372465c4d4ea38633e0ba940b6e05/3%204.png)