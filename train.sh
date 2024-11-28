MODEL_NAME="CompVis/stable-diffusion-v1-4"
Out_dir="models/exp"
modifier_token="<s1*>+<c1*>+<s2*>+<c2*>+<s3*>+<c3*>"

CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 python src/train/train_colorpeel.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$Out_dir \
  --concepts_list=train_instances.json \
  --resolution=512  \
  --train_batch_size=1  \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=1125 \
  --cos_weight=0.2 \
  --scale_lr --hflip  \
  --modifier_token $modifier_token \
  --initializer_token "cat+color+crow+color+bird+color"

python src/test/test.py --exp exp --prompt "a photo of <c1*> <s3*>" --modifier_token $modifier_token --samples 10
