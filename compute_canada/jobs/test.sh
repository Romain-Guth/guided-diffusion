cd ../..
python scripts/grad_guided_sample.py \
  --attention_resolutions 32,16,8 \
  --class_cond True \
  --diffusion_steps 1000 \
  --dropout 0.1 \
  --use_ddim False \
  --image_size 64 \
  --batch_size 1 \
  --data_dir /home/sabzi/scratch/data/ImageNet/dir1 \
  --learn_sigma True \
  --noise_schedule cosine \
  --num_channels 192 \
  --num_head_channels 64 \
  --num_res_blocks 3 \
  --resblock_updown True \
  --use_new_attention_order True \
  --use_fp16 True \
  --use_scale_shift_norm True \
  --classifier_scales 32,16,8 \
  --num_samples 1 \
  --classifier_path models/64x64_classifier.pt \
  --classifier_depth 4 \
  --model_path models/64x64_diffusion.pt