export PYTHONPATH=$PWD
python main.py \
  --root_path /usr/home/kop/datasets/DAD/ \
  --mode train \
  --view top_depth \
  --model_type resnet \
  --model_depth 18 \
  --shortcut_type A \
  --pre_train_model False \
  --n_train_batch_size 10 \
  --a_train_batch_size 140 \
  --val_batch_size 70\
  --learning_rate 0.01 \
  --epochs 250 \
  --norm_value 255 \
  --cal_vec_batch_size 100 \
  --tau 0.1 \
  --manual_seed 26 \
  --memory_bank_size 200 \
  --resume_path '' \
  --resume_head_path '' \
  --val_step 1 \
  --save_step 10 \
  --train_crop 'random' \
  --n_scales 5 \
  --downsample 2 \
  --log_resume False \
  --width_mult 2.0 \
  --n_split_ratio 1.0 \
  --a_split_ratio 1.0 \








