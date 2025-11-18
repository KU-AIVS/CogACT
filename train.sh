#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
 --pretrained_checkpoint  /dev/hdd/es/CogACT/pretrain/CogACT-Base/checkpoints/CogACT-Base.pt\
 --vla.type prism-dinosiglip-224px+oxe+diffusion\
 --vla.data_mix custom_finetuning\
 --vla.expected_world_size 1\
 --vla.global_batch_size 32\
 --vla.per_device_batch_size 32\
 --vla.learning_rate 2e-5\
 --data_root_dir /dev/hdd/es/dataset\
 --run_root_dir /dev/hdd/es/CogACT/result\
 --run_id Cogact-base-finetune_26\
 --image_aug False\
 --wandb_project CogACT_26\
 --wandb_entity es4402\
 --save_interval 10000\
 --repeated_diffusion_steps 8\
 --future_action_window_size 15\
 --action_model_type DiT-B\
 --is_resume False\
 --vla.epochs 60


#resume
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
# --pretrained_checkpoint /dev/hdd/es/CogACT/result/Cogact-base-finetune_26/checkpoints/step-070000-epoch-41-loss=0.0144.pt\
# --vla.type prism-dinosiglip-224px+oxe+diffusion\
# --resume_step 70000\
# --resume_epoch 41\
# --is_resume True\
# --vla.data_mix custom_finetuning\
# --vla.expected_world_size 1\
# --vla.global_batch_size 32\
# --vla.per_device_batch_size 32\
# --vla.learning_rate 2e-5\
# --data_root_dir /dev/hdd/es/dataset\
# --run_root_dir /dev/hdd/es/CogACT/result\
# --run_id Cogact-base-finetune_26\
# --image_aug False\
# --wandb_project CogACT_26\
# --wandb_entity es4402\
# --save_interval 10000\
# --repeated_diffusion_steps 8\
# --future_action_window_size 15\
# --action_model_type DiT-B\
# --vla.epochs 60
