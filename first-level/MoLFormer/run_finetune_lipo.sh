#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50000M
#SBATCH --time=04:00:00
#SBATCH --account=def-hup-ab

# nvidia-smi

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.7 cudnn arrow/9.0.0 python/3.10 
source /home/yutonglu/chem/bin/activate

python finetune_pubchem_light.py \
        --device cuda \
        --batch_size 32 \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 9.923187674240417e-06 \
        --num_workers 8 \
        --max_epochs 201 \
        --num_feats 32 \
        --seed_path '../data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt' \
        --dataset_name lipo \
        --data_root ../data/lipo \
        --measure_name lipo \
        --dims 768 768 768 1 \
        --checkpoints_folder './checkpoints_lipo/finetune'\
