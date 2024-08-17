#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50000M
#SBATCH --time=04:00:00
#SBATCH --account=def-hup-ab

nvidia-smi

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.7 cudnn arrow/9.0.0 python/3.10 
source /home/yutonglu/chem/bin/activate

python finetune_pubchem_light_classification.py \
        --device cuda \
        --batch_size 32 \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.5 \
        --dropout 0.5 \
        --lr_start 4.732262604993007e-05 \
        --num_workers 8 \
        --max_epochs 50 \
        --num_feats 32 \
        --seed_path '../data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt' \
        --dataset_name bace \
        --data_root ../data/bace \
        --measure_name Class \
        --dims 768 768 768 1 \
        --checkpoints_folder './checkpoints_bace'\
        --num_classes 2 \
