#!/bin/bash
#SBATCH --account=def-cpsmcgil
#SBATCH --output="/home/haolun/scratch/SNAKE/exp_out/infer_T5B_D3.out"
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M

source /home/haolun/projects/def-cpsmcgil/SNAKE/venv_SNAKE/bin/activate
module load cuda
nvidia-smi

python3 /home/haolun/scratch/SNAKE/experiment_E2.py \
  --model_choice="E2" \
  --dataset='wikidata' \
  --pretrained_model_name='t5-base' \
  --log_wandb=False \
  --use_lora=True \
  --lr=3e-4 \
  --weight_decay=1e-2 \
  --noise_std_dev=1e-3 \
  --mode='test'


deactivate
