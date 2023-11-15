#!/bin/bash

#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24g
#SBATCH -A ls_polle 
#SBATCH --job-name=adabins_mask
#SBATCH --output=log/adabins_%A_%a.out


module purge
module load gcc/8.2.0 cuda/11.1.1 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/venv/bin/activate

bash ./tools/dist_train.sh ./configs/adabins/adabins_efnetb5ap_kitti_mask_24e.py 1 \
--work-dir runs/adabins/kitti_adabins_masked_reduced_lr \
--deterministic