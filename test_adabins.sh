#!/bin/bash

#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24g
#SBATCH -A ls_polle 
#SBATCH --job-name=test-adabins_mask
#SBATCH --output=log/test-adabins_%A_%a.out


module purge
module load gcc/8.2.0 cuda/11.1.1 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/venv/bin/activate

python tools/test.py \
    /cluster/project/infk/courses/252-0579-00L/group26/nihars_tests/Monocular-Depth-Estimation-Toolbox/configs/adabins/adabins_efnetb5ap_kitti_mask_24e.py \
    /cluster/project/infk/courses/252-0579-00L/group26/nihars_tests/Monocular-Depth-Estimation-Toolbox/runs/adabins/kitti_adabins_masked_reduced_lr/epoch_35.pth \
    --eval x 