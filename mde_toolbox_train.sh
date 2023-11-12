#!/bin/bash

#SBATCH -c 8
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A ls_polle 
#SBATCH --job-name=mde_toolbox_training
#SBATCH --output=mde_toolbox_training_%j.out
#SBATCH --error=mde_toolbox_training_%j.err

module purge
module load gcc/8.2.0 cuda/11.1.1 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/venv/bin/activate

bash ./tools/dist_train.sh ./configs/dpt/dpt_vit-b16_kitti_interpolated.py 1 \
--work-dir runs/dpt/kitti_interpolated \
--deterministic