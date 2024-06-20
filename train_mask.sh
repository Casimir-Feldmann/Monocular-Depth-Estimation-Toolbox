#!/bin/bash

#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24g
#SBATCH -A ls_polle 
#SBATCH --job-name=depthformer_mask
#SBATCH --output=log/depthformer_%A_%a.out


module purge
module load gcc/8.2.0 cuda/11.1.1 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/venv/bin/activate

bash ./tools/dist_train.sh ./configs/depthformer/mask_config.py 1 \
--resume-from runs/depthformer/contd/iter_3200.pth \
--work-dir runs/depthformer/contd \
--deterministic