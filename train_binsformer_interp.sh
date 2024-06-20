#!/bin/bash

#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24g
#SBATCH -A ls_polle 
#SBATCH --job-name=int_binsformer_mask
#SBATCH --output=log/interp_maskbinsformer_%A_%a.out


module purge
module load gcc/8.2.0 cuda/11.1.1 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/venv/bin/activate

bash ./tools/dist_train.sh ./configs/binsformer/binsformer_swinl_22k_w7_kitti_mask.py 1 \
--resume-from /cluster/project/infk/courses/252-0579-00L/group26/nihars_tests/Monocular-Depth-Estimation-Toolbox/runs/depthformer/interp/iter_105600.pth \
--work-dir runs/binsformer/interp \
--deterministic