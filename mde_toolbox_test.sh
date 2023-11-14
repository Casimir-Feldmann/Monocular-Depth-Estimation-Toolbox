#!/bin/bash

#SBATCH -c 8
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24g
#SBATCH -A ls_polle 
#SBATCH --job-name=depthformer_testing
#SBATCH --output=depthformer_testing_%j.out
#SBATCH --error=depthformer_testing_%j.err

module purge
module load gcc/8.2.0 cuda/11.1.1 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/venv/bin/activate

python tools/test.py \
configs/depthformer/depthformer_swinl_22k_w7_kitti_waymo_eval.py \
/cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/runs/depthformer/kitti_angled/iter_48000.pth \
--eval x \
--show-dir visuals/depthformer/waymo_angled_angled \