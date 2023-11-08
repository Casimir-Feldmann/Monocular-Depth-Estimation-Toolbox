#!/bin/bash

# bash ./tools/dist_train.sh configs/depthformer/depthformer_swinl_22k_w7_kitti_novel.py 1 --work-dir runs/depthformer__interp_filter_500_longer --deterministic

bash ./tools/dist_train.sh configs/depthformer/depthformer_swinl_22k_w7_kitti_novel.py 1 \
--work-dir runs/depthformer_disturb_5503331_weights_filtered \
--load-from runs/depthformer_paper/depthformer_swinl_w7_22k_kitti.pth \
--deterministic

# bash ./tools/dist_train.sh configs/depthformer/depthformer_swinl_22k_w7_kitti_novel.py 1 \
# --work-dir runs/depthformer_interp_filter_500_checkpoint_resume \
# --resume-from runs/depthformer_paper/depthformer_swinl_w7_22k_kitti.pth \
# --deterministic

# python tools/test.py \
# configs/depthformer/depthformer_swinl_22k_w7_kitti.py \
# runs/depthformer_paper/depthformer_swinl_w7_22k_kitti.pth \
# --show-dir visuals/depthformer_baseline

# python tools/test.py \
# configs/depthformer/depthformer_swinl_22k_w7_kitti_novel.py \
# runs/depthformer__interp_filter_500_longer/best_abs_rel_iter_75200.pth \
# --show-dir visuals/depthformer_ours

# python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_kitti.py runs/augmented_filtered/best_abs_rel_iter_35200.pth --eval x
# python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_kitti.py runs/depthformer_paper/depthformer_swinl_w7_22k_kitti.pth --eval x

# bash ./tools/dist_train.sh configs/adabins/adabins_efnetb5ap_kitti_24e.py 1 --work-dir runs/adabins_interp_filter_500 --deterministic






