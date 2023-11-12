_base_ = [
    'dpt_vit-b16_kitti.py',
]

data = dict(
    train=dict(
        split='/cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/splits/kitti_eigen_interpolated_train_files_with_gt.txt', #'kitti_eigen_train.txt' kitti_eigen_novel_train.txt
    )
)