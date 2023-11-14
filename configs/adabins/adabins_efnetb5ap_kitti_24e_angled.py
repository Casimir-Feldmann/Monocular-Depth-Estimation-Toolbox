_base_ = [
    'adabins_efnetb5ap_kitti_24e.py',
]

data = dict(
    train=dict(
        split='/cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/splits/kitti_eigen_angled_train_files_with_gt.txt', #'kitti_eigen_train.txt' kitti_eigen_novel_train.txt
    )
)