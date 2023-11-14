_base_ = [
    '../_base_/models/dpt.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    pretrained='pretrain/jx_vit_base_p16_224-80ecf9dd.pth',
    decode_head=dict(
        min_depth=1e-3,
        max_depth=80,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0, warm_up=True),
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
max_lr=1e-4 * 0.25
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
        }))

lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False,
)
momentum_config = dict(
    policy='OneCycle'
)
runner = dict(type='EpochBasedRunner', max_epochs=24 * 4)

evaluation = dict(interval=1,
                  pre_eval=True, 
                  rule='less', 
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"), 
                  less_keys=("abs_rel", "rmse"))

# dataset settings
dataset_type_train = 'KITTIDataset'
dataset_type_val = 'KITTIDataset'
dataset_type_test = 'WaymoDataset'
data_root = 'data/kitti'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size= (352, 704)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='LoadKITTICamIntrinsic'),
    dict(type='KBCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(352, 704)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.9, 1.1], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg')),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKITTICamIntrinsic'),
    dict(type='KBCrop', depth=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1216, 352),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', 
                 keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 
                            'flip', 'flip_direction', 'img_norm_cfg')),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ScaleWaymoToKITTI'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(564, 376),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', 
                 keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 
                            'flip', 'flip_direction', 'img_norm_cfg')),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type_train,
        data_root=None,
        img_dir='rgb_images',
        ann_dir='depth_mde',
        depth_scale=256,
        split='/cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/splits/kitti_eigen_train.txt', #'kitti_eigen_train.txt' kitti_eigen_novel_train.txt
        pipeline=train_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=80),
    val=dict(
        type=dataset_type_val,
        data_root=None,
        img_dir='rgb_images',
        ann_dir='depth_mde',
        depth_scale=256,
        split='/cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/splits/kitti_eigen_test.txt',
        pipeline=val_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=80),
    test=dict(
        type=dataset_type_test,
        data_root=None,
        img_dir='rgb_images',
        ann_dir='depth_mde',
        depth_scale=255,
        split='/cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/splits/waymo_angled_test_files_with_gt_short.txt',
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=80))

