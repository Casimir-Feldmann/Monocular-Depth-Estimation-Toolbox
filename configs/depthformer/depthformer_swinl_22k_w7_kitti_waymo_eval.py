_base_ = [
    '../_base_/models/depthformer_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth', # noqa
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7),
    neck=dict(
        type='HAHIHeteroNeck',
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=256),
        in_channels=[64, 192, 384, 768, 1536],
        out_channels=[64, 192, 384, 768, 1536],
        embedding_dim=512,
        scales=[1, 1, 1, 1, 1]),
    decode_head=dict(
        type='DenseDepthHead',
        act_cfg=dict(type='LeakyReLU', inplace=True),
        in_channels=[64, 192, 384, 768, 1536],
        up_sample_channels=[64, 192, 384, 768, 1536],
        channels=64,
        min_depth=1e-3,
        max_depth=80,
    ))

# dataset settings
dataset_type_train = 'KITTIDataset'
dataset_type_val = 'KITTIDataset'
dataset_type_test = 'WaymoDataset'
data_root = 'data/kitti'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(
#     mean=[117.55304001, 122.38308141, 122.51610871], std=[87.48778595, 89.14812147, 89.86554578], to_rgb=True)
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
                            'flip', 'flip_direction', 'img_norm_cfg',
                            'cam_intrinsic')),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type_train,
        data_root=None,
        img_dir='rgb_images',
        ann_dir='depth_mde',
        depth_scale=256,
        split='/cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/splits/kitti_eigen_angled_train_files_with_gt.txt', #'kitti_eigen_train.txt' kitti_eigen_novel_train.txt
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
        split='/cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/splits/waymo_angled_test_files_with_gt.txt',
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=80))
# schedules
# optimizer
max_lr=1e-4
optimizer = dict(
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1600 * 8,
    warmup_ratio=1.0 / 1000,
    min_lr_ratio=1e-8,
    by_epoch=False) # test add by_epoch false
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=1600 * 24)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(by_epoch=False, 
                  start=0,
                  interval=800, 
                  pre_eval=True, 
                  rule='less', 
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"), 
                  less_keys=("abs_rel", "rmse"))
# iter runtime
log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])