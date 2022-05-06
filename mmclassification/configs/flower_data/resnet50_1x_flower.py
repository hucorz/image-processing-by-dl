_base_ = [
    '../_base_/models/resnet50.py', 
    '../_base_/schedules/imagenet_bs256.py', 
    '../_base_/default_runtime.py'
]

checkpoint_path = "checkpoint/resnet50_8xb32_in1k_20210831-ea4938fc.pth"

# --- 修改模型配置 ---
model = dict(
    backbone = dict(
        frozen_stages=3,
        init_cfg = dict(
            type="Pretrained",
            checkpoint=checkpoint_path,
            prefix="backbone"
        )
    ),
    head=dict(
        num_classes=200,
        topk=(1, )
    )
)

# --- 修改数据集配置 ---
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/flower_data/train',
        ann_file="data/flower_data/train_annotations.txt",
        classes="data/flower_data/classes.txt",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/flower_data/val',
        ann_file="data/flower_data/val_annotations.txt",
        classes="data/flower_data/classes.txt",
        pipeline=test_pipeline))
evaluation = dict(metric='accuracy', metric_options={'topk': (1, )})

# --- 优化器设置 ---
optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)  # 不设置梯度裁剪

lr_config = dict(policy='step', step=3, gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=5)

# ---- 运行设置 ----
# 每 10 个训练批次输出一次日志
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ],
)

load_from = checkpoint_path