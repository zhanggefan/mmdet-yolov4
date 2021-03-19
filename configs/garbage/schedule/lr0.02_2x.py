_base_ = '../../_base_/schedules/schedule_2x.py'

optimizer_config = dict(
    grad_clip=dict(_delete_=True, max_norm=10, norm_type=2))

cudnn_benchmark = True
