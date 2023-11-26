import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/PE_yolo/PE_yolo.py'
checkpoint_file = 'work_dirs/PE_yolo_2048/epoch_30.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并展示结果
img = 'data/foggy_cityscapes/leftImg8bit_foggy/val_for_paper/lindau/lindau_000014_000019_leftImg8bit_foggy_beta_0.02.png'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
# img = 'data/foggy_cityscapes/leftImg8bit_foggy/val_for_paper/frankfurt/frankfurt_000001_004327_leftImg8bit_foggy_beta_0.02_dehaze_dcp.png'
# img = 'data/foggy_cityscapes/leftImg8bit_foggy/val_for_paper/lindau/lindau_000027_000019_leftImg8bit_foggy_beta_0.02_RIDCP.png'

result = inference_detector(model, img)

# 显示结果
img = cv2.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')


visualizer.add_datasample(
    'lindau_000014_000019_leftImg8bit_foggy_beta_0.02',
    # 'frankfurt_000001_004327_leftImg8bit_foggy_beta_0.02_dehaze_dcp.png',
    img,
    data_sample=result,
    draw_gt=True,
    draw_pred=False,
    show=True,
    wait_time=2,
    out_file='/cpfs01/user/wangyudong/code/syf/lqit/show_dirs/PE_YOLO/llindau_000014_000019_leftImg8bit_foggy_beta_0.02.png'
    # out_file='/cpfs01/user/wangyudong/code/syf/lqit/show_dirs/dcp/frankfurt_000001_004327_leftImg8bit_foggy_beta_0.02_dehaze_dcp.png'
    )
