import os
import sys
import torch
from ultralytics import YOLOWorld, SAM

# 动态设置项目路径，将所需子模块目录添加到 Python 的模块搜索路径中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 当前脚本所在文件夹的上级目录的绝对路径
# os.path.join()：拼接路径，.append()：将新路径添加到最后。作用：让 Python 能够导入这些目录中的自定义 .py 文件
GRASPNET_ROOT = os.environ.get('GRASPNET_BASELINE_ROOT', os.path.join(ROOT_DIR, 'graspnet-baseline'))
for subdir in ('models', 'dataset', 'utils'):
    sys.path.append(os.path.join(GRASPNET_ROOT, subdir))

from graspnet import GraspNet

def model_init(yolo_path=None, sam_path=None, checkpoint_path=None, device=None):

    # YOLO-World: 开放词汇目标检测
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo_world_model = yolo_path or os.environ.get('YOLO_WORLD_WEIGHTS')
    sam2_model = sam_path or os.environ.get('SAM_WEIGHTS')
    checkpoint_path = checkpoint_path or os.environ.get('GRASPNET_CHECKPOINT')
    for path in (yolo_world_model, sam2_model, checkpoint_path):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError('Configure YOLO/SAM/GraspNet weight paths')
    yolo = YOLOWorld(yolo_world_model).to(device)
    # SAM 2: 每帧使用检测框提示进行分割
    sam = SAM(sam2_model).to(device)

    # ----------------------------------------------------
    # GraspNet: 6-DOF 抓取预测网络
    #   input_feature_dim=0:  仅用位置坐标（不需额外特征）
    #   num_view=300:         从 300 个视角检测抓取
    #   num_angle=12:         每个视角 12 个角度
    #   num_depth=4:          4 种抓取深度
    #   cylinder_radius=0.05: 夹爪半径 5cm
    #   hmax_list:            夹爪 4 种开合宽度 [1,2,3,4] cm
    net = GraspNet(input_feature_dim=0, 
                   num_view=300, 
                   num_angle=12, 
                   num_depth=4,
                   cylinder_radius=0.08, 
                   hmin=-0.01,
                   hmax_list=[0.02, 0.04, 0.06, 0.08], 
                   is_training=False)
    net.to(torch.device(device))

    # 加载预训练权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()  # 推理模式

    return yolo, sam, net