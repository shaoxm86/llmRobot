import os
import sys
import torch
from ultralytics import YOLOWorld, SAM

# 动态设置项目路径，将所需子模块目录添加到 Python 的模块搜索路径中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 当前脚本所在文件夹的上级目录的绝对路径
# os.path.join()：拼接路径，.append()：将新路径添加到最后。作用：让 Python 能够导入这些目录中的自定义 .py 文件
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from graspnet import GraspNet, pred_decode

def model_init():

    # YOLO-World: 开放词汇目标检测
    yolo_world_model = '/home/sxm/catkin_ws/src/yoloworld-sam-graspnet/src/yolov8s-worldv2.pt'
    yolo = YOLOWorld(yolo_world_model).to('cuda:0')
    # SAM 2: 视频分割追踪（tiny 版，速度优先）
    sam2_model = '/home/sxm/sam2/sam2.1_t.pt'
    sam = SAM(sam2_model).to('cuda:0')

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
    net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # 加载预训练权重
    checkpoint = torch.load(os.path.join(ROOT_DIR, './logs/log_rs/checkpoint-rs.tar'))
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()  # 推理模式

    return yolo, sam, net