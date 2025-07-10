
# /R-ACP/Camera_calibration/main-multi-gpu.py
import cv2
import torch
import torch.nn as nn
import torchreid
from torchvision import transforms
import numpy as np
import os
import xml.etree.ElementTree as ET
from scipy.stats import entropy

# 设置路径参数
base_path = './Wildtrack/Image_subsets'
output_image_base_dir = "./24-JSAC/Re_ID_Test/match_image"
output_xml_dir = "./24-JSAC/Re_ID_Test/match_log"
frame_start = 0
frame_end = 2000
frame_step = 5

# 初始化行人检测器（例如，使用YOLOv5）
class PersonDetector:
    def __init__(self, device):
        self.device = device
        with torch.cuda.device(self.device):
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.to(self.device)
            self.model.eval()

    def detect(self, image):
        with torch.cuda.device(self.device):
            results = self.model(image)
        return results.xyxy[0].cpu().numpy()  # 返回检测结果

# 初始化特征提取器
class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        with torch.cuda.device(self.device):
            self.extractor = torchreid.utils.FeatureExtractor(
                model_name='osnet_x1_0',
                model_path='osnet_x1_0_imagenet.pth',
                device=f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu'
            )

    def extract(self, image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)
        with torch.cuda.device(self.device):
            features = self.extractor(image)
        return features[0].cpu().detach().numpy()

    def compute_communication_cost(self, features_list):
        # 将所有特征拼接在一起
        all_features = np.hstack(features_list)  # 拼接所有特征为一个大的特征向量

        # 对拼接后的特征进行量化，并计算熵
        hist, _ = np.histogram(all_features, bins=1024, range=(-10, 10), density=True)  # 增加分箱数量，扩展范围
        comm_cost = entropy(hist, base=2)  # 计算拼接后特征的熵值
        return comm_cost



def get_image_path(camera_id, frame_id):
    frame_str = f'{frame_id:08d}.png'
    return os.path.join(base_path, f'C{camera_id}', frame_str)

# 获取脚的中心位置
def get_foot_center(box):
    x_center = (box[0] + box[2]) / 2
    y_bottom = box[3]
    return (int(x_center), int(y_bottom))

# 生成XML日志文件
def create_xml_log(camera_id, frame_id, foot_centers, output_dir, match_camera_id):
    folder_path = os.path.join(output_dir, f'camera{camera_id}_match_camera{match_camera_id}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    root = ET.Element("MatchLog")
    root.set("CameraID", str(camera_id))
    root.set("FrameID", str(frame_id))
    
    for person_id, (x, y) in foot_centers:
        person_elem = ET.SubElement(root, "Person")
        person_elem.set("ID", str(person_id))
        point_elem = ET.SubElement(person_elem, "FootCenter")
        point_elem.set("x", str(x))
        point_elem.set("y", str(y))
    
    tree = ET.ElementTree(root)
    output_path = os.path.join(folder_path, f"match_log_frame{frame_id}.xml")
    tree.write(output_path)
    print(f"XML log saved to {output_path}")

# 主函数
def main(reference_camera_id, unknown_camera_id, N_threshold=5, gpu_id=0):
    torch.cuda.set_device(gpu_id)  # 设置当前GPU
    detector = PersonDetector(gpu_id)
    extractor = FeatureExtractor(gpu_id)

    total_communication_cost = 0  # 累积通信成本

    for frame_id in range(frame_start, frame_end + 1, frame_step):
        # 加载图像
        image1_path = get_image_path(reference_camera_id, frame_id)
        image2_path = get_image_path(unknown_camera_id, frame_id)

        if not os.path.exists(image1_path) or not os.path.exists(image2_path):
            continue

        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        # 检测行人
        boxes1 = detector.detect(image1)
        boxes2 = detector.detect(image2)

        # 提取 reference_camera 的特征
        features1 = [extractor.extract(image1[int(box[1]):int(box[3]), int(box[0]):int(box[2])]) for box in boxes1]

        if features1:
            # 计算 reference_camera 发送到 unknown_camera 的通信成本
            comm_cost = extractor.compute_communication_cost(features1)
            total_communication_cost += comm_cost

            # 匹配行人
            features2 = [extractor.extract(image2[int(box[1]):int(box[3]), int(box[0]):int(box[2])]) for box in boxes2]


            # 计算相似度
            def compute_similarity(feature1, feature2):
                return np.linalg.norm(feature1 - feature2)

            # 匹配行人
            matches = []
            for i, feat1 in enumerate(features1):
                for j, feat2 in enumerate(features2):
                    dist = compute_similarity(feat1, feat2)
                    matches.append((i, j, dist))

            # 按距离排序（距离越小越相似）
            matches = sorted(matches, key=lambda x: x[2])

            # 设置匹配数量
            N = N_threshold  # 你想要显示的匹配对数
            top_matches = matches[:N]

            # 打印通信成本和匹配结果
            print(f'Top {N} matching pairs for frame {frame_id}: {top_matches}')
            print(f'Communication cost for frame {frame_id}: {comm_cost} bits')

    total_communication_cost = total_communication_cost / ((frame_end - frame_start) // frame_step + 1)
    print(f'Total communication cost: {total_communication_cost} bits')

if __name__ == "__main__":
    reference_camera_id = 1  # 示例相机编号
    unknown_camera_id = 7  # 示例相机编号
    N_threshold = 5  # 示例匹配对数

    # 使用多个 GPU 并行运行
    torch.multiprocessing.set_start_method('spawn')
    processes = []
    for gpu_id in range(4):  # 使用 GPU 0 到 3
        p = torch.multiprocessing.Process(target=main, args=(reference_camera_id, unknown_camera_id, N_threshold, gpu_id))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
