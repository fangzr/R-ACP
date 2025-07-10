import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image
from matplotlib.colors import LinearSegmentedColormap


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.lambda_camera = 0.5 # camera sensing rate/ s
        self.capacity = 100 # throughput(KB/s)
        #self.alpha = 1e- #alpha

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        gt_losses = 0 
        bits_losses = 0

        precision_s, recall_s = AverageMeter(), AverageMeter()
        for batch_idx, (data, map_gt, _, _) in enumerate(data_loader):

            optimizer.zero_grad()
            map_res, bits_loss, _  = self.model(data)
            loss = 0
            gt_loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel)

            loss = gt_loss +  bits_loss * 1e-4 

            loss.backward()
            optimizer.step()


            losses += loss.item()
            gt_losses += gt_loss.item()
            bits_losses += bits_loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            # print("pred size",pred.size())

            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                print('Epoch: {}, batch: {}, loss: {:.6f}, gt_losses: {:.6f}, communication cost: {:.2f} KB'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), gt_losses / (batch_idx + 1), bits_losses/(batch_idx + 1)))

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
        from matplotlib.colors import LinearSegmentedColormap

        # 定义自定义颜色映射
        colors = [(0.0, 'darkblue'), (0.5, 'green'), (1.0, 'red')]
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

        print("res_fpath", res_fpath)
        print("gt_fpath", gt_fpath)
        self.model.eval()
        losses = 0
        bits_losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        output_map_res_statistic = 0
        total_aopt = 0
        num_frames = 0  # 计算帧数
        

        if res_fpath is not None:
            assert gt_fpath is not None

            # 获取 res_fpath 的目录路径，并在其下创建 test_data 子目录
            base_dir = os.path.dirname(res_fpath)
            test_data_dir = os.path.join(base_dir, 'test_data')
            if not os.path.exists(test_data_dir):
                os.makedirs(test_data_dir)

        for batch_idx, (data, map_gt, _, frame) in enumerate(data_loader):
            with torch.no_grad():
                # 获取 map_res, bits_loss 和 map_results
                map_res, bits_loss, map_results = self.model(data)


            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)

                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                            data_loader.dataset.grid_reduce, v_s], dim=1))            

            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel)
            output_map_res_statistic += torch.sum(map_res)
            losses += loss.item()
            bits_losses += bits_loss.item()

            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            # 保存 map_res 结果为图片
            if visualize:
                for b in range(map_res.size(0)):
                    result_image = map_res[b, 0, :, :].cpu().numpy()

                    # 处理当前帧的编号
                    current_frame = frame[b].item() if isinstance(frame, torch.Tensor) and frame.numel() > 1 else frame.item()

                    # 创建 frame_id 子目录
                    frame_dir = os.path.join(test_data_dir, f"frame_{current_frame}")
                    if not os.path.exists(frame_dir):
                        os.makedirs(frame_dir)

                    # 保存热力图
                    image_path = os.path.join(frame_dir, f"batch_{b}_{current_frame}_map_res.png")
                    plt.imsave(image_path, result_image, cmap=custom_cmap)

            # 处理 map_results 的每个相机部分，生成并保存为 .txt 文件
            num_cam = self.model.num_cam
            for cam_num in range(num_cam):
                cam_map_res = map_results[:, cam_num, :, :]  # 提取每个相机的 map_results

                # 遍历每个 batch
                for b in range(cam_map_res.size(0)):
                    map_grid_res = cam_map_res[b].detach().cpu().squeeze()

                    # 提取超过阈值的分数
                    scores = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)

                    # 获取超过阈值的坐标
                    positions = (map_grid_res > self.cls_thres).nonzero()

                    if positions.nelement() > 0:
                        # print(f"Positions detected for camera {cam_num} at frame {current_frame}. Saving...")
                        # current_frame = frame[b].item() if isinstance(frame, torch.Tensor) and frame.numel() > 1 else frame.item()

                        # 将相机编号、帧 ID、坐标和分数保存到 .txt 文件
                        results = torch.cat([torch.ones_like(scores) * cam_num,
                                            torch.ones_like(scores) * current_frame,
                                            positions.float(), scores], dim=1)
                        res_txt_path = os.path.join(frame_dir, f"camera_{cam_num}_coordinates.txt")
                        # print(f"Saving coordinates to: {res_txt_path}")

                        with open(res_txt_path, 'w') as f:
                            np.savetxt(f, results.numpy(), fmt='%.8f')

                        # 保存相机 map_result 结果为图片
                        cam_result_image = cam_map_res[b, :, :].cpu().numpy()
                        image_path = os.path.join(frame_dir, f"batch_{b}_{current_frame}_{cam_num}_map_res.png")
                        plt.imsave(image_path, cam_result_image, cmap=custom_cmap)

                        # 进行 NMS 处理坐标
                        positions_nms, scores_nms = results[:, 2:4], results[:, 4]
                        ids, count = nms(positions_nms, scores_nms, 20, np.inf)  # NMS 处理
                        filtered_positions = positions_nms[ids[:count], :]

                        # 保存 NMS 处理后的坐标
                        nms_txt_path = os.path.join(frame_dir, f"camera_{cam_num}_nms_coordinates.txt")
                        # print(f"Saving NMS coordinates to: {nms_txt_path}")
                        with open(nms_txt_path, 'w') as f:
                            np.savetxt(f, filtered_positions.numpy(), fmt='%.8f')
                            
                        # AoPT计算相关
                        nms_data = np.loadtxt(nms_txt_path)
                        num_targets = len(nms_data) if nms_data.ndim > 1 else 1  # NMS文件的行数即为目标数目
                        

                        print(f"bits_loss: {bits_loss.item()}, capacity: {self.capacity}")
                        print(f"Camera {cam_num} at frame {frame} detected {num_targets} targets, 
                        
                    else:
                        print(f"No positions detected for camera {cam_num} at frame {current_frame}. Skipping...")

            num_frames += 1

            


        moda = 0

        print('test gt losses', losses, 'statistic', output_map_res_statistic)

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.join(base_dir, 'all_res.txt'), all_res_list.numpy(), '%.8f')
            res_list = []
            for frame_num in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame_num, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame_num, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                    data_loader.dataset.base.__name__)

            print('moda: {:.2f}%, modp: {:.2f}%, precision: {:.2f}%, recall: {:.2f}%'.
                format(moda, modp, precision, recall))

        print('Communication cost: {:.2f} KB'.format(bits_losses / (len(data_loader))))

        return losses / len(data_loader), precision_s.avg * 100, moda, bits_losses / (len(data_loader))


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    