import SimpleITK as sitk
import os
import argparse
import numpy as np
import cv2
import copy
import pandas as pd
import numpy as np
import copy
import time
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk 
from PIL import Image, ImageDraw
import json
from scipy.stats import norm

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs, mask_tooth_center


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--cross_section_path', type=str, default='results/cross_section')
    parser.add_argument('--results_path', type=str, default='results')
    parser.add_argument('--weights_file', type=str, default='weights/model_79.pth')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    return parser


def get_roi(args) :
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    cross_section_dirs = [item.path for item in os.scandir(args.cross_section_path) if item.is_file()]
    cross_section_dirs.sort()

    os.makedirs(f"{args.results_path}/instance_seg", exist_ok=True)
    os.makedirs(f"{args.results_path}/centers", exist_ok=True)

    # 创建模型
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=1+1,
                     rpn_score_thresh=0.5,
                     box_score_thresh=0.5)
    
    # load train weights 载入模型权重并传送到对应设备
    assert os.path.exists(args.weights_file), "{} file dose not exist.".format(args.weights_file)
    weights_dict = torch.load(args.weights_file, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)
    
    for it in range(len(cross_section_dirs)) :
        # get dicom file and adjustment value according to window
        cross_section_path = cross_section_dirs[it]
        original_img = Image.open(cross_section_path).convert('RGB')
        img_name = os.path.split(cross_section_path)[-1].split(".")[0]

        # from pil image to tensor, do not normalize image 将图像转化为tensor数据以进行训练
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init 初始化
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            # 预测
            predictions = model(img.to(device))[0]

            # 读出预测结果
            predict_boxes = predictions["boxes"].to("cpu").numpy() # 矩形框信息
            predict_classes = predictions["labels"].to("cpu").numpy() # 分类信息
            predict_scores = predictions["scores"].to("cpu").numpy() # 评价box/mask分数
            predict_mask = predictions["masks"].to("cpu").numpy() # mask信息，每个mask经过处理可以为每个牙齿(item)
            predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            # 绘制结果图展示
            plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=None,
                             line_thickness=1,
                             font='arial.ttf',
                             font_size=20,
                             alpha=1,
                             box_thresh=0.96)
            # 保存预测的图片结果
            plot_img.save(f"{args.results_path}/instance_seg/{img_name}_seg.png")

            # 用连通区域函数获取牙齿中心
            centers = mask_tooth_center(original_img, 
                                boxes=predict_boxes, 
                                classes=predict_classes,
                                scores=predict_scores,
                                masks=predict_mask,
                                thresh=0.96)
            
            # 取上面7颗
            y_index = np.argsort(centers[:, 1], kind='stable')
            y_centers = centers[y_index] 
            y_centers = y_centers[:7]
            
            # 计算上7牙之间的左右邻牙距离
            x_mean = np.mean(y_centers[:,0])
            y_left_centers = y_centers[y_centers[:,0] < x_mean]
            y_right_centers = y_centers[y_centers[:,0] >= x_mean]
            y_left_index = np.argsort(y_left_centers[:, 0] - y_left_centers[:, 1], kind='stable')
            y_left_centers = y_left_centers[y_left_index]
            y_right_index = np.argsort(y_right_centers[:, 0] + y_right_centers[:, 1], kind='stable')
            y_right_centers = y_right_centers[y_right_index]
            sorted_centers = np.append(y_left_centers, y_right_centers, axis=0)
            distances = []
            for i in range(1, len(sorted_centers)) :
                distances.append(np.linalg.norm(sorted_centers[i-1] - sorted_centers[i]))
            
            # 取最大距离的两颗牙，作为缺牙的邻牙，中心即为缺牙位点
            max_dis = np.argmax(distances)
            neighbour1 = sorted_centers[max_dis]
            neighbour2 = sorted_centers[max_dis+1]
            missing = (neighbour1 + neighbour2) / 2
            
            # 打开一张图片
            image = cv2.imread(cross_section_path)
            for center in centers:
                cv2.circle(image, center, radius=2, color=(0, 0, 255), thickness=2)
            for center in y_centers:
                cv2.circle(image, center, radius=2, color=(255, 0, 0), thickness=2)
            cv2.circle(image, neighbour1, radius=2, color=(0, 255, 255), thickness=2)
            cv2.circle(image, neighbour2, radius=2, color=(0, 255, 255), thickness=2)
            cv2.circle(image, (int(missing[0]), int(missing[1])), radius=2, color=(0, 255, 255), thickness=2)
            cv2.imwrite(f"{args.results_path}/centers/{img_name}_centers.png", image)

        print(f"{img_name} done! ")



# 在每个mask上取出牙齿的中心
def mask_tooth_center(image, boxes, classes, scores, masks, box_thresh: float = 0.7, thresh:float = 0.7):
     # 过滤掉低概率的目标
    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)
    centers = []
    for mask in masks:
        mask = mask.astype("uint8")
        ret, labels, stats, centroid = cv2.connectedComponentsWithStats(mask)
        centroid = centroid.astype(int)
        if len(centroid) >= 2:
            centers.append(centroid[1])
    return np.array(centers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    get_roi(args)