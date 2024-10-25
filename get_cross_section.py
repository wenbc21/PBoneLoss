import SimpleITK as sitk
import os
from get_dicom import get_dcm_3d_array, window_transform_3d
import argparse
import numpy as np
import cv2
import copy


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dicom_path', type=str, default='datasets/CBCT')
    parser.add_argument('--results_path', type=str, default='results')

    return parser


def get_cross_section(args) :

    dicom_dirs = [item.path for item in os.scandir(args.dicom_path) if item.is_dir()]
    dicom_dirs.sort()

    os.makedirs(f"{args.results_path}/mip", exist_ok=True)
    os.makedirs(f"{args.results_path}/cross_section", exist_ok=True)
    
    for it in range(len(dicom_dirs)) :
        # get dicom file and adjustment value according to window
        dicom_dir = list([item.path for item in os.scandir(dicom_dirs[it]) if item.is_dir()])[0]
        dicom = get_dcm_3d_array(dicom_dir)
        dicom = window_transform_3d(dicom, window_width=1700, window_center=1500).astype(np.uint8)

        # Maximum Intensity Projection
        mip_img = np.max(dicom, axis=2)
        mip_img[mip_img != 255] = 0

        # open and close
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        mip_img = cv2.morphologyEx(mip_img, cv2.MORPH_OPEN, kernel, iterations=1)
        mip_img = cv2.morphologyEx(mip_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # get roi slice base on corner detection
        bi_img, coordinates = harris_corner_detection(mip_img, gamma=0.25)  # 对投影图使用哈里斯角点检测
        index = np.argsort(coordinates[:, 1])  # 输出第二维度从小到大排序的索引序列
        coordinates = coordinates[index]  # 将角点坐标序列排序
        del_idx = np.array([], dtype=np.int16)
        for i in range(coordinates.shape[0]):
            x = int(coordinates[i, 0])
            y = int(coordinates[i, 1])
            if np.count_nonzero(mip_img[x-1, (y-1):(y+2)]) >= 2 or x >= 200:
                del_idx = np.append(del_idx, int(i))
        
        coordinates = np.delete(coordinates, del_idx, axis=0)  # 删除不符合要求的角点
        coordinates = coordinates.astype(int)
        ori_coordinates = coordinates
        
        # find anomalies
        coordinates_y = coordinates[:, 0]
        data_std = np.std(coordinates_y)
        data_mean = np.mean(coordinates_y)
        anomaly = data_std * 3

        coordinates = []
        for num in coordinates_y:
            if num <= data_mean + anomaly and num >= data_mean - anomaly :
                coordinates.append(num)

        cs_num = np.max(coordinates[:len(coordinates) // 4])   # 选择前三个坐标中最大的
        cross_section = dicom[cs_num, :, :]
        cv2.imwrite(f"{args.results_path}/cross_section/{str(it+1).zfill(4)}_cross_section.png", cross_section)
        
        mip_img = cv2.cvtColor(mip_img, cv2.COLOR_GRAY2BGR)
        cv2.line(mip_img, (0, cs_num), (mip_img.shape[1], cs_num), (255, 0, 0), 2)
        for coor in ori_coordinates:
            cv2.circle(mip_img, coor[::-1], radius=3, color=(0, 0, 255), thickness=-1)  # 填充圆
        cv2.imwrite(f"{args.results_path}/mip/{str(it+1).zfill(4)}_mip.png", mip_img)

        print(f"{str(it+1).zfill(4)} done! ")


def harris_corner_detection(img, gamma):
    # print(gamma)
    img = img.astype("uint8")  # 转换格式
    img = np.float32(img)
    width, height = img.shape
    
    # 对图像执行harris
    Harris_detector = cv2.cornerHarris(img, 2, 3, 0.04)  # cv库的函数
    # img - 数据类型为 float32 的输入图像。
    # blockSize - 角点检测中要考虑的领域大小。
    # ksize - Sobel 求导中使用的窗口大小
    # k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06]

    dst = Harris_detector
    # 设置阈值
    thres = gamma * dst.max()  # 阈值，大于thres为角点, gamma值可以考究一下
    # print('thres =', thres)
    gray_img = copy.deepcopy(img)  # 复制一个投影图
    gray_img[dst <= thres] = 0
    gray_img[dst > thres] = 255
    gray_img = gray_img.astype("uint8")

    coor = np.array([])
    for i in range(width):
        for j in range(height):
            if gray_img[i][j] == 255:  # 角点
                # print([i, j])
                coor = np.append(coor, [i, j], axis=0)  # 嵌入坐标
    
    coor = np.reshape(coor, (-1, 2))  # 变成两列
    return gray_img, coor


if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    get_cross_section(args)