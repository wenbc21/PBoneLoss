import cv2
import numpy as np
import math
import copy

# 旋转后映射的点的位置，便于之后获取矢状面所在slice
def rotate(ps, m):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [[target_point[0][x],target_point[1][x]] for x in range(len(target_point[0]))]
    return target_point

# center_x, center_y一般都是图片的中心
# points要旋转的点
# angle是要旋转的角度
def rotate_img_and_point(img, points, angle, center_x, center_y, resize_rate=1.0):
    img = img.astype("uint8")
    h, w = img.shape
    M = cv2.getRotationMatrix2D((center_x,center_y), angle, resize_rate)
    res_img = cv2.warpAffine(img, M, (w, h))
    out_points = rotate(points, M)
    return res_img, out_points

# 将整个CBCT图像旋转一定角度，目的是方便直接根据下标获取矢状面
def rotate_plane(dcm_numpy, angle, col, center_x, center_y, resize_rate=1.0):
    height, depth, width = dcm_numpy.shape
    result = np.zeros((height, depth), dtype=np.uint8)
    for i in range(height):
        img2d = dcm_numpy[i, :, :]
        img2d = img2d.astype("uint8")
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, resize_rate)
        rotate_img = cv2.warpAffine(img2d, M, (width, depth))
        result[i, :] = rotate_img[:, col]
        # print(result[i, :], rotate_img[:, col])
        # if i == 160:
        #     cv2.imshow("img", rotate_img)

    return result

# 获取四个矢状面并保存
def rotate_results(dcm_3d_array, four_centers, four_deris, store_path):
    height, depth, width = dcm_3d_array.shape
    tooth_pos = ["12", "11", "21", "22"]
    for i in range(four_centers.shape[0]):
        dcm_3d_new = copy.deepcopy(dcm_3d_array)
        deri = four_deris[i]
        angle = math.atan(deri) * 180 / math.pi
        center_x = depth / 2
        center_y = width / 2
        tmp_img, rotate_points = rotate_img_and_point(dcm_3d_new[0, :, :], four_centers, angle, center_x, center_y)
        # print("rotate_points: ", rotate_points[i][0])
        col = int(rotate_points[i][0])
        result = rotate_plane(dcm_3d_new, angle, col, center_x, center_y)
        # cv2.imwrite("{}/{}.png".format(store_path, tooth_pos[i]), result)
        cv2.imencode(".png", result)[1].tofile("{}/{}.png".format(store_path, tooth_pos[i]))


def rotate_results_all_p(dcm_3d_array, four_centers, four_deris, store_path, cs_num):
    height, depth, width = dcm_3d_array.shape
    # tooth_pos = ["12", "11", "21", "22"]
    for i in range(four_centers.shape[0]):
        if np.random.uniform() < 0.7 :
            continue
        dcm_3d_new = copy.deepcopy(dcm_3d_array)
        deri = four_deris[i]
        angle = math.atan(deri) * 180 / math.pi
        center_x = depth / 2
        center_y = width / 2
        tmp_img, rotate_points = rotate_img_and_point(dcm_3d_new[0, :, :], four_centers, angle, center_x, center_y)
        # print("rotate_points: ", rotate_points[i][0])
        col = int(rotate_points[i][0])
        result = rotate_plane(dcm_3d_new, angle, col, center_x, center_y)
        result = result[cs_num-100:, :256]
        # cv2.imwrite("{}/{}.png".format(store_path, tooth_pos[i]), result)
        cv2.imencode(".png", result)[1].tofile("{}/{}.png".format(store_path, str(i+1).zfill(2)))

