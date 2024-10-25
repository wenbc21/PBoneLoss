from PIL.Image import Image, fromarray
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
import numpy as np
import cv2
import os
import copy

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_text(draw,
              box: list,
              cls: int,
              score: float,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                  ds,
                  fill='black',
                  font=font)
        left += text_width

# 绘制所有单个实例的mask的热力图
def draw_heatmap(masks, scores, img_path, des_root, fold=1, thresh=0.7):

    def sava_path_item(des_root, img_path, fold=1, item=1):
        # 分离路径和文件名:
        img = os.path.split(img_path)[1]
        img_pre = os.path.splitext(img)[0]
        img_post = os.path.splitext(img)[1]
        des_path = f"{des_root}/fold{fold}/{img_pre}"
        if not os.path.exists(des_path):
            os.makedirs(des_path)
        item_path = f"item{item}{img_post}"
        return os.path.join(des_path, item_path)
    
    idxs = np.greater(scores, thresh)
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    for i in range(masks.shape[0]):
        x_visualize = copy.deepcopy(masks[i])
        x_visualize = abs((((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8))#归一化并映射到0-255的整数，方便伪彩色化
        x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理COLORMAP_JET
        cv2.imwrite(sava_path_item(des_root, img_path, fold=fold, item=i), x_visualize)

# 将所有实例整合在一起的热力图
def draw_one_heatmap(masks, scores, fold=1, thresh=0.7):
    
    idxs = np.greater(scores, thresh)
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    heatmap = np.max(masks, axis=0)
    x_visualize = copy.deepcopy(heatmap)
    x_visualize = abs((((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8))#归一化并映射到0-255的整数，方便伪彩色化
    x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理COLORMAP_JET
    # cv2.imwrite(sava_path_item(des_root, img_path, fold=fold, item=i), x_visualize)
    return x_visualize

# 在图片上绘制mask
def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))

# 只绘制mask，不加上图片
def draw_masks_only(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)
    np_image = np.zeros(np_image.shape)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))

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

def draw_objs(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.7,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = True,
              alpha: float = 0.5):
    """
    将目标边界框信息，类别信息，mask信息绘制在图片上
    Args:
        image: 需要绘制的图片
        boxes: 目标边界框信息
        classes: 目标类别信息
        scores: 目标概率信息
        masks: 目标mask信息
        category_index: 类别与名称字典
        box_thresh: 过滤的概率阈值
        mask_thresh:
        line_thickness: 边界框宽度
        font: 字体类型
        font_size: 字体大小
        draw_boxes_on_image:
        draw_masks_on_image:

    Returns:

    """

    # 过滤掉低概率的目标
    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        return image

    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        colors = [ImageColor.getrgb(STANDARD_COLORS[mask_color % len(STANDARD_COLORS)]) for mask_color in range(len(masks))]
        image = draw_masks_only(image, masks, colors, mask_thresh, alpha)

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]
    if draw_boxes_on_image:
        # Draw all boxes onto image.
        draw = ImageDraw.Draw(image)
        for box, cls, score, color in zip(boxes, classes, scores, colors):
            left, top, right, bottom = box
            # 绘制目标边界框
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thickness, fill=color)
            # 绘制类别和概率信息
            # draw_text(draw, box.tolist(), int(cls), float(score), category_index, color, font, font_size)

    return image
