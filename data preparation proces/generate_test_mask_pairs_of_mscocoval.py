#!/usr/bin/env python
# coding=utf-8

import argparse
import math
import random
import json
import cv2
from scipy.interpolate import splprep, splev
import os, tarfile, logging
from skimage.color import rgb2gray
from skimage.feature import canny
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from accelerate.utils import set_seed

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple training script.")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for reproducibility.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for input images.")
    parser.add_argument("--train_data_dir", type=str, default='/projects/yonglzha_proj/Our_sketch_textInp/PartialSketchNet/data/MSCOCO2014-val-tar/filtered_coco_val2014_with_edges_underTwoAnns.tar', help="Path to the tar file.")
    parser.add_argument("--mscoco_annotations", type=str, default='/projects/yonglzha_proj/Our_sketch_textInp/PartialSketchNet/data/mscoco2014/annotations/annotations', help="Path to the annotations file.")
    parser.add_argument("--save_dir", type=str,
                        default='/projects/yonglzha_proj/Our_sketch_textInp/PartialSketchNet/data/MSCOCO2014-val-tar/test_baseline_dataset_basedon_filtered_coco_val2014_with_edges_underTwoAnnstar',
                        help="Path to the save dir file.")
    parser.add_argument("--random_mask", action="store_true", help="Enable random mask training.")
    parser.add_argument('--area_range', type=float, nargs=2, default=(0.3, 0.5), help='Range of partial mask area.')
    parser.add_argument('--dilate_interval_len', type=int, default=5, help='Interval length for dilation.')
    parser.add_argument('--gaussian_blur_interval_len', type=int, default=5, help='Interval length for Gaussian blur.')

    args = parser.parse_args(input_args) if input_args else parser.parse_args()
    if args.resolution % 8 != 0:
        raise ValueError("Resolution must be divisible by 8 for consistent encoding.")
    return args


class MyWebDataset():
    def __init__(self, resolution, random_mask, annotations_dirs, mask_area_range, dilate_len, blur_len, save_dir, canny_edge=None):
        self.resolution = resolution
        self.random_mask = random_mask
        self.canny_indicator = canny_edge
        self.save_dir = save_dir
        self.mapping_files = {}
        # 读取 COCO 标注信息
        self.coco_instances = COCO(os.path.join(annotations_dirs, "instances_val2014.json"))
        self.coco_captions = COCO(os.path.join(annotations_dirs, "captions_val2014.json"))
        if not self.random_mask:
            self.d_masks = []
            self.s_masks = {}
            self.mask_area_range = mask_area_range
            self.dilate_len= dilate_len
            self.blur_len = blur_len

    def random_brush_gen(self, max_tries, h, w, min_num_vertex=0, max_num_vertex=8, mean_angle=2 * math.pi / 5, angle_range=2 * math.pi / 15, min_width=128, max_width=128):
        H, W = h, w
        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 0)
        for _ in range(np.random.randint(max_tries)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=1)
            if np.random.random() > 0.5:
                mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.uint8)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 0)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 1)
        return mask # value 1 represent holes

    def random_mask_gen(self, h, w):
        mask = np.ones((h, w), np.uint8)
        # random_brush_gen() 中非空洞区域（即值为 0 的地方，）才能在最终 mask 中保留为 True
        mask = np.logical_and(mask, 1 - self.random_brush_gen(4, h, w))  # hole denoted as 0, reserved as 1
        return mask[np.newaxis, ...].astype(np.float32)

    def generate_initial_bezier_curve(self, mask_shape, num_points=5):
        """
        生成一条随机初始曲线，并展示生成过程。支持上下左右4个方向生成。
        :param mask_shape: 掩模的形状
        :param num_points: 生成曲线的控制点数量
        :return: 曲线的坐标
        """
        height, width = mask_shape
        direction = np.random.choice(['up', 'down', 'left', 'right'])

        if direction == 'down':
            points = np.zeros((num_points, 2), dtype=np.int32)
            points[:, 0] = np.linspace(0, width - 1, num_points)
            points[:, 1] = np.random.randint(0, height // 20, num_points)
        elif direction == 'up':
            points = np.zeros((num_points, 2), dtype=np.int32)
            points[:, 0] = np.linspace(0, width - 1, num_points)
            points[:, 1] = np.random.randint(height // 20 * 19, height, num_points)
        elif direction == 'right':
            points = np.zeros((num_points, 2), dtype=np.int32)
            points[:, 1] = np.linspace(0, height - 1, num_points)
            points[:, 0] = np.random.randint(0, width // 20, num_points)
        elif direction == 'left':
            points = np.zeros((num_points, 2), dtype=np.int32)
            points[:, 1] = np.linspace(0, height - 1, num_points)
            points[:, 0] = np.random.randint(width // 20 * 19, width, num_points)

        # plt.figure(figsize=(10, 10))
        # plt.xlim(0, width)
        # plt.ylim(height, 0)
        #
        # # 绘制初始控制点
        # plt.scatter(points[:, 0], points[:, 1], color='blue')
        # plt.title('Control Points for Initial Curve')
        # plt.pause(1.0)  # 暂停以展示控制点

        # 生成贝塞尔曲线
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        u_fine = np.linspace(0, 1, 500)
        x_fine, y_fine = splev(u_fine, tck)

        # 绘制生成的曲线
        # plt.plot(x_fine, y_fine, color='red')
        # plt.title('Generated Initial Curve')
        # plt.pause(1.0)  # 暂停以展示生成的曲线
        #
        # plt.show()

        return np.array([x_fine, y_fine], dtype=np.int32).T, direction

    def scan_with_bezier_curve(self, mask, curve, direction, image_id, area_range=(0.3, 0.5), step=0.1):
        """
        使用曲线扫描算法获取非零遮挡区域的部分区域，并展示扫描过程。
        :param mask: 二值化掩模，mask区域用值1表示
        :param curve: 扫描的初始曲线
        :param direction: 曲线扫描的方向
        :param area_range: 扫描到的目标区域占整个物体区域的比例区间
        :return: 扫描得到的部分遮挡区域
        """
        height, width = mask.shape
        total_area = np.sum(mask == 1)
        start_curve = curve
        # 根据输入的范围和步长划分区间
        start, end = area_range
        target_area_ranges = [(round(start + i * step, 2), round(min(start + (i + 1) * step, end), 2))
                              for i in range(round((end - start) / step))]
        # 随机选择一个子区间
        chosen_range = random.choice(target_area_ranges)

        # 在选择的区间中均匀采样一个比例
        target_area_ratio = np.random.uniform(chosen_range[0], chosen_range[1])
        target_area = total_area * target_area_ratio

        scanned_area = np.zeros_like(mask)
        accumulated_area = 0
        # plt.figure(figsize=(10, 10))
        # Step 1: 初始方向扫描
        for i in range(max(height, width)):
            # 更新曲线位置，根据当前方向移动
            if direction == 'down':
                curve[:, 1] += 1
            elif direction == 'up':
                curve[:, 1] -= 1
            elif direction == 'right':
                curve[:, 0] += 1
            elif direction == 'left':
                curve[:, 0] -= 1

            # 保证曲线不会超出图像边界
            curve = np.clip(curve, 0, [width - 1, height - 1])
            rr, cc = curve[:, 1], curve[:, 0]
            scanned_area[rr, cc] = mask[rr, cc]
            accumulated_area = np.sum(scanned_area == 1)

            # 显示扫描过程
            # if i % 10 == 0 or accumulated_area >= target_area:  # 控制显示的频率
            #     plt.imshow(mask, cmap='gray')
            #     plt.plot(curve[:, 0], curve[:, 1], color='red', lw=2)  # 绘制当前曲线
            #     plt.imshow(scanned_area, cmap='gray', alpha=0.5)  # 显示累积扫描区域
            #     plt.title(f'Scanned Area: {accumulated_area}/{total_area} pixels')
            #     plt.pause(0.1)  # 短暂暂停以显示进度

            if accumulated_area >= target_area:
                break
        # plt.show()
        # Step 2: 如果初始方向不成功，反转方向
        if accumulated_area < target_area:
            print(
                f"Initial direction '{direction}' did not reach target area for image id '{image_id}', reversing direction.")
            # 反转方向
            if direction == 'down':
                direction = 'up'
            elif direction == 'up':
                direction = 'down'
            elif direction == 'right':
                direction = 'left'
            elif direction == 'left':
                direction = 'right'

            # 重新初始化扫描区域和累积面积
            # scanned_area = np.zeros_like(mask)
            # accumulated_area = 0
            # plt.figure(figsize=(10, 10))
            # 继续反向扫描
            for i in range(max(height, width)):
                # 更新曲线位置，根据反转后的方向移动
                if direction == 'down':
                    start_curve[:, 1] += 1
                elif direction == 'up':
                    start_curve[:, 1] -= 1
                elif direction == 'right':
                    start_curve[:, 0] += 1
                elif direction == 'left':
                    start_curve[:, 0] -= 1

                # 保证曲线不会超出图像边界
                start_curve = np.clip(start_curve, 0, [width - 1, height - 1])
                rr, cc = start_curve[:, 1], start_curve[:, 0]
                scanned_area[rr, cc] = mask[rr, cc]
                accumulated_area = np.sum(scanned_area == 1)

                # # 显示扫描过程
                # if i % 10 == 0 or accumulated_area >= target_area:  # 控制显示的频率
                #     plt.imshow(mask, cmap='gray')
                #     plt.plot(start_curve[:, 0], start_curve[:, 1], color='red', lw=2)  # 绘制当前曲线
                #     plt.imshow(scanned_area, cmap='gray', alpha=0.5)  # 显示累积扫描区域
                #     plt.title(f'Scanned Area: {accumulated_area}/{total_area} pixels')
                #     plt.pause(0.1)  # 短暂暂停以显示进度

                if accumulated_area >= target_area:
                    break
            # plt.show()
        # 形态学操作填充扫描区域，避免黑线
        kernel = np.ones((3, 3), np.uint8)
        scanned_area = cv2.dilate(scanned_area, kernel, iterations=1)

        # 保留最大的联通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(scanned_area, connectivity=8)

        if stats.shape[0] > 1:  # 确保有超过一个连通区域
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 忽略背景标签0
            largest_connected_area = np.zeros_like(scanned_area)
            largest_connected_area[labels == largest_label] = 1  # 将最大区域的像素设为1
        else:
            print(f"No connected components found in the scanned area for '{image_id}', returning random mask.")
            largest_connected_area = 1.0 - self.random_mask_gen(mask.shape[0], mask.shape[1])[0] # 1 represents inpainting holes

        return largest_connected_area


    def dilate_mask(self, mask, bbox_mask, d, D):
        if d == 0:
            return mask
        elif d >= D:
            return bbox_mask
        else:
            kd = int((d / D) * D)
            dilation_kernel = np.ones((kd, kd), np.uint8)
            dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1)
            return dilated_mask

    # 定义高斯模糊处理的函数
    def gaussian_blur_between_masks(self, dilated_mask1, dilated_mask2, s, S):
        if s == 0:
            return dilated_mask1
        elif s >= S:
            return dilated_mask2
        else:
            # 根据s计算高斯核大小和标准差
            ks = int((s / S) * S)
            if ks % 2 == 0:  # 核大小必须是奇数
                ks += 1
            cs = (s / S) * S  # 标准差根据s线性变化
            # 应用高斯模糊
            blurred_mask = cv2.GaussianBlur(dilated_mask1, (ks, ks), sigmaX=0)

            return blurred_mask

    # 定义完整的mask precision predictor
    def mask_precision_predictor(self, segmentation_mask, bbox_mask, d, D, S):
        # global  d_masks
        # global  s_masks
        # 获取膨胀后的掩模 d
        d1 = self.dilate_mask(segmentation_mask, bbox_mask, d, D)
        self.d_masks.append(d1)
        if len(self.d_masks) >= 2:
            # get last two elements
            d_mask1, d_mask2 = self.d_masks[-2:]
            s_values = np.linspace(0, 1, self.blur_len)
            for s_value in s_values:
                s = s_value * S
                blurred_mask = self.gaussian_blur_between_masks(d_mask1, d_mask2, s, S)
                self.s_masks['d=' + str(d) + ' ' + 's=' + str(s)] = blurred_mask
        return self.d_masks, self.s_masks

    def process_data_from_tar(self, tar_path):
        """
        从给定的 tar 文件中处理数据
        """
        captions_dict = {}

        with tarfile.open(tar_path, 'r') as archive:
            members = archive.getmembers()
            for member in members:
                if member.isfile() and member.name.endswith(".image"):
                    class_img_name = member.name.split('.')[0]
                    complete_sketches = []
                    self.d_masks = []
                    self.s_masks = {}
                    # 1. 从 COCO 实例中获取标注信息
                    captions = self.coco_captions.loadAnns(self.coco_captions.getAnnIds(imgIds=int(class_img_name)))
                    caption_text = random.choice([cap['caption'] for cap in captions])

                    # 2. 处理图像数据（使用 OpenCV）,ndarray(h,w,3)
                    image_file = archive.extractfile(f"{class_img_name}.image")
                    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR) # 解码为 BGR 格式的 OpenCV 图像

                    # 3. 提取最显著物体
                    anns = self.coco_instances.loadAnns(self.coco_instances.getAnnIds(imgIds=int(class_img_name)))
                    most_significant_ann = max(anns, key=lambda ann: np.sum(self.coco_instances.annToMask(ann)))
                    seg_mask = self.coco_instances.annToMask(most_significant_ann).astype(np.uint8)
                    # 初始化 partial_mask_is_bbox
                    partial_mask_is_bbox = False

                    # for template caption: a photo of a + category
                    caption_text = "A photo of a " + self.coco_instances.loadCats(most_significant_ann['category_id'])[0]['name'] + "."

                    # 4. 处理 bounding box 信息,{list:4}
                    bbox = most_significant_ann['bbox']
                    x, y, w, h = map(int, bbox)
                    bbox_mask = np.zeros_like(seg_mask)
                    bbox_mask[y:y + h, x:x + w] = 1

                    # 定义最大膨胀核大小 D 和高斯模糊强度 S
                    D = max(w, h) / 2  # D 与物体的 bbox 尺寸有关
                    S = max(w, h) / 2  # S 是手动设定的最大模糊强度，可以根据需求调整
                    d_values = np.linspace(0, 1, self.dilate_len)
                    for d_value in d_values:
                        d = d_value * D
                        d_masks, s_masks = self.mask_precision_predictor(seg_mask, bbox_mask, d, D, S)
                    # plt.figure(figsize=(100, 100))
                    # i = 0
                    # # show s_masks and partial masks
                    # for key, mask in s_masks.items():
                    #     plt.subplot(int(len(s_masks)/len(d_values)), 2*len(d_values), i+1)
                    #     initial_curve, direction = self.generate_initial_bezier_curve(mask.shape)
                    #     stroke_mask = self.scan_with_bezier_curve(mask, initial_curve, direction, example['__key__'], area_range=self.mask_area_range)
                    #     # initial_curve = generate_initial_spline_curve(mask.shape)
                    #     # stroke_mask = scan_with_spline_curve(mask, initial_curve)
                    #     plt.imshow(cv2.resize(mask,(512, 512), interpolation=cv2.INTER_NEAREST), cmap='gray')
                    #     plt.title(f'{key}')
                    #     plt.axis('off')
                    #     # i += 1
                    #     plt.subplot(int(len(s_masks)/len(d_values)), 2*len(d_values), i+2)
                    #     plt.imshow(cv2.resize(stroke_mask,(512, 512), interpolation=cv2.INTER_NEAREST), cmap='gray')
                    #     plt.title('partial mask')
                    #     plt.axis('off')
                    #     i += 2
                    #
                    # # show s_masks
                    # for key, mask in s_masks.items():
                    #     plt.subplot(int(len(s_masks)/len(d_values)), len(d_values), i+1)
                    #     plt.imshow(mask, cmap='gray')
                    #     # cv2.resize(mask,(512, 512), interpolation=cv2.INTER_NEAREST)
                    #     plt.title(f'{key}')
                    #     plt.axis('off')
                    #     i += 1
                    # plt.show()
                    # 随机选择一个掩模
                    ds_mask = random.choice(list(s_masks.values())[self.blur_len:len(s_masks) - self.blur_len]) # 1 is the partial mask inpainting region
                    # 随机贝塞尔曲线生成partial掩模
                    initial_curve, direction = self.generate_initial_bezier_curve(ds_mask.shape)
                    stroke_mask = self.scan_with_bezier_curve(ds_mask, initial_curve, direction, class_img_name, area_range=self.mask_area_range)  # 1 is the partial mask inpainting region
                    partial_mask = stroke_mask[:, :, np.newaxis]  # 1 is the partial mask inpainting region

                    # random for segmentation mask and bbox mask
                    if random.random() < 0.1:
                        partial_mask = random.choice([d_masks[0], d_masks[-1]])[:, :, np.newaxis]
                        # 检查是否使用 bbox 掩模
                        if np.array_equal(partial_mask, d_masks[-1][:, :, np.newaxis]):  # if bbox mask is choiced
                            partial_mask_is_bbox = True

                    # Step 2: 判断是否覆盖为随机掩模
                    if self.random_mask:
                        partial_mask = 1.0 - self.random_mask_gen(image.shape[0], image.shape[1])[0][:, :, np.newaxis] # 1 represents inpainting holes


                    # 处理图像和掩码 of inpainting task
                    partial_mask = 1 - partial_mask # 0 is the partial mask inpainting region
                    masked_image = image * partial_mask

                    # random for outpainting
                    # if random.random() < 0.5:
                    #     masked_image = image - masked_image
                    #     partial_mask = 1 - partial_mask

                    if self.canny_indicator:
                        canny_edge = canny(rgb2gray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), sigma=2).astype(np.uint8) * 255
                        canny_edge = canny_edge[:, :, np.newaxis] # ndarray(h,w,1) uint8, value 255 for canny edge regions
                        # canny_edge = canny_edge.astype(np.float32) / 255.
                        complete_sketches.append(canny_edge)
                    # pidinet for soft-threshold edge
                    # pidinet_edge = cv2.imdecode(np.asarray(bytearray(example["pidinet_edge"]), dtype="uint8"), cv2.IMREAD_COLOR) # 获取pidinet edge cv2.IMREAD_COLOR 会将图像读取为 三通道 BGR format, 即便输入图像是灰度图
                    # if np.all(pidinet_edge[:, :, 0] == pidinet_edge[:, :, 1]) and np.all(pidinet_edge[:, :, 1] == pidinet_edge[:, :, 2]):
                    #     pidinet_edge = pidinet_edge[:,:,0][:, :, np.newaxis]
                    pidinet_file = archive.extractfile(f"{class_img_name}.pidinet_edge")
                    pidinet_sigmoid_edge = cv2.imdecode(np.frombuffer(pidinet_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
                    # pidinet_sigmoid_edge = 1.0 - pidinet_sigmoid_edge.astype(np.float32) / 255.0  # ndarray(h,w,1) in [0~1], value nearly 1.0 for edge regions
                    pidinet_sigmoid_edge = 255 - pidinet_sigmoid_edge # value nearly 255 for edge regions

                    complete_sketches.append(pidinet_sigmoid_edge)
                    # T2I-adapter for edge extraction
                    half_hard_edge = (pidinet_sigmoid_edge > 127.5).astype(np.uint8) * 255
                    complete_sketches.append(half_hard_edge)
                    complete_sketch = random.choice(complete_sketches)

                    # Resize and Normalize images
                    image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)
                    masked_image = cv2.resize(masked_image, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)
                    partial_mask = cv2.resize(partial_mask, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]
                    complete_sketch = cv2.resize(complete_sketch, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]

                    image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5) - 1.0
                    masked_image = (cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5) - 1.0
                    partial_mask = partial_mask.astype(np.float32) # 0 is the partial mask inpainting region
                    complete_sketch = complete_sketch.astype(np.float32) / 255. # value 1 nearly for edge regions
                    partial_sketch = (1. - partial_mask) * complete_sketch
                    # since partial mask is enlarged partially based on seg_mask, again impose seg_mask on partial_sketch to
                    # get object partial sketch inside partial_sketch
                    seg_mask = cv2.resize(seg_mask, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]  # use segmentation mask and value 1 is mask inpainting region
                    partial_sketch = seg_mask.astype(np.float32)*partial_sketch

                    if partial_mask_is_bbox:
                        partial_sketch = cv2.resize(d_masks[0], (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis] # use segmentation mask and value 1 is mask inpainting region
                        partial_sketch = partial_sketch.astype(np.float32) * complete_sketch

                    # add full object sketch for global clues
                    full_obj_sketch = seg_mask.astype(np.float32) * complete_sketch

                    image_uint8 = ((image + 1) * 127.5).astype(np.uint8)
                    image_uint8 = Image.fromarray(image_uint8, mode='RGB')
                    os.makedirs(os.path.join(self.save_dir, 'images'), exist_ok=True) or image_uint8.save(
                        os.path.join(self.save_dir, 'images', class_img_name + '.png'), 'PNG')

                    masked_image_uint8 = ((masked_image + 1) * 127.5).astype(np.uint8)
                    masked_image_uint8 = Image.fromarray(masked_image_uint8, mode='RGB')
                    os.makedirs(os.path.join(self.save_dir, 'masked_images'), exist_ok=True) or masked_image_uint8.save(
                        os.path.join(self.save_dir, 'masked_images', class_img_name + '_masked.png'), 'PNG')

                    unique_elements_m, counts_m = np.unique(partial_mask, return_counts=True)
                    p_mask_uint8 = (partial_mask * 255).astype(np.uint8)
                    unique_elements_mm, counts_mm = np.unique(p_mask_uint8, return_counts=True)
                    p_mask_uint8 = Image.fromarray(np.concatenate([p_mask_uint8, p_mask_uint8, p_mask_uint8], axis=2), mode='RGB')
                    os.makedirs(os.path.join(self.save_dir, 'p_masks'), exist_ok=True) or p_mask_uint8.save(
                        os.path.join(self.save_dir, 'p_masks', class_img_name + '_p_mask.png'), 'PNG')

                    p_sketch_uint8 = (partial_sketch * 255).astype(np.uint8)
                    p_sketch_uint8 = Image.fromarray(np.concatenate([p_sketch_uint8, p_sketch_uint8, p_sketch_uint8], axis=2), mode='RGB')
                    os.makedirs(os.path.join(self.save_dir, 'partial_sketches'), exist_ok=True) or p_sketch_uint8.save(
                        os.path.join(self.save_dir, 'partial_sketches', class_img_name + '_p_sketch.png'), 'PNG')

                    f_sketch_uint8 = (full_obj_sketch * 255).astype(np.uint8)
                    f_sketch_uint8 = Image.fromarray(np.concatenate([f_sketch_uint8, f_sketch_uint8, f_sketch_uint8], axis=2), mode='RGB')
                    os.makedirs(os.path.join(self.save_dir, 'full_sketches'), exist_ok=True) or f_sketch_uint8.save(
                        os.path.join(self.save_dir, 'full_sketches', class_img_name + '_f_sketch.png'), 'PNG')

                    seg_mask_uint8 = ((1 - seg_mask) * 255).astype(np.uint8)
                    seg_mask_uint8 = Image.fromarray(np.concatenate([seg_mask_uint8, seg_mask_uint8, seg_mask_uint8], axis=2), mode='RGB')
                    os.makedirs(os.path.join(self.save_dir, 'seg_mask'), exist_ok=True) or seg_mask_uint8.save(
                        os.path.join(self.save_dir, 'seg_mask', class_img_name + '_seg_mask.png'), 'PNG')

                    bbox_mask_uint8 = ((1 - cv2.resize(bbox_mask, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]) * 255).astype(np.uint8)
                    bbox_mask_uint8 = Image.fromarray(np.concatenate([bbox_mask_uint8, bbox_mask_uint8, bbox_mask_uint8], axis=2), mode='RGB')
                    os.makedirs(os.path.join(self.save_dir, 'bbox_mask'), exist_ok=True) or bbox_mask_uint8.save(
                        os.path.join(self.save_dir, 'bbox_mask', class_img_name + '_bbox_mask.png'), 'PNG')

                    # 将图像名称（去掉扩展名）作为键，caption 作为值，存入字典
                    captions_dict[class_img_name] = caption_text

        json_file_path = os.path.join(self.save_dir, 'captions_template.json')
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(captions_dict, f, indent=4, ensure_ascii=False)

def main(args):
    if args.seed is not None:
        set_seed(args.seed)
    dataset_instance = MyWebDataset(resolution=args.resolution, random_mask=args.random_mask,annotations_dirs=args.mscoco_annotations,
                                mask_area_range=args.area_range, dilate_len=args.dilate_interval_len,
                                blur_len=args.gaussian_blur_interval_len,
                                save_dir=args.save_dir, canny_edge=None)

    import logging

    logging.basicConfig(level=logging.INFO)

    # 在关键部分插入日志
    logging.info("开始处理数据")
    try:
        dataset_instance.process_data_from_tar(args.train_data_dir)
    except Exception as e:
        logging.error(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
