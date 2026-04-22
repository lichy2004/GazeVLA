import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import numpy as np
from scipy.signal import savgol_filter, medfilt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

HAND_JOINTS = [
    'left_wrist',  'left_thumb',  'left_index',  'left_middle',  'left_ring',  'left_little',
    'right_wrist', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little',
]


def bgr_to_rgb(image):
    if image.ndim == 3:  # 单张图像 [H, W, C]
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 4:  # 视频帧 [N, H, W, C]
        rgb_frames = []
        for frame in image:
            rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.array(rgb_frames)
    else:
        raise ValueError(f"不支持的图像维度: {image.ndim}，期望3或4维")

def read_entire_video(cap, bgr_to_rgb=True):
    """
    读取整个视频并返回[N, H, W, C]格式的numpy数组
    """
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if bgr_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    video_array = np.array(frames)
    return video_array

def create_rotation_transform(axis, angle_degrees):
    rotation_matrix = create_rotation_matrix(axis, angle_degrees)
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    return transform

def create_rotation_matrix(axis, angle_degrees):
    if axis == 'x':
        return create_rotation_matrix_x(angle_degrees)
    elif axis == 'y':
        return create_rotation_matrix_y(angle_degrees)
    elif axis == 'z':
        return create_rotation_matrix_z(angle_degrees)
    else:
        raise ValueError(f"Invalid axis: {axis}")

def create_rotation_matrix_x(angle_degrees):
    """绕X轴顺时针旋转"""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    return np.array([
        [1,     0,      0],
        [0,  cos_a,  sin_a],
        [0, -sin_a,  cos_a]
    ])

def create_rotation_matrix_y(angle_degrees):
    """绕Y轴顺时针旋转"""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    return np.array([
        [ cos_a,  0, -sin_a],
        [     0,  1,      0],
        [ sin_a,  0,  cos_a]
    ])

def create_rotation_matrix_z(angle_degrees):
    """绕Z轴顺时针旋转"""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    return np.array([
        [ cos_a, -sin_a, 0],
        [ sin_a,  cos_a, 0],
        [     0,      0, 1]
    ])

def rotation_matrix_to_quat_wxyz(rotation_matrix):
    """
    将旋转矩阵转换为wxyz格式的四元数
    
    Args:
        rotation_matrix: 3x3旋转矩阵
        
    Returns:
        wxyz格式的四元数 [w, x, y, z]
    """
    # 使用scipy.spatial.transform.Rotation
    r = R.from_matrix(rotation_matrix)
    quat_xyzw = r.as_quat()  # scipy默认返回[x, y, z, w]
    # 转换为[w, x, y, z]格式
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

def quat_wxyz_to_rotation_matrix(quaternion_wxyz):
    """
    将wxyz格式的四元数转换为旋转矩阵
    
    Args:
        quaternion_wxyz: wxyz格式的四元数 [w, x, y, z]
        
    Returns:
        3x3旋转矩阵
    """
    # 转换为scipy期望的[x, y, z, w]格式
    quat_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    r = R.from_quat(quat_xyzw)
    return r.as_matrix()

def euler_to_quat_wxyz(euler_angles, seq='xyz'):
    """
    将欧拉角转换为wxyz格式的四元数
    
    Args:
        euler_angles: 欧拉角数组 [angle1, angle2, angle3] (弧度)
        seq: 旋转顺序，如'xyz', 'zyx'等
        
    Returns:
        wxyz格式的四元数 [w, x, y, z]
    """
    r = R.from_euler(seq, euler_angles)
    quat_xyzw = r.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

def quat_wxyz_to_euler(quaternion_wxyz, seq='xyz'):
    """
    将wxyz格式的四元数转换为欧拉角
    
    Args:
        quaternion_wxyz: wxyz格式的四元数 [w, x, y, z]
        seq: 旋转顺序，如'xyz', 'zyx'等
        
    Returns:
        欧拉角数组 [angle1, angle2, angle3] (弧度)
    """
    quat_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    r = R.from_quat(quat_xyzw)
    return r.as_euler(seq)

def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def image_resize_with_pad(image, target_size=(224, 224)):
    """
    first resize image to target size, then pad zero to the rest of the image

    input:
    image: [bs, h, w, c] or [h, w, c]
    target_size: (h', w')

    return:
    resized_image: [bs, h', w', c] or [h', w', c]
    """
    batch_size, h, w, c = image.shape
    target_h, target_w = target_size
    
    # 计算缩放比例，保持宽高比
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # 等比例缩放
    resized_images = []
    for i in range(batch_size):
        resized = cv2.resize(image[i], (new_w, new_h))
        resized_images.append(resized)
    
    # 创建目标尺寸的黑色画布
    result = np.zeros((batch_size, target_h, target_w, c), dtype=image.dtype)
    
    # 计算居中位置
    start_h = (target_h - new_h) // 2
    start_w = (target_w - new_w) // 2
    
    # 将缩放后的图像放在中心
    for i in range(batch_size):
        result[i, start_h:start_h+new_h, start_w:start_w+new_w] = resized_images[i]
    
    return result


def image_resize_with_crop(image, target_size=(224, 224)):
    """
    first resize image to target size, then crop

    input:
    image: [bs, h, w, c] or [h, w, c]
    target_size: (h', w')

    return:
    resized_image: [bs, h', w', c] or [h', w', c]
    """
    batch_size, h, w, c = image.shape
    target_h, target_w = target_size
    
    # 计算缩放比例，保持宽高比
    scale = max(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # 等比例缩放
    resized_images = []
    for i in range(batch_size):
        resized = cv2.resize(image[i], (new_w, new_h))
        resized_images.append(resized)
    
    # 计算裁剪位置（居中）
    start_h = (new_h - target_h) // 2
    start_w = (new_w - target_w) // 2
    
    # 裁剪到目标尺寸
    result = np.zeros((batch_size, target_h, target_w, c), dtype=image.dtype).squeeze()

    for i in range(batch_size):
        result[i] = resized_images[i][start_h:start_h+target_h, start_w:start_w+target_w]
    
    return result


def gaze_resize_with_pad(gaze_pixel, origin_size, target_size=(224, 224)):
    """
    Apply the same transformation as image_resize_with_pad to gaze coordinates
    
    input:
    gaze_pixel: [bs, 2] - gaze coordinates in original image (row, col)
    origin_size: (h, w) - original image size
    target_size: (h', w') - target image size

    return:
    transformed_gaze: [bs, 2] - transformed gaze coordinates
    """
    gaze_pixel = np.array(gaze_pixel)
    origin_h, origin_w = origin_size
    target_h, target_w = target_size
    
    scale = min(target_h / origin_h, target_w / origin_w)
    new_h, new_w = int(origin_h * scale), int(origin_w * scale)
    
    start_h = (target_h - new_h) // 2
    start_w = (target_w - new_w) // 2
    
    if gaze_pixel.ndim == 1:
        scaled_gaze = gaze_pixel * scale
        transformed_gaze = scaled_gaze + np.array([start_h, start_w])
        return transformed_gaze.astype(np.int64)
    else:
        scaled_gaze = gaze_pixel * scale
        transformed_gaze = scaled_gaze + np.array([start_h, start_w])
        return transformed_gaze.astype(np.int64)


def gaze_resize_with_crop(gaze_pixel, origin_size, target_size=(224, 224)):
    """
    Apply the same transformation as image_resize_with_crop to gaze coordinates

    input:
    gaze_pixel: [bs, 2] - gaze coordinates in original image (row, col)
    origin_size: (h, w) - original image size
    target_size: (h', w') - target image size

    return:
    transformed_gaze: [bs, 2] - transformed gaze coordinates
    """
    gaze_pixel = np.array(gaze_pixel)
    origin_h, origin_w = origin_size
    target_h, target_w = target_size
    
    scale = max(target_h / origin_h, target_w / origin_w)
    new_h, new_w = int(origin_h * scale), int(origin_w * scale)
    
    start_h = (new_h - target_h) // 2
    start_w = (new_w - target_w) // 2
    
    if gaze_pixel.ndim == 1:
        scaled_gaze = gaze_pixel * scale
        transformed_gaze = scaled_gaze - np.array([start_h, start_w])
        return transformed_gaze.astype(np.int64)
    else:
        scaled_gaze = gaze_pixel * scale
        transformed_gaze = scaled_gaze - np.array([start_h, start_w])
        return transformed_gaze.astype(np.int64)


def generate_heatmap(image_size, gaze_point, sigma=2):
    '''
    Generate a heatmap for given gaze point(s)
    Args:
        image_size: [H, W] - height and width of the image
        gaze_point: [2,] or [bs, 2] - normalized gaze coordinates (0-1)
                   Single point: [h, w] or batch of points: [[h1, w1], [h2, w2], ...]
        sigma: standard deviation for the Gaussian distribution
    Returns:
        heatmap: [H, W] or [bs, H, W] - 2D heatmap(s) with Gaussian distribution centered at gaze_point(s)
    '''
    H, W = image_size
    gaze_point = np.array(gaze_point)
    
    if gaze_point.ndim == 1:
        cx = gaze_point[1] * W
        cy = gaze_point[0] * H

        x = np.arange(0, W, 1)
        y = np.arange(0, H, 1)
        xx, yy = np.meshgrid(x, y)

        heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        heatmap /= np.sum(heatmap)
        return heatmap
    elif gaze_point.ndim == 2:
        bs = gaze_point.shape[0]
        cx = gaze_point[:, 1] * W  # shape: [bs,]
        cy = gaze_point[:, 0] * H  # shape: [bs,]
        
        x = np.arange(0, W, 1)
        y = np.arange(0, H, 1)
        xx, yy = np.meshgrid(x, y)  # shape: [H, W]
        
        xx = np.broadcast_to(xx[None, :, :], (bs, H, W))
        yy = np.broadcast_to(yy[None, :, :], (bs, H, W))
        cx = cx[:, None, None]  # shape: [bs, 1, 1]
        cy = cy[:, None, None]  # shape: [bs, 1, 1]
        
        heatmaps = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        heatmaps = heatmaps / np.sum(heatmaps, axis=(1, 2), keepdims=True)
        return heatmaps
    

def visualize_heatmap(heatmap, image=None, save_path=None, dpi=100, brightness_factor=1.5):
    '''
    Visualize a heatmap and optionally overlay it on an image
    Args:
        heatmap: [H, W] - 2D heatmap array
        image: [H, W, 3] or None - background image to overlay heatmap on
        save_path: str or None - path to save the visualization
        brightness_factor: float - factor to adjust image brightness (>1.0 makes brighter, <1.0 makes darker)
    Returns:
        result_image: [H, W, 3] - visualization result as numpy array with same dimensions as input image
    '''
    # Determine target dimensions
    if image is not None:
        target_height, target_width = image.shape[:2]
        figsize = (target_width / dpi, target_height / dpi)
    else:
        target_height, target_width = heatmap.shape
        figsize = (target_width / dpi, target_height / dpi)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    if image is not None:
        # Adjust image brightness
        brightened_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        ax.imshow(brightened_image)
        ax.imshow(heatmap, alpha=0.5, cmap='hot')
        ax.set_title('Heatmap Overlay on Image')
    else:
        ax.imshow(heatmap, cmap='hot')
        ax.set_title('Gaze Heatmap')
    ax.axis('off')
    
    # Remove margins and padding to get exact dimensions
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0)
        print(f"Heatmap saved to: {save_path}")
    
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    if buf.shape[:2] != (target_height, target_width):
        buf = cv2.resize(buf, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return buf


def classify_gaze_type(
    gaze_points, 
    confidences=None, 
    saccade_threshold=0.1, 
    valid_threshold=0.05, 
    window=5
):
    """
    对归一化的gaze点进行分类（直接使用归一化阈值）
    新增：前后window帧都需要满足扫视阈值，才判定为注视，否则为扫视

    Args:
        gaze_points: [N, 2] 归一化的gaze坐标 (0-1范围)，格式为[y, x]
        confidences: [N] 置信度数组，如果为None则默认为1.0
        saccade_threshold: float, 扫视判断的归一化距离阈值（0-1范围）
        valid_threshold: float, 有效性判断的置信度阈值
        window: int, 前后多少帧都要满足阈值才判定为注视（默认2）

    Returns:
        gaze_types: [N] int8数组，每个gaze点的类别
            -1: 出界（超出视频帧）
            -2: 不valid（置信度低）
            0: 扫视（saccade，眼球快速移动）
            1: 注视（fixation，眼球相对静止）
    """
    gaze_points = np.array(gaze_points, dtype=np.float32)
    n_points = len(gaze_points)
    
    # 默认置信度为1.0
    if confidences is None:
        confidences = np.ones(n_points, dtype=np.float32)
    else:
        confidences = np.array(confidences, dtype=np.float32)
    
    # 初始化gaze类型数组
    gaze_types = np.zeros(n_points, dtype=np.int8)
    
    for i in range(n_points):
        gaze_y, gaze_x = gaze_points[i]
        
        # 规则1：检查是否出界（超出0-1范围）
        if gaze_x < 0 or gaze_x > 1 or gaze_y < 0 or gaze_y > 1:
            gaze_types[i] = -1  # 出界
            continue
            
        # 规则2：检查置信度是否足够（不valid）
        if confidences[i] < valid_threshold:
            gaze_types[i] = -2  # 不valid
            continue

        # 规则3：前后window帧都要满足移动距离小于阈值，才判定为注视
        valid_count = 0
        for offset in range(-window, window + 1):
            j = i + offset
            if j < 0 or j >= n_points:
                continue
            prev_y, prev_x = gaze_points[j]
            dist = np.sqrt((gaze_x - prev_x) ** 2 + (gaze_y - prev_y) ** 2)
            if dist <= saccade_threshold:
                valid_count += 1
        if valid_count >= int(window*1.5):
            gaze_types[i] = 1  # 注视
        else:
            gaze_types[i] = 0  # 扫视

    return gaze_types


def filter_gaze_coordinates(
    gaze_points, 
    filter_type=None, 
    window_size=5, 
    poly_order=2,
    alpha=0.3
):
    """
    对gaze坐标进行滤波处理，支持多种滤波方法，包括卡尔曼滤波
    
    Args:
        gaze_points: [N, 2] numpy数组，gaze坐标序列，格式为[y, x]
        filter_type: str, 滤波类型，支持:
            - 'moving_average': 移动平均滤波
            - 'median': 中值滤波
            - 'savgol': Savitzky-Golay滤波
            - 'exponential': 指数加权移动平均滤波
        window_size: int, 滤波窗口大小（奇数，默认5）
        sigma: float, 高斯滤波的标准差（默认1.0）
        poly_order: int, Savitzky-Golay滤波的多项式阶数（默认2）
        alpha: float, 指数加权移动平均的平滑因子（0-1，默认0.3）
        kalman_params: dict, 卡尔曼滤波参数（可选）
        
    Returns:
        filtered_gaze: [N, 2] numpy数组，滤波后的gaze坐标
    """
    if not filter_type or len(gaze_points) < 3:
        return gaze_points.copy()
    gaze_points = np.array(gaze_points, dtype=np.float32)
    if gaze_points.shape[1] != 2:
        raise ValueError(f"输入gaze_points应为[N, 2]形状，实际为{gaze_points.shape}")
    
    n_points = len(gaze_points)
    window_size = min(window_size, n_points)
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 3)
    
    filtered_gaze = gaze_points.copy()
    
    if filter_type == 'moving_average':
        # 移动平均滤波
        x = gaze_points[:, 0]
        y = gaze_points[:, 1]
        kernel = np.ones(window_size) / window_size
    
        # 用卷积计算移动平均，mode='same'保证输出长度与输入一致（边缘处理与原逻辑相同）
        filtered_x = np.convolve(x, kernel, mode='same')
        filtered_y = np.convolve(y, kernel, mode='same')
        filtered_gaze = np.column_stack((filtered_x, filtered_y))
    
            
    elif filter_type == 'median':
        # 中值滤波
        med_x = medfilt(gaze_points[:, 0], kernel_size=window_size)
        med_y = medfilt(gaze_points[:, 1], kernel_size=window_size)
        filtered_gaze = np.column_stack((med_x, med_y))
            
    elif filter_type == 'savgol':
        # Savitzky-Golay滤波
        # 确保多项式阶数小于窗口大小
        smooth_x = savgol_filter(gaze_points[:, 0], window_length=window_size, polyorder=poly_order, mode='nearest')
        smooth_y = savgol_filter(gaze_points[:, 1], window_length=window_size, polyorder=poly_order, mode='nearest')
        filtered_gaze = np.column_stack((smooth_x, smooth_y))
            
    elif filter_type == 'exponential':
        # 指数加权移动平均滤波 (EWMA)
        filtered_gaze[0] = gaze_points[0]  # 第一个点保持不变
        x = gaze_points[:, 0]
        y = gaze_points[:, 1]
        
        # 生成衰减系数：(1-alpha)^0, (1-alpha)^1, ..., (1-alpha)^(n_points-1)
        decay = np.power(1 - alpha, np.arange(n_points))
        
        # 计算每个位置的加权和（向量化核心）
        # 对X坐标：filtered_x[i] = alpha * sum(x[0..i] * decay[i..0])
        filtered_x = alpha * np.convolve(x, decay, mode='full')[:n_points]
        filtered_y = alpha * np.convolve(y, decay, mode='full')[:n_points]
        filtered_gaze = np.column_stack((filtered_x, filtered_y))
    else:
        raise ValueError(f"不支持的滤波类型: {filter_type}。支持的类型: "
                        "'moving_average', 'median', 'savgol', 'exponential'")
    
    return filtered_gaze


def adaptive_filter_gaze_coordinates(
    gaze_points, 
    confidences=None, 
    velocity_threshold=0.05, 
    low_conf_threshold=0.3,
    static_filter='median',
    dynamic_filter='moving_average',
    **filter_kwargs
):
    """
    自适应gaze坐标滤波：根据gaze运动速度和置信度选择不同的滤波策略
    
    Args:
        gaze_points: [N, 2] numpy数组，gaze坐标序列
        confidences: [N] numpy数组，置信度序列（可选）
        velocity_threshold: float, 速度阈值，用于区分静态和动态gaze
        low_conf_threshold: float, 低置信度阈值
        static_filter: str, 静态区域使用的滤波方法
        dynamic_filter: str, 动态区域使用的滤波方法
        **filter_kwargs: 其他滤波参数
        
    Returns:
        filtered_gaze: [N, 2] numpy数组，自适应滤波后的gaze坐标
        filter_mask: [N] numpy数组，标记每个点使用的滤波类型（0=静态，1=动态）
    """
    gaze_points = np.array(gaze_points, dtype=np.float32)
    n_points = len(gaze_points)
    
    if n_points < 3:
        return gaze_points.copy(), np.zeros(n_points, dtype=int)
    
    # 计算速度（相邻帧之间的距离）
    velocities = np.zeros(n_points)
    velocities[1:] = np.linalg.norm(gaze_points[1:] - gaze_points[:-1], axis=1)
    velocities[0] = velocities[1]  # 第一帧使用第二帧的速度
    
    # 判断静态/动态区域
    is_dynamic = velocities > velocity_threshold
    
    # 如果有置信度信息，低置信度区域优先使用更强的滤波
    if confidences is not None:
        confidences = np.array(confidences)
        low_conf_mask = confidences < low_conf_threshold
        # 低置信度区域强制使用静态滤波（更强的平滑）
        is_dynamic = is_dynamic & (~low_conf_mask)
    
    # 分别对静态和动态区域应用不同滤波
    filtered_gaze = np.zeros_like(gaze_points)
    filter_mask = is_dynamic.astype(int)
    
    # 静态区域滤波（更强的平滑）
    static_indices = np.where(~is_dynamic)[0]
    if len(static_indices) > 0:
        static_gaze = filter_gaze_coordinates(
            gaze_points[static_indices], 
            filter_type=static_filter, 
            **filter_kwargs
        )
        filtered_gaze[static_indices] = static_gaze
    
    # 动态区域滤波（保留更多细节）
    dynamic_indices = np.where(is_dynamic)[0]
    if len(dynamic_indices) > 0:
        # 动态区域使用较小的窗口
        dynamic_kwargs = filter_kwargs.copy()
        if 'window_size' in dynamic_kwargs:
            dynamic_kwargs['window_size'] = max(3, dynamic_kwargs['window_size'] // 2)
        
        dynamic_gaze = filter_gaze_coordinates(
            gaze_points[dynamic_indices], 
            filter_type=dynamic_filter, 
            **dynamic_kwargs
        )
        filtered_gaze[dynamic_indices] = dynamic_gaze
    
    return filtered_gaze, filter_mask

def transform_images(images):
    # Convert to numpy array
    if isinstance(images, torch.Tensor):
        images_np = images.detach().cpu().numpy()
    else:
        images_np = np.array(images)
    
    # Convert from [-1, 1] to [0, 255]
    images_np = ((images_np + 1.0) * 127.5).astype(np.uint8)
    
    # Transform dimensions: [batch_size, channels, height, width] -> [batch_size, height, width, channels]
    if len(images_np.shape) == 4:
        images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    return images_np