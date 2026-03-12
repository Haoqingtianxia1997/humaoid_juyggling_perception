import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def find_matching_depth_file(rgb_filename, depth_files):
    """找到与 RGB 图片时间戳最接近的深度文件"""
    rgb_timestamp = rgb_filename.replace('.png', '')
    
    # 找到最接近的深度文件
    closest_file = None
    min_diff = float('inf')
    
    for depth_file in depth_files:
        depth_timestamp = depth_file.replace('.npy', '')
        # 计算时间戳差异（简单字符串比较）
        diff = abs(int(depth_timestamp.replace('_', '')) - int(rgb_timestamp.replace('_', '')))
        if diff < min_diff:
            min_diff = diff
            closest_file = depth_file
    
    return closest_file


def detect_spheres_contour(masked_rgb):
    """使用轮廓检测球体"""
    # 转换为灰度图
    gray = cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # 形态学操作，去除噪声

    open_kernel = np.ones((21, 21), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    close_kernel = np.ones((21, 21), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选圆形轮廓
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000 or area > 9000:  # 过滤太小或太大的轮廓
            continue
        
        # 计算圆形度（越接近 1 越圆）
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 如果圆形度足够高，认为是球体
        if circularity > 0.7:
            # 获取最小外接圆的半径（但不使用其中心点）
            _, radius = cv2.minEnclosingCircle(contour)
            
            # 找到轮廓的极值点
            contour_points = contour.reshape(-1, 2)
            
            # 最左点和最右点
            leftmost_idx = contour_points[:, 0].argmin()
            leftmost = tuple(contour_points[leftmost_idx])
            rightmost_idx = contour_points[:, 0].argmax()
            rightmost = tuple(contour_points[rightmost_idx])
            
            # 最上点和最下点
            topmost_idx = contour_points[:, 1].argmin()
            topmost = tuple(contour_points[topmost_idx])
            bottommost_idx = contour_points[:, 1].argmax()
            bottommost = tuple(contour_points[bottommost_idx])
            
            # 计算水平线和垂直线的交点作为中心
            # 水平线：从最左到最右
            # 垂直线：从最上到最下
            x = int((leftmost[0] + rightmost[0]) / 2)
            y = int((topmost[1] + bottommost[1]) / 2)
            
            circles.append((x, y, int(radius), contour, area))
    
    return circles

def process_image(rgb_path, depth_path, output_path, depth_threshold=1.0):
    """处理单张图片"""
    # 读取 RGB 图片
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        print(f"无法读取图片: {rgb_path}")
        return False
    
    # 读取深度数据
    try:
        depth_data = np.load(depth_path)
    except Exception as e:
        print(f"无法读取深度数据: {depth_path}, 错误: {e}")
        return False
    
    # 确保深度数据和 RGB 图片尺寸一致
    if depth_data.shape[:2] != rgb_img.shape[:2]:
        print(f"深度数据尺寸 {depth_data.shape} 与 RGB 图片尺寸 {rgb_img.shape} 不匹配")
        # 尝试调整深度数据大小
        depth_data = cv2.resize(depth_data, (rgb_img.shape[1], rgb_img.shape[0]))
    
    # 创建深度 mask（保留深度 <= 1m 的像素）
    # 假设深度单位是米，如果是毫米则需要调整阈值
    depth_mask = depth_data <= depth_threshold
    
    # 创建 masked RGB 图片
    masked_rgb = rgb_img.copy()
    masked_rgb[~depth_mask] = 0  # 将深度大于 1m 的像素设为黑色
    
    # 计算下方1/3的分界线位置
    height = masked_rgb.shape[0]
    boundary_y = int(height * 3 / 4)  # 上方2/3和下方1/3的分界线
    
    # 创建检测区域mask（只检测上方2/3）
    detection_mask = np.ones(masked_rgb.shape[:2], dtype=bool)
    detection_mask[boundary_y:, :] = False  # 下方1/3不检测
    
    # 将下方1/3区域设为黑色
    masked_rgb_for_detection = masked_rgb.copy()
    masked_rgb_for_detection[~detection_mask] = 0

    circles_contour = detect_spheres_contour(masked_rgb_for_detection)
    
    result_img = masked_rgb.copy()

    cv2.line(result_img, (0, boundary_y), (result_img.shape[1], boundary_y), 
             (255, 255, 0), 2)  # 青色分界线
    
    # 画轮廓检测到的圆
    for circle_data in circles_contour:
        x, y, radius, contour, area = circle_data
        center = (x, y)
        
        # 计算圆心处的深度值（使用较小的 ROI）
        roi_radius = int(radius * 0.3)  # 使用0.3倍半径避开边缘
        
        # 创建圆形 mask
        y_grid, x_grid = np.ogrid[:depth_data.shape[0], :depth_data.shape[1]]
        circle_mask = (x_grid - x)**2 + (y_grid - y)**2 <= roi_radius**2
        
        # 提取 ROI 区域的深度值
        depth_roi = depth_data[circle_mask]
        
        # 过滤无效值（NaN、Inf、0）
        valid_depths = depth_roi[(~np.isnan(depth_roi)) & 
                                 (~np.isinf(depth_roi)) & 
                                 (depth_roi > 0)]
        
        # 计算深度（使用修剪均值）
        if len(valid_depths) > 0:
            # 去掉上下10%的值，使用修剪均值
            if len(valid_depths) >= 10:
                sorted_depths = np.sort(valid_depths)
                trim_count = int(len(sorted_depths) * 0.1)
                trimmed_depths = sorted_depths[trim_count:-trim_count] if trim_count > 0 else sorted_depths
                depth_at_center = np.mean(trimmed_depths)
            else:
                # 数据点少时使用中位数
                depth_at_center = np.median(valid_depths)
        else:
            depth_at_center = 0.0
        
        # 画轮廓（使用红色）
        cv2.drawContours(result_img, [contour], -1, (0, 0, 255), 2)
        # 画 ROI 圆（用于显示实际计算深度的区域，使用紫色）
        cv2.circle(result_img, center, roi_radius, (255, 0, 255), 1)
        # 画圆心（使用绿色）
        cv2.circle(result_img, center, 3, (0, 255, 0), -1)
        
        # 在圆上方显示面积和深度信息
        text1 = f"Area: {int(area)} px"
        text2 = f"Depth: {depth_at_center:.3f} m"
        
        # 计算文本位置
        text_size1 = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_size2 = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        max_text_width = max(text_size1[0], text_size2[0])
        
        text_x = x - max_text_width // 2
        text_y1 = y - radius - 35
        text_y2 = y - radius - 10
        
        # 绘制第一行文本背景（面积）
        cv2.rectangle(result_img, (text_x - 5, text_y1 - text_size1[1] - 5), 
                     (text_x + text_size1[0] + 5, text_y1 + 5), (0, 0, 0), -1)
        # 绘制第一行文本
        cv2.putText(result_img, text1, (text_x, text_y1), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制第二行文本背景（深度）
        cv2.rectangle(result_img, (text_x - 5, text_y2 - text_size2[1] - 5), 
                     (text_x + text_size2[0] + 5, text_y2 + 5), (0, 0, 0), -1)
        # 绘制第二行文本
        cv2.putText(result_img, text2, (text_x, text_y2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # 也可以直接画轮廓
        # cv2.drawContours(result_img, [contour], -1, (255, 0, 0), 2)
    
    # 保存结果
    cv2.imwrite(output_path, result_img)
    
    return True

def create_video_from_images(image_dir, output_video_path, fps=30):
    """将图片序列合成为视频"""
    # 获取所有处理后的图片
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    if len(image_files) == 0:
        print("没有找到图片，无法创建视频")
        return False
    
    # 读取第一张图片以获取尺寸
    first_image = cv2.imread(str(image_dir / image_files[0]))
    if first_image is None:
        print("无法读取图片")
        return False
    
    height, width, _ = first_image.shape
    
    # 尝试多种编码器，优先使用 H.264
    codecs = [
        ('avc1', '.mp4'),  # H.264 (最通用)
        ('H264', '.mp4'),  # H.264 的另一种写法
        ('X264', '.mp4'),  # x264 编码器
        ('mp4v', '.mp4'),  # MPEG-4
        ('XVID', '.avi'),  # Xvid (备选方案)
    ]
    
    video_writer = None
    final_output_path = output_video_path
    
    for fourcc_str, ext in codecs:
        try:
            # 如果需要，更改输出文件扩展名
            if not str(output_video_path).endswith(ext):
                final_output_path = str(output_video_path).rsplit('.', 1)[0] + ext
            
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            video_writer = cv2.VideoWriter(str(final_output_path), fourcc, fps, (width, height))
            
            if video_writer.isOpened():
                print(f"使用编码器: {fourcc_str}")
                break
            else:
                video_writer.release()
                video_writer = None
        except Exception as e:
            print(f"编码器 {fourcc_str} 不可用: {e}")
            continue
    
    if video_writer is None or not video_writer.isOpened():
        print("所有编码器都无法使用，尝试使用 ffmpeg...")
        video_writer.release() if video_writer else None
        return create_video_with_ffmpeg(image_dir, output_video_path, fps)
    
    # 将每张图片写入视频
    for image_file in tqdm(image_files, desc="合成视频"):
        image_path = image_dir / image_file
        frame = cv2.imread(str(image_path))
        
        if frame is not None:
            video_writer.write(frame)
        else:
            print(f"警告：无法读取图片 {image_file}")
    
    # 释放视频写入器
    video_writer.release()
    
    print(f"视频创建成功：{final_output_path}")
    return True

def create_video_with_ffmpeg(image_dir, output_video_path, fps=30):
    """使用 ffmpeg 命令行工具合成视频（备选方案）"""
    import subprocess
    
    # 创建临时文件列表
    temp_list_file = image_dir / 'temp_image_list.txt'
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    try:
        with open(temp_list_file, 'w') as f:
            for img_file in image_files:
                f.write(f"file '{img_file}'\n")
        
        # 使用 ffmpeg 合成视频
        cmd = [
            'ffmpeg', '-y',
            '-r', str(fps),
            '-f', 'concat',
            '-safe', '0',
            '-i', str(temp_list_file),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            str(output_video_path)
        ]
        
        result = subprocess.run(cmd, cwd=str(image_dir), capture_output=True, text=True)
        
        # 删除临时文件
        temp_list_file.unlink()
        
        if result.returncode == 0:
            print(f"使用 ffmpeg 创建视频成功：{output_video_path}")
            return True
        else:
            print(f"ffmpeg 错误: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("未找到 ffmpeg，请安装 ffmpeg: sudo apt install ffmpeg")
        return False
    except Exception as e:
        print(f"使用 ffmpeg 创建视频时出错: {e}")
        if temp_list_file.exists():
            temp_list_file.unlink()
        return False

def calculate_fps_from_timestamps(image_files):
    """根据图片时间戳计算帧率"""
    if len(image_files) < 2:
        return 30.0  # 默认 30 fps
    
    try:
        # 提取前几个文件的时间戳
        timestamps = []
        for i in range(min(10, len(image_files))):
            # 时间戳格式: YYYYMMDD_HHMMSS_microseconds
            timestamp_str = image_files[i].replace('.png', '').replace('.npy', '')
            parts = timestamp_str.split('_')
            if len(parts) == 3:
                # 转换为微秒
                microseconds = int(parts[2])
                timestamps.append(microseconds)
        
        # 计算平均时间间隔
        if len(timestamps) >= 2:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval_microseconds = np.mean([abs(x) for x in intervals if x != 0])
            
            # 转换为帧率
            if avg_interval_microseconds > 0:
                fps = 1000000.0 / avg_interval_microseconds
                print(f"根据时间戳计算的帧率: {fps:.2f} fps")
                return min(max(fps, 1.0), 120.0)  # 限制在 1-120 fps 之间
    except Exception as e:
        print(f"计算帧率时出错: {e}")
    
    return 30.0  # 默认 30 fps

def main():
    # 定义路径
    workspace_root = Path(__file__).parent.parent.parent
    rgb_dir = workspace_root / 'data' / 'rgb'
    depth_dir = workspace_root / 'data' / 'depth'
    output_dir = workspace_root / 'data' / 'processed_rgb'
    tracking_output_dir = workspace_root / 'data' / 'tracking_output'
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 RGB 图片和深度文件
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    
    print(f"找到 {len(rgb_files)} 张 RGB 图片和 {len(depth_files)} 个深度文件")
    
    # 深度阈值（米）
    depth_threshold = 1.0
    
    # 处理每张图片
    success_count = 0
    for rgb_file in tqdm(rgb_files, desc="处理图片"):
        # 找到对应的深度文件
        depth_file = find_matching_depth_file(rgb_file, depth_files)
        
        if depth_file is None:
            print(f"未找到与 {rgb_file} 匹配的深度文件")
            continue
        
        rgb_path = rgb_dir / rgb_file
        depth_path = depth_dir / depth_file
        output_path = output_dir / rgb_file
        
        # 处理图片
        if process_image(str(rgb_path), str(depth_path), str(output_path), depth_threshold):
            success_count += 1
    
    print(f"\n处理完成！成功处理 {success_count}/{len(rgb_files)} 张图片")
    print(f"结果保存在: {output_dir}")
    
    # 合成视频
    if success_count > 0:
        print("\n开始合成视频...")
        # 计算帧率
        fps = calculate_fps_from_timestamps(rgb_files)
        
        # 视频速度调整系数（大于1加速，小于1减速）
        speed_multiplier = 0.5 # 修改这个值来调整视频速度
        # 例如：2.0 = 2倍速，0.5 = 0.5倍速（慢放）
        fps = fps * speed_multiplier
        
        print(f"视频帧率: {fps:.2f} fps (速度: {speed_multiplier}x)")
        
        # 生成视频文件名（使用第一张图片的日期时间）
        if rgb_files:
            first_timestamp = rgb_files[0].split('_')[0] + '_' + rgb_files[0].split('_')[1]
            video_filename = f"sphere_detection_{first_timestamp}.mp4"
        else:
            from datetime import datetime
            video_filename = f"sphere_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        video_path = tracking_output_dir / video_filename
        
        if create_video_from_images(output_dir, video_path, fps):
            print(f"视频已保存到: {video_path}")
        else:
            print("视频合成失败")
    else:
        print("\n没有成功处理的图片，跳过视频合成")

if __name__ == "__main__":
    main()
