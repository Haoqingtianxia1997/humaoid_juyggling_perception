import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime
import argparse
import message_filters


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
        if circularity > 0.5:
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
            x = int((leftmost[0] + rightmost[0]) / 2)
            y = int((topmost[1] + bottommost[1]) / 2)
            
            circles.append((x, y, int(radius), contour, area))
    
    return circles


class ZedImageSaver(Node):
    def __init__(self, save_enabled=False):
        super().__init__('zed_image_saver')
        self.bridge = CvBridge()

        # 保存开关
        self.save_enabled = save_enabled

        # 创建订阅器并使用消息同步
        rgb_sub = message_filters.Subscriber(
            self,
            Image,
            '/zed/zed_node/left/image_rect_color'
        )
        
        depth_sub = message_filters.Subscriber(
            self,
            Image,
            '/zed/zed_node/depth/depth_registered'
        )
        
        # 使用近似时间同步器（允许10ms的时间差）
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=10,
            slop=0.01  # 10ms 时间容差
        )
        self.sync.registerCallback(self.images_callback)

        self.rgb_dir = os.path.expanduser('~/zed_ws/data/rgb')
        self.depth_dir = os.path.expanduser('~/zed_ws/data/depth')
        self.depth_image_dir = os.path.expanduser('~/zed_ws/data/depth_image')

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.depth_image_dir, exist_ok=True)

        # 用于存储最新图像以便显示
        self.latest_rgb = None
        self.latest_depth_colormap = None
        self.latest_depth_data = None  # 存储原始深度数据用于计算

        # 创建显示窗口
        cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)

        # 创建定时器用于检查键盘输入和刷新显示
        self.timer = self.create_timer(0.03, self.update_display)

        self.get_logger().info("✅ ZED Image Saver Node Started.")
        self.get_logger().info(f"RGB images -> {self.rgb_dir}")
        self.get_logger().info(f"Depth images -> {self.depth_dir}")
        self.get_logger().info(f"Depth images (visual) -> {self.depth_image_dir}")
        self.get_logger().info("按 'q' 退出")
        self.get_logger().info(f"保存状态: {'开启' if self.save_enabled else '关闭'}")

    def images_callback(self, rgb_msg, depth_msg):
        """
        同步接收 RGB 和深度图像（保证时间对齐）
        
        Args:
            rgb_msg: RGB 图像消息
            depth_msg: 深度图像消息
        """
        # 处理 RGB 图像
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        # 调整图像大小到 640x360
        cv_image = cv2.resize(cv_image, (640, 360))
        # 顺时针旋转90度
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        
        # 处理深度图像
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        # 调整深度图大小到 640x360
        cv_depth = cv2.resize(cv_depth, (640, 360), interpolation=cv2.INTER_NEAREST)
        # 顺时针旋转90度
        cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_90_CLOCKWISE)
        
        # 生成可视化深度图
        valid_mask = np.isfinite(cv_depth) & (cv_depth > 0)
        if np.any(valid_mask):
            valid_depth = cv_depth[valid_mask]
            min_depth = np.percentile(valid_depth, 1)
            max_depth = np.percentile(valid_depth, 99)
            depth_clipped = np.clip(cv_depth, min_depth, max_depth)
            depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros(cv_depth.shape, dtype=np.uint8)
        
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # 创建用于显示的 masked RGB（根据深度mask掉1m之外的像素）
        cv_image_display = cv_image.copy()
        # 创建深度mask：深度在0到1米之间的像素
        depth_mask = (cv_depth > 0) & (cv_depth <= 1.0) & np.isfinite(cv_depth)
        # 将深度大于1米的区域设置为黑色
        cv_image_display[~depth_mask] = 0
        
        # === 球体检测部分 ===
        # 计算下方1/3的分界线位置
        height = cv_image_display.shape[0]
        boundary_y = int(height * 3 / 4)  # 上方2/3和下方1/3的分界线
        
        # 创建检测区域mask（只检测上方2/3）
        detection_mask = np.ones(cv_image_display.shape[:2], dtype=bool)
        detection_mask[boundary_y:, :] = False  # 下方1/3不检测
        
        # 将下方1/3区域设为黑色用于检测
        masked_rgb_for_detection = cv_image_display.copy()
        masked_rgb_for_detection[~detection_mask] = 0
        
        # 检测球体
        circles_contour = detect_spheres_contour(masked_rgb_for_detection)
        
        # 创建结果图像（在完整的masked_rgb上绘制）
        result_img = cv_image_display.copy()
        
        # 画分界线
        cv2.line(result_img, (0, boundary_y), (result_img.shape[1], boundary_y), 
                 (255, 255, 0), 2)  # 青色分界线
        
        # 画检测到的球体
        for circle_data in circles_contour:
            x, y, radius, contour, area = circle_data
            center = (x, y)
            
            # 计算圆心处的深度值（使用较小的 ROI）
            roi_radius = int(radius * 0.3)  # 使用0.3倍半径避开边缘
            
            # 创建圆形 mask
            y_grid, x_grid = np.ogrid[:cv_depth.shape[0], :cv_depth.shape[1]]
            circle_mask = (x_grid - x)**2 + (y_grid - y)**2 <= roi_radius**2
            
            # 提取 ROI 区域的深度值
            depth_roi = cv_depth[circle_mask]
            
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
        
        # 存储最新图像用于显示（使用带检测结果的版本）
        self.latest_rgb = result_img
        self.latest_depth_colormap = depth_colormap.copy()
        self.latest_depth_data = cv_depth.copy()
        
        # 如果保存开关打开，则保存图像（使用相同的时间戳）
        if self.save_enabled:
            # 使用 RGB 消息时间戳（两个消息已同步，时间戳应该相同或非常接近）
            timestamp = rgb_msg.header.stamp
            timestamp_str = f"{timestamp.sec}_{timestamp.nanosec:09d}"
            
            # 保存 RGB 图像
            rgb_filename = timestamp_str + ".png"
            rgb_path = os.path.join(self.rgb_dir, rgb_filename)
            cv2.imwrite(rgb_path, cv_image)
            
            # 保存原始深度数据为 .npy
            npy_filename = timestamp_str + ".npy"
            npy_path = os.path.join(self.depth_dir, npy_filename)
            np.save(npy_path, cv_depth)

            # 保存可视化深度图为 .png
            depth_png_filename = timestamp_str + ".png"
            depth_png_path = os.path.join(self.depth_image_dir, depth_png_filename)
            cv2.imwrite(depth_png_path, depth_colormap)

            self.get_logger().info(f"Saved: RGB={rgb_path}, Depth={npy_path}")

    def update_display(self):
        """更新显示窗口和处理键盘输入"""
        # 显示RGB图像
        if self.latest_rgb is not None:
            display_rgb = self.latest_rgb.copy()
            # 添加状态文字
            status_text = f"Save: {'ON' if self.save_enabled else 'OFF'} | Press 'q' to quit"
            cv2.putText(display_rgb, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('RGB', display_rgb)
        
        # 显示深度图
        if self.latest_depth_colormap is not None:
            cv2.imshow('Depth', self.latest_depth_colormap)
        
        # 检查键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("退出程序...")
            cv2.destroyAllWindows()
            rclpy.shutdown()


def main(args=None):
    """启动图像保存节点（用于调试）"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ZED Image Saver')
    parser.add_argument('--save', action='store_true', help='启动时开启保存功能')
    parsed_args = parser.parse_args()
    
    rclpy.init(args=args)
    node = ZedImageSaver(save_enabled=parsed_args.save)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
    
    
    
if __name__ == '__main__':
    main()  