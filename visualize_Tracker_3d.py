#!/usr/bin/env python3
"""
使用Open3D可视化球体追踪轨迹
逐帧显示点云、检测位置和卡尔曼滤波位置
关闭窗口后显示下一帧
"""

import json
import numpy as np
import open3d as o3d
import cv2
import argparse
import yaml
from pathlib import Path


class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, trajectory_json_path):
        """
        初始化可视化器
        
        Args:
            trajectory_json_path: 轨迹JSON文件路径
        """
        self.json_path = Path(trajectory_json_path)
        self.data_dir = self.json_path.parent
        
        # 加载轨迹数据
        with open(self.json_path, 'r') as f:
            self.trajectory = json.load(f)
        
        self.tracker_id = self.trajectory['tracker_id']
        self.frames = self.trajectory['frames']
        self.image_dir = self.data_dir / self.trajectory['image_dir']
        
        print(f"Loaded trajectory for tracker {self.tracker_id}")
        print(f"Total frames: {len(self.frames)}")
        print(f"Image directory: {self.image_dir}")
        
        # 加载相机配置
        config_path = Path(__file__).parent / 'Tracker_config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"相机配置文件不存在: {config_path}")
        
        with open(config_path, 'r') as f:
            camera_config = yaml.safe_load(f)
        print(f"Loaded camera config from: {config_path}")
        
        # 相机内参（ZED相机）
        intr = camera_config['intrinsics']
        self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=intr['width'],
            height=intr['height'],
            fx=intr['fx'],
            fy=intr['fy'],
            cx=intr['cx'],
            cy=intr['cy']
        )
        
        # 相机外参
        extr = camera_config['extrinsics']
        camera_position_in_body = np.array(extr['position'])
        camera_rotation_in_body = np.array(extr['rotation'])
        
        # 构建从相机坐标系到body坐标系的4x4变换矩阵
        self.camera_to_body_transform = np.eye(4)
        self.camera_to_body_transform[:3, :3] = camera_rotation_in_body
        self.camera_to_body_transform[:3, 3] = camera_position_in_body
    
    def create_point_cloud_from_depth(self, rgb_image, depth_array):
        """
        从RGB和深度图像创建点云
        
        Args:
            rgb_image: RGB图像（BGR格式）
            depth_array: 深度数组
            
        Returns:
            Open3D点云对象
        """
        # 转换BGR到RGB
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # 创建Open3D图像
        color_image = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_image = o3d.geometry.Image(depth_array.astype(np.float32))
        
        # 创建RGBD图像
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, 
            depth_image,
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        )
        
        # 创建点云（相机光学坐标系）
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            self.camera_intrinsics
        )
        
        # 将点云从相机坐标系转换到body坐标系（imu_link）
        pcd.transform(self.camera_to_body_transform)
        
        return pcd
    
    def create_sphere_marker(self, position, color, radius=0.025, wireframe=False):
        """
        创建球体标记
        
        Args:
            position: 位置 [x, y, z]
            color: 颜色 [r, g, b]
            radius: 半径（米）
            wireframe: 是否使用线框模式（实现透明效果）
            
        Returns:
            Open3D mesh对象或LineSet对象
        """
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        sphere.translate(position)
        
        if wireframe:
            # 转换为线框模式实现透明效果
            lines = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
            lines.paint_uniform_color(color)
            return lines
        else:
            return sphere
    
    def create_coordinate_frame(self, size=0.2):
        """
        创建坐标系
        
        Args:
            size: 坐标轴长度
            
        Returns:
            Open3D坐标系对象
        """
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    def visualize_frame(self, frame_idx):
        """
        可视化单帧
        
        Args:
            frame_idx: 帧索引
        """
        frame = self.frames[frame_idx]
        
        # 加载RGB和深度图像
        rgb_path = self.image_dir / frame['rgb_file']
        depth_path = self.image_dir / frame['depth_file']
        
        if not rgb_path.exists() or not depth_path.exists():
            print(f"警告: 帧 {frame_idx} 的图像文件不存在")
            return
        
        rgb_image = cv2.imread(str(rgb_path))
        depth_array = np.load(str(depth_path))
        
        # 创建点云
        pcd = self.create_point_cloud_from_depth(rgb_image, depth_array)
        
        # 下采样点云（加速显示）
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        # 创建几何体列表
        geometries = [pcd]
        
        # 添加坐标系（imu_link原点，body坐标系）
        coord_frame = self.create_coordinate_frame(size=0.2)
        geometries.append(coord_frame)
        
        # 添加检测位置标记（红色，实心，半径25mm）
        if frame['detection_pos'] is not None:
            detection_sphere = self.create_sphere_marker(
                frame['detection_pos'],
                color=[1.0, 0.0, 0.0],  # 红色
                radius=0.025,  # 25mm
                wireframe=False  # 实心
            )
            geometries.append(detection_sphere)
        
        # 添加卡尔曼滤波位置标记（蓝色，线框半透明，半径37.5mm）
        if frame['kf_pos'] is not None:
            kf_sphere = self.create_sphere_marker(
                frame['kf_pos'],
                color=[0.0, 0.0, 1.0],  # 蓝色
                radius=0.0375,  # 37.5mm
                wireframe=True  # 线框模式（透明效果）
            )
            geometries.append(kf_sphere)
        
        # 添加速度向量（绿色箭头）
        if frame['kf_vel'] is not None and frame['kf_pos'] is not None:
            kf_pos = np.array(frame['kf_pos'])
            kf_vel = np.array(frame['kf_vel'])
            
            # 速度向量缩放（用于可视化）
            vel_scale = 0.1
            vel_end = kf_pos + kf_vel * vel_scale
            
            # 创建箭头（使用圆柱体和圆锥）
            arrow_length = np.linalg.norm(kf_vel * vel_scale)
            if arrow_length > 0.001:  # 避免零长度
                # 归一化方向
                direction = (vel_end - kf_pos) / arrow_length
                
                # 创建圆柱体（箭身）
                cylinder_height = arrow_length * 0.8
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=0.005,
                    height=cylinder_height
                )
                cylinder.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
                
                # 创建圆锥（箭头）
                cone_height = arrow_length * 0.2
                cone = o3d.geometry.TriangleMesh.create_cone(
                    radius=0.01,
                    height=cone_height
                )
                cone.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
                
                # 计算旋转矩阵（从Z轴对齐到velocity方向）
                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, direction)
                rotation_axis_norm = np.linalg.norm(rotation_axis)
                
                if rotation_axis_norm > 1e-6:
                    rotation_axis = rotation_axis / rotation_axis_norm
                    rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                    
                    # Rodrigues旋转公式
                    K = np.array([
                        [0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0]
                    ])
                    R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
                else:
                    # 已经对齐或相反，使用单位矩阵或翻转
                    if np.dot(z_axis, direction) < 0:
                        R = np.diag([1, 1, -1])
                    else:
                        R = np.eye(3)
                
                # 应用旋转和平移
                cylinder.rotate(R, center=[0, 0, 0])
                cylinder.translate(kf_pos + direction * cylinder_height / 2)
                
                cone.rotate(R, center=[0, 0, 0])
                cone.translate(kf_pos + direction * (cylinder_height + cone_height / 2))
                
                geometries.append(cylinder)
                geometries.append(cone)
        
        # 设置可视化窗口
        vis = o3d.visualization.Visualizer()
        
        # 构建窗口标题（包含位置和差值信息）
        title = f"Tracker {self.tracker_id} - Frame {frame_idx + 1}/{len(self.frames)}"
        
        # 计算并显示位置信息
        if frame['detection_pos'] is not None and frame['kf_pos'] is not None:
            det_pos = np.array(frame['detection_pos'])
            kf_pos = np.array(frame['kf_pos'])
            diff = det_pos - kf_pos
            dist = np.linalg.norm(diff)
            title += f" | Err: {dist*1000:.1f}mm"
        
        vis.create_window(
            window_name=title,
            width=1280,
            height=720
        )
        
        # 添加几何体
        for geom in geometries:
            vis.add_geometry(geom)
        
        # 设置相机视角（从左后方向右前方看）
        view_control = vis.get_view_control()
        view_control.set_front([1, 1, 0])   # 相机朝向：向右前方
        view_control.set_up([0, 0, 1])      # 上方向：Z轴向上
        view_control.set_zoom(0.5)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
        
        # 添加文本信息（在终端详细显示）
        print(f"\n{'='*80}")
        print(f"帧 {frame_idx + 1}/{len(self.frames)} | 时间: {frame['relative_time']:.3f}s")
        print(f"{'-'*80}")
        
        if frame['detection_pos'] is not None:
            det_pos = np.array(frame['detection_pos'])
            print(f"检测位置 (Detection): [{det_pos[0]:7.4f}, {det_pos[1]:7.4f}, {det_pos[2]:7.4f}] m")
        else:
            print(f"检测位置 (Detection): None")
        
        if frame['kf_pos'] is not None:
            kf_pos = np.array(frame['kf_pos'])
            print(f"卡尔曼位置 (KF):     [{kf_pos[0]:7.4f}, {kf_pos[1]:7.4f}, {kf_pos[2]:7.4f}] m")
        else:
            print(f"卡尔曼位置 (KF):     None")
        
        if frame['detection_pos'] is not None and frame['kf_pos'] is not None:
            det_pos = np.array(frame['detection_pos'])
            kf_pos = np.array(frame['kf_pos'])
            diff = det_pos - kf_pos
            dist = np.linalg.norm(diff)
            print(f"位置差值 (Diff):     [{diff[0]:7.4f}, {diff[1]:7.4f}, {diff[2]:7.4f}] m")
            print(f"欧氏距离 (Distance): {dist*1000:.2f} mm")
        
        if frame['kf_vel'] is not None:
            kf_vel = np.array(frame['kf_vel'])
            vel_mag = np.linalg.norm(kf_vel)
            print(f"速度 (Velocity):     [{kf_vel[0]:7.4f}, {kf_vel[1]:7.4f}, {kf_vel[2]:7.4f}] m/s")
            print(f"速度大小 (Speed):    {vel_mag:.3f} m/s")
        
        print(f"{'='*80}\n")
        
        # 运行可视化（使用轮询方式，可响应Ctrl+C）
        try:
            while True:
                if not vis.poll_events():
                    break
                vis.update_renderer()
        except KeyboardInterrupt:
            print("\n检测到Ctrl+C，退出当前帧")
            raise
        finally:
            # 确保窗口被销毁
            try:
                vis.destroy_window()
            except:
                pass
    
    def visualize_all_interactive(self, start_frame=0, end_frame=None, trajectory_info=None):
        """
        交互式可视化所有帧（支持前后翻页）
        
        Args:
            start_frame: 起始帧索引
            end_frame: 结束帧索引（None表示到最后一帧）
            trajectory_info: 轨迹信息元组 (当前索引, 总数) 用于显示
        """
        if end_frame is None:
            end_frame = len(self.frames)
        
        print(f"\n开始交互式可视化 {end_frame - start_frame} 帧")
        print("控制按键：")
        print("  → / D / Space : 下一帧")
        print("  ← / A         : 上一帧")
        print("  ↑ / W         : 上一条轨迹")
        print("  ↓ / S         : 下一条轨迹")
        print("  Q / ESC       : 退出")
        print("  鼠标          : 旋转/缩放视角\n")
        
        current_frame = start_frame
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Trajectory Viewer", width=1280, height=720)
        
        # 用于控制帧切换的状态变量
        class FrameController:
            def __init__(self):
                self.current_frame = start_frame
                self.should_update = True
                self.should_exit = False
                self.switch_trajectory = 0  # -1: 上一条, 0: 无, 1: 下一条
        
        controller = FrameController()
        
        # 键盘回调函数
        def key_next(vis):
            if controller.current_frame < end_frame - 1:
                controller.current_frame += 1
                controller.should_update = True
            return False
        
        def key_prev(vis):
            if controller.current_frame > start_frame:
                controller.current_frame -= 1
                controller.should_update = True
            return False
        
        def key_next_trajectory(vis):
            controller.switch_trajectory = 1
            controller.should_exit = True
            return False
        
        def key_prev_trajectory(vis):
            controller.switch_trajectory = -1
            controller.should_exit = True
            return False
        
        def key_quit(vis):
            controller.should_exit = True
            return False
        
        # 注册键盘回调
        vis.register_key_callback(262, key_next)  # 右箭头
        vis.register_key_callback(ord('D'), key_next)
        vis.register_key_callback(32, key_next)   # 空格
        vis.register_key_callback(263, key_prev)  # 左箭头
        vis.register_key_callback(ord('A'), key_prev)
        vis.register_key_callback(265, key_prev_trajectory)  # 上箭头
        vis.register_key_callback(ord('W'), key_prev_trajectory)
        vis.register_key_callback(264, key_next_trajectory)  # 下箭头
        vis.register_key_callback(ord('S'), key_next_trajectory)
        vis.register_key_callback(ord('Q'), key_quit)
        vis.register_key_callback(256, key_quit)  # ESC
        
        # 初始化几何体列表
        geometries = []
        
        try:
            while not controller.should_exit:
                if controller.should_update:
                    # 清除旧的几何体
                    for geom in geometries:
                        vis.remove_geometry(geom, reset_bounding_box=False)
                    geometries.clear()
                    
                    # 加载当前帧
                    frame_idx = controller.current_frame
                    frame = self.frames[frame_idx]
                    
                    rgb_path = self.image_dir / frame['rgb_file']
                    depth_path = self.image_dir / frame['depth_file']
                    
                    if rgb_path.exists() and depth_path.exists():
                        rgb_image = cv2.imread(str(rgb_path))
                        depth_array = np.load(str(depth_path))
                        
                        # 创建点云
                        pcd = self.create_point_cloud_from_depth(rgb_image, depth_array)
                        pcd = pcd.voxel_down_sample(voxel_size=0.01)
                        geometries.append(pcd)
                        
                        # 添加坐标系
                        coord_frame = self.create_coordinate_frame(size=0.2)
                        geometries.append(coord_frame)
                        
                        # 添加检测位置标记（红色，实心，半径25mm）
                        if frame['detection_pos'] is not None:
                            detection_sphere = self.create_sphere_marker(
                                frame['detection_pos'], 
                                color=[1.0, 0.0, 0.0],
                                radius=0.025,  # 25mm
                                wireframe=False  # 实心
                            )
                            geometries.append(detection_sphere)
                        
                        # 添加卡尔曼滤波位置标记（蓝色，线框半透明，半径37.5mm）
                        if frame['kf_pos'] is not None:
                            kf_sphere = self.create_sphere_marker(
                                frame['kf_pos'], 
                                color=[0.0, 0.0, 1.0],
                                radius=0.0375,  # 37.5mm
                                wireframe=True  # 线框模式（透明效果）
                            )
                            geometries.append(kf_sphere)
                        
                        # 添加速度向量（绿色箭头）
                        if frame['kf_vel'] is not None and frame['kf_pos'] is not None:
                            kf_pos = np.array(frame['kf_pos'])
                            kf_vel = np.array(frame['kf_vel'])
                            vel_scale = 0.1
                            arrow_length = np.linalg.norm(kf_vel * vel_scale)
                            
                            if arrow_length > 0.001:
                                vel_end = kf_pos + kf_vel * vel_scale
                                direction = (vel_end - kf_pos) / arrow_length
                                
                                # 创建箭头
                                cylinder_height = arrow_length * 0.8
                                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                                    radius=0.005, height=cylinder_height)
                                cylinder.paint_uniform_color([0.0, 1.0, 0.0])
                                
                                cone_height = arrow_length * 0.2
                                cone = o3d.geometry.TriangleMesh.create_cone(
                                    radius=0.01, height=cone_height)
                                cone.paint_uniform_color([0.0, 1.0, 0.0])
                                
                                # 计算旋转
                                z_axis = np.array([0, 0, 1])
                                rotation_axis = np.cross(z_axis, direction)
                                rotation_axis_norm = np.linalg.norm(rotation_axis)
                                
                                if rotation_axis_norm > 1e-6:
                                    rotation_axis = rotation_axis / rotation_axis_norm
                                    rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                                    K = np.array([
                                        [0, -rotation_axis[2], rotation_axis[1]],
                                        [rotation_axis[2], 0, -rotation_axis[0]],
                                        [-rotation_axis[1], rotation_axis[0], 0]
                                    ])
                                    R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
                                else:
                                    R = np.diag([1, 1, -1]) if np.dot(z_axis, direction) < 0 else np.eye(3)
                                
                                cylinder.rotate(R, center=[0, 0, 0])
                                cylinder.translate(kf_pos + direction * cylinder_height / 2)
                                cone.rotate(R, center=[0, 0, 0])
                                cone.translate(kf_pos + direction * (cylinder_height + cone_height / 2))
                                
                                geometries.append(cylinder)
                                geometries.append(cone)
                        
                        # 添加所有几何体
                        for geom in geometries:
                            vis.add_geometry(geom, reset_bounding_box=(controller.current_frame == start_frame))
                        
                        # 设置视角（仅首帧）
                        if controller.current_frame == start_frame:
                            view_control = vis.get_view_control()
                            view_control.set_front([-300, 100, 50])
                            view_control.set_up([0, 0, 1])
                            view_control.set_zoom(0.4)
                        
                        # 设置渲染选项
                        render_option = vis.get_render_option()
                        render_option.point_size = 2.0
                        render_option.background_color = np.array([1.0, 1.0, 1.0])
                        
                        # 更新窗口标题
                        if trajectory_info is not None:
                            traj_idx, traj_total = trajectory_info
                            title = f"轨迹 {traj_idx}/{traj_total} | Tracker {self.tracker_id} - 帧 {frame_idx + 1}/{len(self.frames)}"
                        else:
                            title = f"Tracker {self.tracker_id} - 帧 {frame_idx + 1}/{len(self.frames)}"
                        if frame['detection_pos'] is not None and frame['kf_pos'] is not None:
                            det_pos = np.array(frame['detection_pos'])
                            kf_pos = np.array(frame['kf_pos'])
                            dist = np.linalg.norm(det_pos - kf_pos)
                            title += f" | Err: {dist*1000:.1f}mm"
                        
                        # 打印详细帧信息到终端
                        print(f"\n{'='*80}")
                        print(f"帧 {frame_idx + 1}/{len(self.frames)} | 时间: {frame['relative_time']:.3f}s")
                        
                        # 显示KF状态（upgrade或predict）
                        kf_state = frame.get('kf_state', 'unknown')
                        print(f"KF状态: {kf_state.upper()}")
                        print(f"{'-'*80}")
                        
                        if frame['detection_pos'] is not None:
                            det_pos = np.array(frame['detection_pos'])
                            print(f"检测位置 (Detection): [{det_pos[0]:7.4f}, {det_pos[1]:7.4f}, {det_pos[2]:7.4f}] m")
                        else:
                            print(f"检测位置 (Detection): None")
                        
                        if frame['kf_pos'] is not None:
                            kf_pos = np.array(frame['kf_pos'])
                            print(f"卡尔曼位置 (KF):      [{kf_pos[0]:7.4f}, {kf_pos[1]:7.4f}, {kf_pos[2]:7.4f}] m")
                        else:
                            print(f"卡尔曼位置 (KF):      None")
                        
                        if frame['detection_pos'] is not None and frame['kf_pos'] is not None:
                            det_pos = np.array(frame['detection_pos'])
                            kf_pos = np.array(frame['kf_pos'])
                            diff = det_pos - kf_pos
                            dist = np.linalg.norm(diff)
                            print(f"位置差值 (Diff):      [{diff[0]:7.4f}, {diff[1]:7.4f}, {diff[2]:7.4f}] m")
                            print(f"欧氏距离 (Distance):  {dist*1000:.2f} mm")
                        
                        if frame['kf_vel'] is not None:
                            kf_vel = np.array(frame['kf_vel'])
                            vel_mag = np.linalg.norm(kf_vel)
                            print(f"速度 (Velocity):      [{kf_vel[0]:7.4f}, {kf_vel[1]:7.4f}, {kf_vel[2]:7.4f}] m/s")
                            print(f"速度大小 (Speed):     {vel_mag:.3f} m/s")
                        
                        print(f"{'='*80}")
                    
                    controller.should_update = False
                
                # 更新可视化
                if not vis.poll_events():
                    break
                vis.update_renderer()
            
            print("\n\n可视化结束")
            return controller.switch_trajectory
        
        except KeyboardInterrupt:
            print("\n\n检测到Ctrl+C，退出可视化")
            vis.destroy_window()
            raise
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            vis.destroy_window()
            return 0
        finally:
            try:
                vis.destroy_window()
            except:
                pass
    
    def visualize_all(self, start_frame=0, end_frame=None, trajectory_info=None):
        """
        逐帧可视化所有帧（兼容旧接口，调用交互式版本）
        
        Args:
            start_frame: 起始帧索引
            end_frame: 结束帧索引（None表示到最后一帧）
            trajectory_info: 轨迹信息元组 (当前索引, 总数)
        """
        return self.visualize_all_interactive(start_frame, end_frame, trajectory_info)


def main():
    parser = argparse.ArgumentParser(
        description='使用Open3D可视化球体追踪轨迹',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 可视化所有轨迹文件（自动扫描trajectory_data目录）
  python3 visualize_trajectory_3d.py
  
  # 可视化指定轨迹文件
  python3 visualize_trajectory_3d.py trajectory_data/trajectory_tracker0_20260309_123456_0000.json
  
  # 从第10帧开始可视化
  python3 visualize_trajectory_3d.py trajectory_data/trajectory_tracker0_20260309_123456_0000.json --start 10
  
  # 只可视化前20帧
  python3 visualize_trajectory_3d.py trajectory_data/trajectory_tracker0_20260309_123456_0000.json --end 20
  
  # 可视化第10-20帧
  python3 visualize_trajectory_3d.py trajectory_data/trajectory_tracker0_20260309_123456_0000.json --start 10 --end 20
        """
    )
    
    parser.add_argument('trajectory_file', type=str, nargs='?', default=None,
                       help='轨迹JSON文件路径（不提供则自动扫描trajectory_data目录）')
    parser.add_argument('--start', type=int, default=0,
                       help='起始帧索引（默认0）')
    parser.add_argument('--end', type=int, default=None,
                       help='结束帧索引（默认到最后一帧）')
    parser.add_argument('--dir', type=str, default=None,
                       help='轨迹数据目录（默认为脚本所在目录的trajectory_data）')
    parser.add_argument('--cam-height', type=float, default=0.3,
                       help='相机相对于imu_link的高度偏移（米，默认0.3）')
    parser.add_argument('--cam-forward', type=float, default=0.0,
                       help='相机相对于imu_link的前向偏移（米，默认0.0）')
    parser.add_argument('--cam-lateral', type=float, default=0.0,
                       help='相机相对于imu_link的侧向偏移（米，默认0.0）')
    
    args = parser.parse_args()
    
    # 如果未指定文件，自动扫描目录
    if args.trajectory_file is None:
        # 默认目录为脚本所在目录的trajectory_data
        if args.dir is None:
            script_dir = Path(__file__).parent
            data_dir = script_dir / 'trajectory_data'
        else:
            data_dir = Path(args.dir)
        if not data_dir.exists():
            print(f"错误: 目录不存在: {data_dir}")
            return
        
        # 查找所有轨迹JSON文件
        json_files = sorted(data_dir.glob('trajectory_*.json'))
        if len(json_files) == 0:
            print(f"错误: 在 {data_dir} 中未找到任何轨迹文件")
            return
        
        print(f"\n找到 {len(json_files)} 个轨迹文件")
        print("使用↑↓或W/S切换轨迹，使用←→或A/D切换帧，按Q或ESC退出\n")
        
        current_trajectory = 0
        
        try:
            while 0 <= current_trajectory < len(json_files):
                json_file = json_files[current_trajectory]
                print(f"\n{'='*60}")
                print(f"轨迹 {current_trajectory+1}/{len(json_files)}: {json_file.name}")
                print('='*60)
                
                try:
                    visualizer = TrajectoryVisualizer(json_file)
                    # 如果提供了命令行参数，更新相机变换
                    if args.cam_height != 0.3 or args.cam_forward != 0.0 or args.cam_lateral != 0.0:
                        print(f"警告: 使用自定义相机参数会覆盖实际外参")
                        camera_rotation_in_body = visualizer.camera_to_body_transform[:3, :3]
                        visualizer.camera_to_body_transform = np.eye(4)
                        visualizer.camera_to_body_transform[:3, :3] = camera_rotation_in_body
                        visualizer.camera_to_body_transform[:3, 3] = [args.cam_forward, args.cam_lateral, args.cam_height]
                    
                    switch = visualizer.visualize_all(
                        start_frame=args.start, 
                        end_frame=args.end,
                        trajectory_info=(current_trajectory+1, len(json_files))
                    )
                    
                    # 根据返回值决定下一步
                    if switch == 1:  # 下一条轨迹
                        current_trajectory += 1
                    elif switch == -1:  # 上一条轨迹
                        current_trajectory -= 1
                    else:  # 退出
                        break
                        
                except KeyboardInterrupt:
                    print("\n检测到Ctrl+C，停止可视化")
                    break
            
            print(f"\n可视化完成")
        except KeyboardInterrupt:
            print("\n\n程序已中断")
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 可视化指定文件
        trajectory_path = Path(args.trajectory_file)
        if not trajectory_path.exists():
            print(f"错误: 文件不存在: {trajectory_path}")
            return
        
        visualizer = TrajectoryVisualizer(trajectory_path)
        # 如果提供了命令行参数，更新相机变换
        if args.cam_height != 0.3 or args.cam_forward != 0.0 or args.cam_lateral != 0.0:
            print(f"警告: 使用自定义相机参数会覆盖实际外参")
            camera_rotation_in_body = visualizer.camera_to_body_transform[:3, :3]
            visualizer.camera_to_body_transform = np.eye(4)
            visualizer.camera_to_body_transform[:3, :3] = camera_rotation_in_body
            visualizer.camera_to_body_transform[:3, 3] = [args.cam_forward, args.cam_lateral, args.cam_height]
        visualizer.visualize_all(start_frame=args.start, end_frame=args.end)


if __name__ == '__main__':
    main()
