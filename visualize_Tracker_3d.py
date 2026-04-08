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

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if plt is not None:
    # 避免部分环境下负号显示异常
    plt.rcParams['axes.unicode_minus'] = False


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
        self.data_coord_frame = str(self.trajectory.get('coord_frame', 'body')).lower()
        if self.data_coord_frame not in ('body', 'world'):
            print(f"警告: 未知coord_frame={self.data_coord_frame}，回退为body")
            self.data_coord_frame = 'body'

        # 当前可视化/处理坐标系（默认跟随轨迹数据坐标系）
        self.coord_frame = self.data_coord_frame
        
        print(f"Loaded trajectory for tracker {self.tracker_id}")
        print(f"Total frames: {len(self.frames)}")
        print(f"Image directory: {self.image_dir}")
        print(f"Trajectory coordinate frame: {self.data_coord_frame}")
        
        # 加载相机配置
        config_path = Path(__file__).parent / 'Tracker_config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"相机配置文件不存在: {config_path}")
        
        with open(config_path, 'r') as f:
            camera_config = yaml.safe_load(f)
        print(f"Loaded camera config from: {config_path}")
      
        self.visualize_icp = camera_config['tracker'].get('visualize_icp', False)
        
        print(f"Visualize ICP: {self.visualize_icp}")
        
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

        
        use_robot_data = camera_config.get('use_robot_data', False)
        # 相机外参
        if use_robot_data:
            extr = camera_config['extrinsics']
            camera_position_in_body = np.array(extr['position'])
            camera_rotation_in_body = np.array(extr['rotation'])
            
        else:
            # 使用已知标定外参（相机坐标系 -> body/imu_link 坐标系）
            camera_position_in_body = np.array([-0.010, 0.060, 0.015], dtype=np.float64)
            camera_rotation_in_body = np.array([
                [0.0, -0.050, 0.999 ],
                [-1.0, 0.0  , 0.0   ],
                [0.0, -0.999, -0.050],
            ], dtype=np.float64)
            
       
            
        # 构建从相机坐标系到body坐标系的4x4变换矩阵
        self.camera_to_body_transform = np.eye(4)
        self.camera_to_body_transform[:3, :3] = camera_rotation_in_body
        self.camera_to_body_transform[:3, 3] = camera_position_in_body

        # 点云过滤开关：True时剔除 body_x > self.x_filter_threshold 的点
        self.enable_x_filter = False
        self.x_filter_threshold = 1.0

        # 球体显示开关
        self.show_detection_sphere = True
        self.show_kf_sphere = True

        # Ground truth（由点云拟合）显示开关与球半径
        self.show_ground_truth_marker = True
        self.ball_radius = 0.0375
        # GT有效点云范围：仅使用 x <= 0.7m 的点
        self.gt_max_x = 0.7

        # 轨迹线显示开关（交互3D）
        self.show_detection_trajectory = True
        self.show_ground_truth_trajectory = True

        # 缓存每帧的GT估计结果，避免重复计算
        self.gt_result_cache = {}
    
    def _get_body_pose_from_frame(self, frame):
        """从帧数据提取body位姿（世界坐标系下）。"""
        body_pos = frame.get('body_pos', None)
        body_rot = frame.get('body_rot', None)
        if body_pos is None or body_rot is None:
            return None, None

        body_pos = np.array(body_pos, dtype=np.float64).reshape(3)
        body_rot = np.array(body_rot, dtype=np.float64).reshape(3, 3)
        return body_pos, body_rot

    def _get_body_to_world_transform(self, frame):
        """构造 body->world 4x4 变换矩阵。"""
        body_pos, body_rot = self._get_body_pose_from_frame(frame)
        if body_pos is None or body_rot is None:
            return None

        T = np.eye(4)
        T[:3, :3] = body_rot
        T[:3, 3] = body_pos
        return T

    def _get_world_to_body_transform(self, frame):
        """构造 world->body 4x4 变换矩阵。"""
        body_pos, body_rot = self._get_body_pose_from_frame(frame)
        if body_pos is None or body_rot is None:
            return None

        T = np.eye(4)
        T[:3, :3] = body_rot.T
        T[:3, 3] = -body_rot.T @ body_pos
        return T

    def _get_camera_position_in_current_frame(self, frame=None):
        """获取当前轨迹坐标系下的相机位置。"""
        cam_pos_body = self.camera_to_body_transform[:3, 3].astype(np.float64)

        if self.coord_frame == 'world' and frame is not None:
            body_to_world = self._get_body_to_world_transform(frame)
            if body_to_world is not None:
                return body_to_world[:3, :3] @ cam_pos_body + body_to_world[:3, 3]

        return cam_pos_body

    def _convert_points_between_frames(self, points, source_frame, target_frame, frame=None):
        """
        在body/world之间转换点坐标。

        Args:
            points: Nx3点数组或(3,)单点
            source_frame: 'body' 或 'world'
            target_frame: 'body' 或 'world'
            frame: 当前帧（用于读取body位姿）

        Returns:
            转换后的点数组（保持输入形状），若无法转换返回None
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.size == 0:
            return pts.copy()

        reshape_back = False
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
            reshape_back = True

        source = str(source_frame).lower()
        target = str(target_frame).lower()

        if source == target:
            out = pts.copy()
            return out.reshape(3) if reshape_back else out

        if source == 'body' and target == 'world':
            T = self._get_body_to_world_transform(frame)
        elif source == 'world' and target == 'body':
            T = self._get_world_to_body_transform(frame)
        else:
            return None

        if T is None:
            return None

        R = T[:3, :3]
        t = T[:3, 3]
        out = (R @ pts.T).T + t
        return out.reshape(3) if reshape_back else out

    def _convert_point_between_frames(self, point, source_frame, target_frame, frame=None):
        """单点版本坐标系转换。"""
        return self._convert_points_between_frames(point, source_frame, target_frame, frame=frame)

    def _build_gt_body_x_mask(self, points_current, frame=None):
        """
        构造 GT 过滤掩码：始终按 body 坐标系的 x <= self.gt_max_x 过滤。

        Args:
            points_current: 当前显示坐标系下的点云 Nx3
            frame: 当前帧（当当前坐标系为 world 时需要用于 world->body）

        Returns:
            mask: bool[N]
        """
        pts = np.asarray(points_current, dtype=np.float64)
        if pts.size == 0:
            return np.zeros((0,), dtype=bool)

        # 若当前点云在 world 坐标系，则先转回 body 再按 x 过滤
        if self.coord_frame == 'world':
            world_to_body = self._get_world_to_body_transform(frame) if frame is not None else None
            if world_to_body is not None:
                R = world_to_body[:3, :3]
                t = world_to_body[:3, 3]
                pts_body = (R @ pts.T).T + t
                return pts_body[:, 0] <= self.gt_max_x

        # body 坐标系（或无法转换时）直接按当前 x 过滤
        return pts[:, 0] <= self.gt_max_x

    def _refine_center_for_tangent(self, points, center, radius, camera_pos, max_iters=4):
        """
        后处理：将球心沿“远离相机”的方向轻微平移，使点云尽可能与球面相切（减少点云落在球内）。
        """
        c = np.array(center, dtype=np.float64).reshape(3)
        pts = np.asarray(points, dtype=np.float64)
        cam = np.array(camera_pos, dtype=np.float64).reshape(3)

        for _ in range(max_iters):
            signed = np.linalg.norm(pts - c, axis=1) - radius
            med = float(np.median(signed))

            # 中位残差已接近相切则停止
            if med >= -5e-4:
                break

            push_dir = c - cam
            n = np.linalg.norm(push_dir)
            if n < 1e-8:
                push_dir = c - np.median(pts, axis=0)
                n = np.linalg.norm(push_dir)
                if n < 1e-8:
                    break
            push_dir = push_dir / n

            # 负残差越大，后推越多；单次限制防止过冲
            step = float(np.clip(-med, 0.0, 0.008))
            c = c + push_dir * step

        return c

    def create_point_cloud_from_depth(self, rgb_image, depth_array, frame=None):
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
        
        # 先将点云从相机坐标系转换到body坐标系（imu_link）
        pcd.transform(self.camera_to_body_transform)

        # 若轨迹保存为world坐标系，则进一步变换到world
        if self.coord_frame == 'world' and frame is not None:
            body_to_world = self._get_body_to_world_transform(frame)
            if body_to_world is not None:
                pcd.transform(body_to_world)
        
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

    def create_world_body_coordinate_frames(self, frame, size=0.2):
        """
        同时创建world和body坐标系，并转换到当前显示坐标系。
        """
        geoms = []

        world_cf = self.create_coordinate_frame(size=size * 1.8)
        body_cf = self.create_coordinate_frame(size=size * 0.85)

        if self.coord_frame == 'world':
            # world显示下：world原点不动，body按位姿放置
            body_to_world = self._get_body_to_world_transform(frame)
            if body_to_world is not None:
                body_cf.transform(body_to_world)
            geoms.extend([world_cf, body_cf])
        else:
            # body显示下：body在原点，world按逆位姿放置
            world_to_body = self._get_world_to_body_transform(frame)
            if world_to_body is not None:
                world_cf.transform(world_to_body)
            geoms.extend([world_cf, body_cf])

        return geoms

    def filter_point_cloud_by_x(self, pcd, frame=None):
        """
        根据x阈值过滤点云（始终按 body 坐标系过滤，保留 body_x <= 阈值 的点）

        Args:
            pcd: Open3D点云对象
            frame: 当前帧（当当前坐标系为world时用于 world->body 变换）

        Returns:
            过滤后的Open3D点云对象
        """
        if not self.enable_x_filter:
            return pcd

        points = np.asarray(pcd.points)
        if points.size == 0:
            return pcd

        # 始终按 body 坐标系 x 做过滤
        if self.coord_frame == 'world':
            world_to_body = self._get_world_to_body_transform(frame) if frame is not None else None
            if world_to_body is not None:
                R = world_to_body[:3, :3]
                t = world_to_body[:3, 3]
                points_body = (R @ points.T).T + t
                mask = points_body[:, 0] <= self.x_filter_threshold
            else:
                # 无法转换时兜底：按当前点云x过滤
                mask = points[:, 0] <= self.x_filter_threshold
        else:
            mask = points[:, 0] <= self.x_filter_threshold

        return pcd.select_by_index(np.where(mask)[0])

    
    def estimate_sphere_center_icp(self, points, radius, initial_center=None):
        """
        使用ICP将“已知半径球面模型”对齐到点云，直接估计球心。

        Args:
            points: Nx3点数组（已完成基础离群值处理）
            radius: 球半径（米）
            initial_center: 初始球心（可选）

        Returns:
            (center, inlier_mask, rmse, fitness, inlier_rmse)
        """
        n = points.shape[0]
        if n < 12:
            return None, None, None, 0.0, None

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # 构建单位姿态下的球面模型（中心在原点）
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=40)
        sample_n = int(np.clip(n * 4, 400, 3000))
        source_pcd = sphere_mesh.sample_points_poisson_disk(number_of_points=sample_n)

        if initial_center is None:
            # 中位数比均值更抗离群值
            initial_center = np.median(points, axis=0)

        T_init = np.eye(4)
        T_init[:3, 3] = np.asarray(initial_center, dtype=np.float64).reshape(3)

        # 多阶段ICP：粗配准 -> 细配准 -> 超细配准（提高迭代次数）
        reg_coarse = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            0.04,
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=120)
        )
        reg_fine = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            0.012,
            reg_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=240)
        )
        reg_ultra = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            0.005,
            reg_fine.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=360)
        )
        
        reg_ultra = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            0.001,
            reg_fine.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=720)
        )

        center = reg_ultra.transformation[:3, 3].copy()
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)

        # 更严格的内点门限；不足时放宽兜底
        inlier_mask = residual < 0.008
        if np.count_nonzero(inlier_mask) < 8:
            inlier_mask = residual < 0.012

        if np.count_nonzero(inlier_mask) == 0:
            rmse = float(np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius) ** 2)))
        else:
            rmse = float(np.sqrt(np.mean((np.linalg.norm(points[inlier_mask] - center, axis=1) - radius) ** 2)))

        return center, inlier_mask, rmse, float(reg_ultra.fitness), float(reg_ultra.inlier_rmse)

    def estimate_ground_truth_from_point_cloud(self, pcd, frame=None):
        """
        从点云估计球心作为ground truth

        注意：不能直接用点云质心，因为相机只看到球的一侧。
        这里使用“已知半径球拟合 + 离群点去除”。

        Returns:
            dict: {
                'position': np.ndarray(3,) or None,
                'rmse': float or None,
                'inlier_points': int,
                'cluster_points': int,
                'total_points': int
            }
        """
        prior = None
        if frame is not None:
            # 约束：有GT必然要有Detection
            if frame.get('detection_pos') is None:
                return {
                    'position': None,
                    'rmse': None,
                    'inlier_points': 0,
                    'cluster_points': 0,
                    'total_points': 0
                }
            prior = frame['detection_pos']

        # GT仅在 x<=1.0m 有点云时存在（与显示过滤开关无关）
        points_all = np.asarray(pcd.points)
        if points_all.shape[0] == 0:
            return {
                'position': None,
                'rmse': None,
                'inlier_points': 0,
                'cluster_points': 0,
                'total_points': 0
            }

        gt_mask = self._build_gt_body_x_mask(points_all, frame=frame)
        gt_indices = np.where(gt_mask)[0]
        if gt_indices.size < 8:
            return {
                'position': None,
                'rmse': None,
                'inlier_points': 0,
                'cluster_points': int(gt_indices.size),
                'total_points': int(points_all.shape[0])
            }

        pcd_for_gt = pcd.select_by_index(gt_indices)

        # 先对 x<=gt_max_x 的剩余点云做一轮离群值剔除，再进行球簇提取
        # （避免远端飞点/孤立点影响后续DBSCAN与ICP初始化）
        if len(pcd_for_gt.points) >= 12:
            nb_neighbors = int(min(20, max(6, len(pcd_for_gt.points) - 1)))
            _, inlier_idx = pcd_for_gt.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=1.0
            )
            if len(inlier_idx) >= 8:
                pcd_for_gt = pcd_for_gt.select_by_index(inlier_idx)

        if len(pcd_for_gt.points) >= 12:
            _, inlier_idx = pcd_for_gt.remove_radius_outlier(nb_points=4, radius=0.03)
            if len(inlier_idx) >= 8:
                pcd_for_gt = pcd_for_gt.select_by_index(inlier_idx)

        # 这里使用“去完离群值后的中心点”作为先验，而不是原始detection_pos
        prefiltered_points = np.asarray(pcd_for_gt.points)
        if prefiltered_points.shape[0] > 0:
            prior_after_outlier = np.median(prefiltered_points, axis=0)
        elif prior is not None:
            prior_after_outlier = np.array(prior, dtype=np.float64)
        else:
            prior_after_outlier = None

        init_center = np.array(prior_after_outlier, dtype=np.float64) if prior_after_outlier is not None else np.median(prefiltered_points, axis=0)
        center, inlier_mask, rmse, fitness, icp_inlier_rmse = self.estimate_sphere_center_icp(
            prefiltered_points,
            self.ball_radius,
            initial_center=init_center
        )

        # 期望“尽可能相切”而非点云落在球内部：按相机方向做轻量后推修正
        if center is not None:
            cam_pos = self._get_camera_position_in_current_frame(frame)
            center = self._refine_center_for_tangent(
                prefiltered_points,
                center,
                self.ball_radius,
                cam_pos,
                max_iters=4
            )

            residual = np.abs(np.linalg.norm(prefiltered_points - center, axis=1) - self.ball_radius)
            inlier_mask = residual < 0.001
            if np.count_nonzero(inlier_mask) < 8:
                inlier_mask = residual < 0.008

            if np.count_nonzero(inlier_mask) == 0:
                rmse = float(np.sqrt(np.mean((np.linalg.norm(prefiltered_points - center, axis=1) - self.ball_radius) ** 2)))
            else:
                rmse = float(np.sqrt(np.mean((np.linalg.norm(prefiltered_points[inlier_mask] - center, axis=1) - self.ball_radius) ** 2)))
        
        # 按需求：去完离群值后，先用Open3D可视化一次
        if self.visualize_icp :
            try:
                preview_geoms = [pcd_for_gt.voxel_down_sample(voxel_size=0.005)]
                if center is not None:
                    fitted_sphere = self.create_sphere_marker(
                        center,
                        color=[0.0, 0.7, 0.0],
                        radius=self.ball_radius,
                        wireframe=True
                    )
                    preview_geoms.append(fitted_sphere)
                if frame is not None:
                    preview_geoms.extend(self.create_world_body_coordinate_frames(frame, size=0.15))
                else:
                    preview_geoms.append(self.create_coordinate_frame(size=0.15))

                o3d.visualization.draw_geometries(
                    preview_geoms,
                    window_name=f"GT Prefilter Preview | Frame:{self.coord_frame}",
                    width=1024,
                    height=720
                )
            except Exception as e:
                print(f"警告: GT预过滤点云可视化失败: {e}")
      
                   
        if center is None:
            return {
                'position': None,
                'rmse': None,
                'inlier_points': 0,
                'cluster_points': int(prefiltered_points.shape[0]),
                'total_points': int(prefiltered_points.shape[0])
            }

        return {
            'position': center,
            'rmse': rmse,
            'inlier_points': int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else int(prefiltered_points.shape[0]),
            'cluster_points': int(prefiltered_points.shape[0]),
            'total_points': int(prefiltered_points.shape[0]),
            'icp_fitness': fitness,
            'icp_inlier_rmse': icp_inlier_rmse
        }

    def create_trajectory_lineset(self, points, color):
        """
        根据轨迹点创建LineSet。

        Args:
            points: 轨迹点列表 [[x,y,z], ...]
            color: 线颜色 [r,g,b]

        Returns:
            Open3D LineSet或None
        """
        if points is None or len(points) < 2:
            return None

        pts = np.asarray(points, dtype=np.float64)
        lines = np.array([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)
        colors = np.tile(np.array(color, dtype=np.float64), (len(lines), 1))

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def _visualize_gt_preview(self, pcd, center, frame=None):
        """可视化GT预览：预处理点云 + 拟合球 + 坐标系。"""
        try:
            preview_geoms = [pcd.voxel_down_sample(voxel_size=0.005)]
            if center is not None:
                fitted_sphere = self.create_sphere_marker(
                    center,
                    color=[0.0, 0.7, 0.0],
                    radius=self.ball_radius,
                    wireframe=True
                )
                preview_geoms.append(fitted_sphere)
            if frame is not None:
                preview_geoms.extend(self.create_world_body_coordinate_frames(frame, size=0.15))
            else:
                preview_geoms.append(self.create_coordinate_frame(size=0.15))

            o3d.visualization.draw_geometries(
                preview_geoms,
                window_name=f"GT Prefilter Preview | Frame:{self.coord_frame}",
                width=1024,
                height=720
            )
        except Exception as e:
            print(f"警告: GT预过滤点云可视化失败: {e}")

    def get_or_compute_gt_result(self, frame_idx, frame=None, pcd=None):
        """
        获取或计算指定帧的GT拟合结果

        Args:
            frame_idx: 帧索引
            frame: 帧字典（可选）
            pcd: 已构建好的点云（可选）

        Returns:
            GT结果字典
        """
        if frame_idx in self.gt_result_cache:
            return self.gt_result_cache[frame_idx]

        if frame is None:
            frame = self.frames[frame_idx]

        # 若轨迹中已存在gt_pos，直接复用，避免重复ICP
        gt_pos_existing = frame.get('gt_pos', None)
        if gt_pos_existing is not None:
            gt_pos_arr = np.asarray(gt_pos_existing, dtype=np.float64)
            result = {
                'position': gt_pos_arr,
                'rmse': float(frame['gt_pos_rmse']) if frame.get('gt_pos_rmse', None) is not None else None,
                'inlier_points': 0,
                'cluster_points': 0,
                'total_points': 0,
                'icp_fitness': None,
                'icp_inlier_rmse': None
            }

            # 开启预览时，可直接用已有gt_pos做可视化（无需重复ICP）
            if self.visualize_icp:
                try:
                    if pcd is None:
                        rgb_path = self.image_dir / frame['rgb_file']
                        depth_path = self.image_dir / frame['depth_file']
                        if rgb_path.exists() and depth_path.exists():
                            rgb_image = cv2.imread(str(rgb_path))
                            depth_array = np.load(str(depth_path))
                            pcd = self.create_point_cloud_from_depth(rgb_image, depth_array, frame=frame)
                    if pcd is not None:
                        points_all = np.asarray(pcd.points)
                        if points_all.shape[0] > 0:
                            gt_mask = self._build_gt_body_x_mask(points_all, frame=frame)
                            gt_indices = np.where(gt_mask)[0]
                            pcd_for_gt = pcd.select_by_index(gt_indices) if gt_indices.size > 0 else pcd
                        else:
                            pcd_for_gt = pcd
                        self._visualize_gt_preview(pcd_for_gt, gt_pos_arr, frame=frame)
                except Exception as e:
                    print(f"警告: 复用gt_pos可视化失败: {e}")

            self.gt_result_cache[frame_idx] = result
            return result

        result = {
            'position': None,
            'rmse': None,
            'inlier_points': 0,
            'cluster_points': 0,
            'total_points': 0
        }

        try:
            if pcd is None:
                rgb_path = self.image_dir / frame['rgb_file']
                depth_path = self.image_dir / frame['depth_file']
                if not rgb_path.exists() or not depth_path.exists():
                    self.gt_result_cache[frame_idx] = result
                    return result

                rgb_image = cv2.imread(str(rgb_path))
                depth_array = np.load(str(depth_path))
                pcd = self.create_point_cloud_from_depth(rgb_image, depth_array, frame=frame)

            result = self.estimate_ground_truth_from_point_cloud(pcd, frame=frame)
        except Exception as e:
            print(f"警告: 帧 {frame_idx} GT估计失败: {e}")

        self.gt_result_cache[frame_idx] = result
        return result

    def add_gt_pos_and_save(self, start_frame=0, end_frame=None, create_backup=True):
        """
        用拟合得到的GT写入轨迹文档中的 `gt_pos` 字段（不覆盖 detection_pos），并写回JSON文件。

        Args:
            start_frame: 起始帧索引
            end_frame: 结束帧索引（None表示到最后一帧）
            create_backup: 是否先创建备份文件

        Returns:
            dict: 新增统计
        """
        if end_frame is None:
            end_frame = len(self.frames)

        start_frame = max(0, int(start_frame))
        end_frame = min(len(self.frames), int(end_frame))

        if end_frame <= start_frame:
            return {
                'updated': 0,
                'skipped_no_gt': 0,
                'total': 0,
                'backup_path': None,
                'output_path': str(self.json_path)
            }

        backup_path = None
        if create_backup:
            backup_path = self.json_path.with_suffix(self.json_path.suffix + '.bak')
            if not backup_path.exists():
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(self.trajectory, f, ensure_ascii=False, indent=2)

        updated = 0
        skipped_no_gt = 0

        for frame_idx in range(start_frame, end_frame):
            frame = self.frames[frame_idx]
            gt_result = self.get_or_compute_gt_result(frame_idx, frame=frame)
            gt_pos = gt_result.get('position', None)

            if gt_pos is None:
                skipped_no_gt += 1
                continue

            gt_pos_data_frame = self._convert_point_between_frames(
                gt_pos,
                source_frame=self.coord_frame,
                target_frame=self.data_coord_frame,
                frame=frame
            )
            if gt_pos_data_frame is None:
                skipped_no_gt += 1
                continue

            gt_list = np.asarray(gt_pos_data_frame, dtype=np.float64).tolist()

            frame['gt_pos'] = gt_list
            frame['gt_pos_source'] = 'gt_fit_icp'
            frame['gt_pos_frame'] = self.data_coord_frame
            if gt_result.get('rmse') is not None:
                frame['gt_pos_rmse'] = float(gt_result['rmse'])
            updated += 1

        # 写回文档
        self.trajectory['frames'] = self.frames
        self.trajectory.setdefault('postprocess', {})
        self.trajectory['postprocess']['gt_pos_added'] = True
        self.trajectory['postprocess']['gt_pos_added_range'] = [start_frame, end_frame]

        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.trajectory, f, ensure_ascii=False, indent=2)

        return {
            'updated': updated,
            'skipped_no_gt': skipped_no_gt,
            'total': end_frame - start_frame,
            'backup_path': str(backup_path) if backup_path is not None else None,
            'output_path': str(self.json_path)
        }

    @staticmethod
    def _calc_error_stats(values):
        """计算误差统计量。"""
        return {
            'max': float(np.max(values)),
            'min': float(np.min(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'abs_mean': float(np.mean(np.abs(values))),
            'abs_std': float(np.std(np.abs(values)))
        }

    def compute_detection_gt_error_statistics(self, start_frame=0, end_frame=None):
        """
        计算Detection与GT差值（det - gt）的时间序列和统计量。

        Returns:
            dict or None
        """
        if end_frame is None:
            end_frame = len(self.frames)

        times = []
        errors = []

        for frame_idx in range(start_frame, end_frame):
            frame = self.frames[frame_idx]
            det = frame.get('detection_pos', None)
            if det is None:
                continue

            gt_result = self.get_or_compute_gt_result(frame_idx, frame=frame)
            if gt_result.get('position') is None:
                continue

            det_pos = np.array(det, dtype=np.float64)
            gt_pos = np.array(gt_result['position'], dtype=np.float64)
            err = det_pos - gt_pos

            times.append(float(frame.get('relative_time', frame_idx)))
            errors.append(err)

        if len(errors) == 0:
            return None

        t = np.array(times, dtype=np.float64)
        e = np.array(errors, dtype=np.float64)
        stats_x = self._calc_error_stats(e[:, 0])
        stats_y = self._calc_error_stats(e[:, 1])
        stats_z = self._calc_error_stats(e[:, 2])

        return {
            't': t,
            'e': e,
            'stats_list': [stats_x, stats_y, stats_z],
            'valid_count': len(e),
            'total_count': end_frame - start_frame
        }

    def print_detection_gt_error_statistics(self, start_frame=0, end_frame=None):
        """打印当前轨迹的Detection-GT差值统计信息。"""
        result = self.compute_detection_gt_error_statistics(start_frame=start_frame, end_frame=end_frame)
        print(f"\n[Detection-GT差值统计] Trajectory: {self.json_path.name} (det - gt), 单位: m")
        if result is None:
            print("  无可用的Detection-GT配对数据")
            return

        for axis_name, s in zip(['X', 'Y', 'Z'], result['stats_list']):
            print(
                f"  {axis_name}: max={s['max']:.6f}, min={s['min']:.6f}, mean={s['mean']:.6f}, std={s['std']:.6f}, "
                f"abs_mean={s['abs_mean']:.6f}, abs_std={s['abs_std']:.6f}"
            )
        print(f"  有效帧: {result['valid_count']}/{result['total_count']}")

    def compute_kf_gt_error_statistics(self, start_frame=0, end_frame=None):
        """
        计算KF与GT差值（kf - gt）的时间序列和统计量。

        Returns:
            dict or None
        """
        if end_frame is None:
            end_frame = len(self.frames)

        times = []
        errors = []

        for frame_idx in range(start_frame, end_frame):
            frame = self.frames[frame_idx]
            kf = frame.get('kf_pos', None)
            if kf is None:
                continue

            gt_result = self.get_or_compute_gt_result(frame_idx, frame=frame)
            if gt_result.get('position') is None:
                continue

            kf_pos = np.array(kf, dtype=np.float64)
            gt_pos = np.array(gt_result['position'], dtype=np.float64)
            err = kf_pos - gt_pos

            times.append(float(frame.get('relative_time', frame_idx)))
            errors.append(err)

        if len(errors) == 0:
            return None

        t = np.array(times, dtype=np.float64)
        e = np.array(errors, dtype=np.float64)
        stats_x = self._calc_error_stats(e[:, 0])
        stats_y = self._calc_error_stats(e[:, 1])
        stats_z = self._calc_error_stats(e[:, 2])

        return {
            't': t,
            'e': e,
            'stats_list': [stats_x, stats_y, stats_z],
            'valid_count': len(e),
            'total_count': end_frame - start_frame
        }

    def print_kf_gt_error_statistics(self, start_frame=0, end_frame=None):
        """打印当前轨迹的KF-GT差值统计信息。"""
        result = self.compute_kf_gt_error_statistics(start_frame=start_frame, end_frame=end_frame)
        print(f"\n[KF-GT差值统计] Trajectory: {self.json_path.name} (kf - gt), 单位: m")
        if result is None:
            print("  无可用的KF-GT配对数据")
            return

        for axis_name, s in zip(['X', 'Y', 'Z'], result['stats_list']):
            print(
                f"  {axis_name}: max={s['max']:.6f}, min={s['min']:.6f}, mean={s['mean']:.6f}, std={s['std']:.6f}, "
                f"abs_mean={s['abs_mean']:.6f}, abs_std={s['abs_std']:.6f}"
            )
        print(f"  有效帧: {result['valid_count']}/{result['total_count']}")

    def show_detection_gt_error_plots(self, start_frame=0, end_frame=None):
        """
        绘制Detection与GT差值（det - gt）在XYZ方向随时间变化图，并展示统计信息。
        """
        if plt is None:
            print("警告: 未安装matplotlib，无法显示误差统计图")
            return

        result = self.compute_detection_gt_error_statistics(start_frame=start_frame, end_frame=end_frame)
        if result is None:
            print("\n[统计] 当前轨迹无可用的Detection-GT配对数据，跳过绘图")
            return

        t = result['t']
        e = result['e']
        stats_list = result['stats_list']

        self.print_detection_gt_error_statistics(start_frame=start_frame, end_frame=end_frame)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axis_names = ['X', 'Y', 'Z']
        colors = ['tab:red', 'tab:green', 'tab:blue']

        for i, ax in enumerate(axes):
            # 曲线使用mm展示，更直观
            y_mm = e[:, i] * 1000.0
            s = stats_list[i]

            ax.plot(t, y_mm, color=colors[i], linewidth=1.5)
            ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.set_ylabel(f"{axis_names[i]} Error (mm)")
            ax.grid(True, linestyle='--', alpha=0.35)

            stat_text = (
                f"max: {s['max']*1000:.2f} mm\n"
                f"min: {s['min']*1000:.2f} mm\n"
                f"mean: {s['mean']*1000:.2f} mm\n"
                f"std: {s['std']*1000:.2f} mm\n"
                f"|.| mean: {s['abs_mean']*1000:.2f} mm\n"
                f"|.| std: {s['abs_std']*1000:.2f} mm"
            )
            ax.text(
                1.02,
                0.5,
                stat_text,
                transform=ax.transAxes,
                va='center',
                ha='left',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
            )

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"Trajectory {self.json_path.name} | Tracker {self.tracker_id}\n"
            f"Detection - GT Error (Valid frames: {result['valid_count']}/{result['total_count']})",
            fontsize=12
        )
        plt.subplots_adjust(right=0.78, hspace=0.28)
        plt.show()

    def show_kf_gt_error_plots(self, start_frame=0, end_frame=None):
        """
        绘制KF与GT差值（kf - gt）在XYZ方向随时间变化图，并展示统计信息。
        """
        if plt is None:
            print("警告: 未安装matplotlib，无法显示误差统计图")
            return

        result = self.compute_kf_gt_error_statistics(start_frame=start_frame, end_frame=end_frame)
        if result is None:
            print("\n[统计] 当前轨迹无可用的KF-GT配对数据，跳过绘图")
            return

        t = result['t']
        e = result['e']
        stats_list = result['stats_list']

        self.print_kf_gt_error_statistics(start_frame=start_frame, end_frame=end_frame)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axis_names = ['X', 'Y', 'Z']
        colors = ['tab:red', 'tab:green', 'tab:blue']

        for i, ax in enumerate(axes):
            y_mm = e[:, i] * 1000.0
            s = stats_list[i]

            ax.plot(t, y_mm, color=colors[i], linewidth=1.5)
            ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.set_ylabel(f"{axis_names[i]} Error (mm)")
            ax.grid(True, linestyle='--', alpha=0.35)

            stat_text = (
                f"max: {s['max']*1000:.2f} mm\n"
                f"min: {s['min']*1000:.2f} mm\n"
                f"mean: {s['mean']*1000:.2f} mm\n"
                f"std: {s['std']*1000:.2f} mm\n"
                f"|.| mean: {s['abs_mean']*1000:.2f} mm\n"
                f"|.| std: {s['abs_std']*1000:.2f} mm"
            )
            ax.text(
                1.02,
                0.5,
                stat_text,
                transform=ax.transAxes,
                va='center',
                ha='left',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
            )

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"Trajectory {self.json_path.name} | Tracker {self.tracker_id}\n"
            f"KF - GT Error (Valid frames: {result['valid_count']}/{result['total_count']})",
            fontsize=12
        )
        plt.subplots_adjust(right=0.78, hspace=0.28)
        plt.show()
    
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
        pcd = self.create_point_cloud_from_depth(rgb_image, depth_array, frame=frame)

        # 可选过滤：剔除 x > 阈值 的点
        pcd = self.filter_point_cloud_by_x(pcd, frame=frame)

        # 估计ground truth球心（基于拟合，不使用点云质心）
        gt_result = self.get_or_compute_gt_result(frame_idx, frame=frame, pcd=pcd)
        
        # 下采样点云（加速显示）
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        # 创建几何体列表
        geometries = [pcd]
        
        # 添加world/body双坐标系
        geometries.extend(self.create_world_body_coordinate_frames(frame, size=0.2))
        
        # 添加检测位置标记（红色，实心，半径25mm）
        if self.show_detection_sphere and frame['detection_pos'] is not None:
            detection_sphere = self.create_sphere_marker(
                frame['detection_pos'],
                color=[1.0, 0.0, 0.0],  # 红色
                radius=0.025,  # 25mm
                wireframe=False  # 实心
            )
            geometries.append(detection_sphere)
        
        # 添加卡尔曼滤波位置标记（蓝色，线框半透明，半径37.5mm）
        if self.show_kf_sphere and frame['kf_pos'] is not None:
            kf_sphere = self.create_sphere_marker(
                frame['kf_pos'],
                color=[0.0, 0.0, 1.0],  # 蓝色
                radius=self.ball_radius,  # 37.5mm
                wireframe=True  # 线框模式（透明效果）
            )
            geometries.append(kf_sphere)

        # 添加ground truth位置标记（绿色，小实心）
        if self.show_ground_truth_marker and gt_result['position'] is not None:
            gt_sphere = self.create_sphere_marker(
                gt_result['position'],
                color=[0.0, 0.7, 0.0],
                radius=self.ball_radius,
                wireframe=False
            )
            geometries.append(gt_sphere)
        
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
        vis = o3d.visualization.VisualizerWithKeyCallback()
        
        # 构建窗口标题（包含位置和差值信息）
        title = f"Tracker {self.tracker_id} - Frame {frame_idx + 1}/{len(self.frames)} | Frame:{self.coord_frame}"
        if self.enable_x_filter:
            title += f" | X<= {self.x_filter_threshold:.2f}m"
        else:
            title += " | X Filter: OFF"
        title += f" | DetBall:{'ON' if self.show_detection_sphere else 'OFF'}"
        title += f" | KFBall:{'ON' if self.show_kf_sphere else 'OFF'}"
        title += f" | GT:{'ON' if self.show_ground_truth_marker else 'OFF'}"
        
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

        # 禁用Open3D默认P键截图/相机参数导出（会生成DepthCapture_*.png和DepthCamera_*.json）
        def _disable_o3d_capture(_vis):
            print("\n[P] 已禁用截图/相机参数导出")
            return False

        vis.register_key_callback(ord('P'), _disable_o3d_capture)
        vis.register_key_callback(ord('p'), _disable_o3d_capture)
        
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
        print(f"轨迹文件: {self.json_path.name}")
        print(f"轨迹坐标系: {self.coord_frame}")
        print(f"帧索引:   {frame.get('frame_index', frame_idx)}")
        print(f"RGB文件:   {rgb_path}")
        print(f"Depth文件: {depth_path}")
        print(f"GT球显示:  {'ON' if self.show_ground_truth_marker else 'OFF'}")
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

        if gt_result['position'] is not None:
            gt_pos = np.array(gt_result['position'])
            print(f"GT位置 (PointCloud Fit): [{gt_pos[0]:7.4f}, {gt_pos[1]:7.4f}, {gt_pos[2]:7.4f}] m")
            print(f"GT拟合RMSE: {gt_result['rmse']*1000:.2f} mm | inlier/cluster/total = "
                  f"{gt_result['inlier_points']}/{gt_result['cluster_points']}/{gt_result['total_points']}")
            if gt_result.get('icp_fitness') is not None:
                print(f"ICP质量: fitness={gt_result['icp_fitness']:.4f}, inlier_rmse={gt_result.get('icp_inlier_rmse', 0.0)*1000:.2f} mm")
        else:
            print(f"GT位置 (PointCloud Fit): None")
        
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
        print("  X             : 切换X轴点云过滤（剔除x>阈值）")
        print("  1             : 切换Detection球显示")
        print("  2             : 切换KF球显示")
        print("  G             : 切换Ground Truth位置显示")
        print("  3             : 切换Detection轨迹线显示")
        print("  4             : 切换GT轨迹线显示")
        print("  P             : 已禁用（不保存DepthCapture/DepthCamera文件）")
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

        def key_disable_capture(vis):
            print("\n[P] 已禁用截图/相机参数导出")
            return False

        def key_toggle_x_filter(vis):
            self.enable_x_filter = not self.enable_x_filter
            status = "ON" if self.enable_x_filter else "OFF"
            print(f"\n[X过滤] 状态: {status} (阈值: body_x <= {self.x_filter_threshold:.2f}m)")
            controller.should_update = True
            return False

        def key_toggle_detection_sphere(vis):
            self.show_detection_sphere = not self.show_detection_sphere
            status = "ON" if self.show_detection_sphere else "OFF"
            print(f"\n[Detection球] 显示: {status}")
            controller.should_update = True
            return False

        def key_toggle_kf_sphere(vis):
            self.show_kf_sphere = not self.show_kf_sphere
            status = "ON" if self.show_kf_sphere else "OFF"
            print(f"\n[KF球] 显示: {status}")
            controller.should_update = True
            return False

        def key_toggle_ground_truth(vis):
            self.show_ground_truth_marker = not self.show_ground_truth_marker
            status = "ON" if self.show_ground_truth_marker else "OFF"
            print(f"\n[GT位置] 显示: {status}")
            controller.should_update = True
            return False

        def key_toggle_detection_trajectory(vis):
            self.show_detection_trajectory = not self.show_detection_trajectory
            status = "ON" if self.show_detection_trajectory else "OFF"
            print(f"\n[Detection轨迹] 显示: {status}")
            controller.should_update = True
            return False

        def key_toggle_ground_truth_trajectory(vis):
            self.show_ground_truth_trajectory = not self.show_ground_truth_trajectory
            status = "ON" if self.show_ground_truth_trajectory else "OFF"
            print(f"\n[GT轨迹] 显示: {status}")
            controller.should_update = True
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
        vis.register_key_callback(ord('X'), key_toggle_x_filter)
        vis.register_key_callback(ord('x'), key_toggle_x_filter)
        vis.register_key_callback(ord('1'), key_toggle_detection_sphere)
        vis.register_key_callback(ord('2'), key_toggle_kf_sphere)
        vis.register_key_callback(ord('G'), key_toggle_ground_truth)
        vis.register_key_callback(ord('g'), key_toggle_ground_truth)
        vis.register_key_callback(ord('3'), key_toggle_detection_trajectory)
        vis.register_key_callback(ord('4'), key_toggle_ground_truth_trajectory)
        vis.register_key_callback(ord('P'), key_disable_capture)
        vis.register_key_callback(ord('p'), key_disable_capture)
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
                        pcd = self.create_point_cloud_from_depth(rgb_image, depth_array, frame=frame)

                        # 可选过滤：剔除 x > 阈值 的点
                        pcd = self.filter_point_cloud_by_x(pcd, frame=frame)

                        # 估计ground truth球心（基于拟合，不使用点云质心）
                        gt_result = self.get_or_compute_gt_result(frame_idx, frame=frame, pcd=pcd)

                        pcd = pcd.voxel_down_sample(voxel_size=0.01)
                        geometries.append(pcd)
                        
                        # 添加world/body双坐标系
                        geometries.extend(self.create_world_body_coordinate_frames(frame, size=0.2))
                        
                        # 添加检测位置标记（红色，实心，半径25mm）
                        if self.show_detection_sphere and frame['detection_pos'] is not None:
                            detection_sphere = self.create_sphere_marker(
                                frame['detection_pos'], 
                                color=[1.0, 0.0, 0.0],
                                radius=0.025,  # 25mm
                                wireframe=False  # 实心
                            )
                            geometries.append(detection_sphere)
                        
                        # 添加卡尔曼滤波位置标记（蓝色，线框半透明，半径37.5mm）
                        if self.show_kf_sphere and frame['kf_pos'] is not None:
                            kf_sphere = self.create_sphere_marker(
                                frame['kf_pos'], 
                                color=[0.0, 0.0, 1.0],
                                radius=self.ball_radius,  # 37.5mm
                                wireframe=True  # 线框模式（透明效果）
                            )
                            geometries.append(kf_sphere)

                        # 添加ground truth位置标记（绿色，小实心）
                        if self.show_ground_truth_marker and gt_result['position'] is not None:
                            gt_sphere = self.create_sphere_marker(
                                gt_result['position'],
                                color=[0.0, 0.7, 0.0],
                                radius=self.ball_radius,
                                wireframe=False
                            )
                            geometries.append(gt_sphere)

                        # 添加Detection轨迹与GT轨迹（从起始帧到当前帧）
                        detection_traj_points = []
                        gt_traj_points = []
                        for hist_idx in range(start_frame, frame_idx + 1):
                            hist_frame = self.frames[hist_idx]

                            if hist_frame.get('detection_pos') is not None:
                                detection_traj_points.append(hist_frame['detection_pos'])

                            hist_gt_result = self.get_or_compute_gt_result(hist_idx, frame=hist_frame)
                            if hist_gt_result.get('position') is not None:
                                gt_traj_points.append(hist_gt_result['position'])

                        if self.show_detection_trajectory:
                            det_line = self.create_trajectory_lineset(detection_traj_points, color=[1.0, 0.3, 0.3])
                            if det_line is not None:
                                geometries.append(det_line)

                        if self.show_ground_truth_trajectory:
                            gt_line = self.create_trajectory_lineset(gt_traj_points, color=[0.0, 0.6, 0.0])
                            if gt_line is not None:
                                geometries.append(gt_line)
                        
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
                            title = f"轨迹 {traj_idx}/{traj_total} | Tracker {self.tracker_id} - 帧 {frame_idx + 1}/{len(self.frames)} | Frame:{self.coord_frame}"
                        else:
                            title = f"Tracker {self.tracker_id} - 帧 {frame_idx + 1}/{len(self.frames)} | Frame:{self.coord_frame}"
                        if self.enable_x_filter:
                            title += f" | X<= {self.x_filter_threshold:.2f}m"
                        else:
                            title += " | X Filter: OFF"
                        title += f" | DetBall:{'ON' if self.show_detection_sphere else 'OFF'}"
                        title += f" | KFBall:{'ON' if self.show_kf_sphere else 'OFF'}"
                        title += f" | GT:{'ON' if self.show_ground_truth_marker else 'OFF'}"
                        title += f" | DetTraj:{'ON' if self.show_detection_trajectory else 'OFF'}"
                        title += f" | GTTraj:{'ON' if self.show_ground_truth_trajectory else 'OFF'}"
                        if frame['detection_pos'] is not None and frame['kf_pos'] is not None:
                            det_pos = np.array(frame['detection_pos'])
                            kf_pos = np.array(frame['kf_pos'])
                            dist = np.linalg.norm(det_pos - kf_pos)
                            title += f" | Err: {dist*1000:.1f}mm"
                        
                        # 打印详细帧信息到终端
                        print(f"\n{'='*80}")
                        print(f"帧 {frame_idx + 1}/{len(self.frames)} | 时间: {frame['relative_time']:.3f}s")
                        print(f"轨迹文件: {self.json_path.name}")
                        print(f"轨迹坐标系: {self.coord_frame}")
                        print(f"帧索引:   {frame.get('frame_index', frame_idx)}")
                        print(f"RGB文件:   {rgb_path}")
                        print(f"Depth文件: {depth_path}")
                        print(f"X过滤: {'ON' if self.enable_x_filter else 'OFF'} (body_x <= {self.x_filter_threshold:.2f}m)")
                        print(f"Detection球显示: {'ON' if self.show_detection_sphere else 'OFF'}")
                        print(f"KF球显示: {'ON' if self.show_kf_sphere else 'OFF'}")
                        print(f"GT球显示: {'ON' if self.show_ground_truth_marker else 'OFF'}")
                        print(f"Detection轨迹显示: {'ON' if self.show_detection_trajectory else 'OFF'}")
                        print(f"GT轨迹显示: {'ON' if self.show_ground_truth_trajectory else 'OFF'}")
                        
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

                        if gt_result['position'] is not None:
                            gt_pos = np.array(gt_result['position'])
                            print(f"GT位置 (PointCloud Fit): [{gt_pos[0]:7.4f}, {gt_pos[1]:7.4f}, {gt_pos[2]:7.4f}] m")
                            print(f"GT拟合RMSE: {gt_result['rmse']*1000:.2f} mm ")
                            if gt_result.get('icp_fitness') is not None:
                                print(f"ICP质量: fitness={gt_result['icp_fitness']:.4f}, inlier_rmse={gt_result.get('icp_inlier_rmse', 0.0)*1000:.2f} mm")
                        else:
                            print(f"GT位置 (PointCloud Fit): None")
                        
                        print(f"{'='*80}")
                    
                    controller.should_update = False
                
                # 更新可视化
                if not vis.poll_events():
                    break
                vis.update_renderer()
            
            print("\n\n可视化结束")
            # 每条轨迹可视化完成后，显示Detection-GT差值统计图
            self.show_detection_gt_error_plots(start_frame=start_frame, end_frame=end_frame)
            # 每条轨迹可视化完成后，显示KF-GT差值统计图
            self.show_kf_gt_error_plots(start_frame=start_frame, end_frame=end_frame)
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
    parser.add_argument('--enable-x-filter', action='store_true',
                       help='启用点云X轴过滤（显示时剔除 x > --x-max 的点）')
    parser.add_argument('--x-max', type=float, default=1.0,
                       help='点云X轴过滤阈值，保留 x <= x-max（米，默认1.0）')
    parser.add_argument('--hide-gt-marker', action='store_true',
                       help='隐藏拟合得到的Ground Truth位置标记（可在交互界面按G再次打开）')
    parser.add_argument('--no-pre-stats', action='store_true',
                       help='关闭可视化前的Detection-GT预统计输出')
    parser.add_argument('--add-gt-pos', action='store_true',
                       help='可开关：将拟合GT写入gt_pos字段并写回轨迹JSON文档（不覆盖detection_pos）')
    
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

        # 可选：先把拟合GT写回文档（新增gt_pos）
        if args.add_gt_pos:
            print("\n[文档写回] 开始执行: 将拟合GT写入gt_pos ...")
            for idx, json_file in enumerate(json_files, start=1):
                try:
                    writer = TrajectoryVisualizer(json_file)
                    writer.enable_x_filter = args.enable_x_filter
                    writer.x_filter_threshold = args.x_max
                    writer.show_ground_truth_marker = not args.hide_gt_marker
                    if args.cam_height != 0.3 or args.cam_forward != 0.0 or args.cam_lateral != 0.0:
                        camera_rotation_in_body = writer.camera_to_body_transform[:3, :3]
                        writer.camera_to_body_transform = np.eye(4)
                        writer.camera_to_body_transform[:3, :3] = camera_rotation_in_body
                        writer.camera_to_body_transform[:3, 3] = [args.cam_forward, args.cam_lateral, args.cam_height]

                    write_result = writer.add_gt_pos_and_save(
                        start_frame=args.start,
                        end_frame=args.end,
                        create_backup=True
                    )
                    print(
                        f"  [{idx}/{len(json_files)}] {json_file.name}: "
                        f"updated={write_result['updated']}, "
                        f"skipped_no_gt={write_result['skipped_no_gt']}, "
                        f"total={write_result['total']}"
                    )
                except Exception as e:
                    print(f"  [失败] {json_file.name}: {e}")
            print("[文档写回] 完成\n")

        if not args.no_pre_stats:
            # 可视化开始前，先打印所有轨迹整体的Detection-GT差值统计
            print("\n" + "="*80)
            print("可视化开始前统计：所有轨迹整体 [Detection-GT / KF-GT 差值统计]")
            print("="*80)

            all_det_errors = []
            det_valid_frame_sum = 0
            det_total_frame_sum = 0
            det_used_trajectory_count = 0

            all_kf_errors = []
            kf_valid_frame_sum = 0
            kf_total_frame_sum = 0
            kf_used_trajectory_count = 0

            for idx, json_file in enumerate(json_files, start=1):
                try:
                    stat_visualizer = TrajectoryVisualizer(json_file)
                    stat_visualizer.enable_x_filter = args.enable_x_filter
                    stat_visualizer.x_filter_threshold = args.x_max
                    stat_visualizer.show_ground_truth_marker = not args.hide_gt_marker
                    if args.cam_height != 0.3 or args.cam_forward != 0.0 or args.cam_lateral != 0.0:
                        camera_rotation_in_body = stat_visualizer.camera_to_body_transform[:3, :3]
                        stat_visualizer.camera_to_body_transform = np.eye(4)
                        stat_visualizer.camera_to_body_transform[:3, :3] = camera_rotation_in_body
                        stat_visualizer.camera_to_body_transform[:3, 3] = [args.cam_forward, args.cam_lateral, args.cam_height]

                    det_stat_result = stat_visualizer.compute_detection_gt_error_statistics(
                        start_frame=args.start,
                        end_frame=args.end
                    )

                    kf_stat_result = stat_visualizer.compute_kf_gt_error_statistics(
                        start_frame=args.start,
                        end_frame=args.end
                    )

                    if det_stat_result is not None:
                        all_det_errors.append(det_stat_result['e'])
                        det_valid_frame_sum += det_stat_result['valid_count']
                        det_total_frame_sum += det_stat_result['total_count']
                        det_used_trajectory_count += 1
                    else:
                        det_total_frame_sum += (args.end - args.start) if args.end is not None else len(stat_visualizer.frames) - args.start

                    if kf_stat_result is not None:
                        all_kf_errors.append(kf_stat_result['e'])
                        kf_valid_frame_sum += kf_stat_result['valid_count']
                        kf_total_frame_sum += kf_stat_result['total_count']
                        kf_used_trajectory_count += 1
                    else:
                        kf_total_frame_sum += (args.end - args.start) if args.end is not None else len(stat_visualizer.frames) - args.start
                except Exception as e:
                    print(f"  轨迹统计失败 [{idx}/{len(json_files)}] {json_file.name}: {e}")

            if len(all_det_errors) == 0:
                print("[Detection-GT差值统计] 所有轨迹整体: 无可用的Detection-GT配对数据")
            else:
                all_e = np.vstack(all_det_errors)
                stats_x = TrajectoryVisualizer._calc_error_stats(all_e[:, 0])
                stats_y = TrajectoryVisualizer._calc_error_stats(all_e[:, 1])
                stats_z = TrajectoryVisualizer._calc_error_stats(all_e[:, 2])

                print(f"[Detection-GT差值统计] 所有轨迹整体 (det - gt), 单位: m")
                for axis_name, s in zip(['X', 'Y', 'Z'], [stats_x, stats_y, stats_z]):
                    print(
                        f"  {axis_name}: max={s['max']:.6f}, min={s['min']:.6f}, mean={s['mean']:.6f}, std={s['std']:.6f}, "
                        f"abs_mean={s['abs_mean']:.6f}, abs_std={s['abs_std']:.6f}"
                    )
                print(f"  有效帧总数: {det_valid_frame_sum}/{det_total_frame_sum}")
                print(f"  参与统计轨迹数: {det_used_trajectory_count}/{len(json_files)}")

            if len(all_kf_errors) == 0:
                print("[KF-GT差值统计] 所有轨迹整体: 无可用的KF-GT配对数据")
            else:
                all_e = np.vstack(all_kf_errors)
                stats_x = TrajectoryVisualizer._calc_error_stats(all_e[:, 0])
                stats_y = TrajectoryVisualizer._calc_error_stats(all_e[:, 1])
                stats_z = TrajectoryVisualizer._calc_error_stats(all_e[:, 2])

                print(f"[KF-GT差值统计] 所有轨迹整体 (kf - gt), 单位: m")
                for axis_name, s in zip(['X', 'Y', 'Z'], [stats_x, stats_y, stats_z]):
                    print(
                        f"  {axis_name}: max={s['max']:.6f}, min={s['min']:.6f}, mean={s['mean']:.6f}, std={s['std']:.6f}, "
                        f"abs_mean={s['abs_mean']:.6f}, abs_std={s['abs_std']:.6f}"
                    )
                print(f"  有效帧总数: {kf_valid_frame_sum}/{kf_total_frame_sum}")
                print(f"  参与统计轨迹数: {kf_used_trajectory_count}/{len(json_files)}")
            print("="*80 + "\n")
        else:
            print("\n已关闭可视化前预统计（--no-pre-stats）\n")
        
        current_trajectory = 0
        
        try:
            while 0 <= current_trajectory < len(json_files):
                json_file = json_files[current_trajectory]
                print(f"\n{'='*60}")
                print(f"轨迹 {current_trajectory+1}/{len(json_files)}: {json_file.name}")
                print('='*60)
                
                try:
                    visualizer = TrajectoryVisualizer(json_file)
                    visualizer.enable_x_filter = args.enable_x_filter
                    visualizer.x_filter_threshold = args.x_max
                    visualizer.show_ground_truth_marker = not args.hide_gt_marker
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
        visualizer.enable_x_filter = args.enable_x_filter
        visualizer.x_filter_threshold = args.x_max
        visualizer.show_ground_truth_marker = not args.hide_gt_marker
        # 如果提供了命令行参数，更新相机变换
        if args.cam_height != 0.3 or args.cam_forward != 0.0 or args.cam_lateral != 0.0:
            print(f"警告: 使用自定义相机参数会覆盖实际外参")
            camera_rotation_in_body = visualizer.camera_to_body_transform[:3, :3]
            visualizer.camera_to_body_transform = np.eye(4)
            visualizer.camera_to_body_transform[:3, :3] = camera_rotation_in_body
            visualizer.camera_to_body_transform[:3, 3] = [args.cam_forward, args.cam_lateral, args.cam_height]

        if args.add_gt_pos:
            write_result = visualizer.add_gt_pos_and_save(
                start_frame=args.start,
                end_frame=args.end,
                create_backup=True
            )
            print(
                f"[文档写回] {trajectory_path.name}: updated={write_result['updated']}, "
                f"skipped_no_gt={write_result['skipped_no_gt']}, total={write_result['total']}"
            )
            if write_result['backup_path'] is not None:
                print(f"[文档写回] 备份文件: {write_result['backup_path']}")

        # 可视化开始前先打印当前轨迹统计（可开关）
        if not args.no_pre_stats:
            visualizer.print_detection_gt_error_statistics(start_frame=args.start, end_frame=args.end)
            visualizer.print_kf_gt_error_statistics(start_frame=args.start, end_frame=args.end)
        else:
            print("\n已关闭可视化前预统计（--no-pre-stats）\n")
        visualizer.visualize_all(start_frame=args.start, end_frame=args.end)


if __name__ == '__main__':
    main()
