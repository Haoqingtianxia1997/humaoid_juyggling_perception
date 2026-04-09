#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from std_msgs.msg import Float32MultiArray, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
import cv2
import os
from datetime import datetime
import numpy as np
import shutil
import time
import yaml
from pathlib import Path
from perception import CameraIntrinsics, BallTracker, BallTrackingVisualizer
import message_filters


class BallTrackingNode(Node):
    """
    完整的球追踪 ROS 节点
    - 订阅 RGB、深度图像
    - 使用 RobotClient 获取 IMU 位置和姿态
    - 使用 TF2 获取相机外参
    - 发布 catch_info
    """
    def __init__(self):
        super().__init__('ball_tracking_node')
        self.bridge = CvBridge()
        
        # === 相机参数配置 ===
        config_path = Path(__file__).parent / 'Tracker_config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"相机配置文件不存在: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.get_logger().info(f"Loaded camera config from: {config_path}")

        camera_section = config.get('camera', {})
        camera_profiles = camera_section.get('profiles', {})
        camera_profile_name = str(camera_section.get('profile', 'left_rect'))
        profile_cfg = camera_profiles.get(camera_profile_name, {})

        self.camera_topic = profile_cfg.get('image_topic', '/zed/zed_node/left/image_rect_color')
        self.depth_topic = profile_cfg.get('depth_topic', '/zed/zed_node/depth/depth_registered')

        undistort_cfg = profile_cfg.get('undistort', {})
        self.use_raw_undistort = bool(undistort_cfg.get('enabled', False))
        self.map1 = None
        self.map2 = None

        if self.use_raw_undistort:
            self.img_width = int(undistort_cfg.get('width', 640))
            self.img_height = int(undistort_cfg.get('height', 360))

            K_cfg = undistort_cfg.get('K', None)
            D_cfg = undistort_cfg.get('D', None)
            if K_cfg is None or D_cfg is None:
                raise ValueError(
                    f"camera.profiles.{camera_profile_name}.undistort 缺少 K 或 D 配置"
                )

            self.K = np.asarray(K_cfg, dtype=np.float64).reshape(3, 3)
            self.D = np.asarray(D_cfg, dtype=np.float64).reshape(-1)

            # rectification rotation
            R = np.eye(3, dtype=np.float64)
            balance = float(undistort_cfg.get('balance', 0.0))

            # 去畸变后的新内参
            self.K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K,
                self.D,
                (self.img_width, self.img_height),
                R,
                balance=balance
            )

            # 预计算 undistort map
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K,
                self.D,
                R,
                self.K_new,
                (self.img_width, self.img_height),
                cv2.CV_16SC2
            )

        self.get_logger().info(
            f"Camera profile={camera_profile_name}, image_topic={self.camera_topic}, depth_topic={self.depth_topic}, "
            f"undistort={'on' if self.use_raw_undistort else 'off'}"
        )
        
        
        # === 数据存储 ===
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_image_timestamp = None  # 保存图像时间戳（用于marker时间同步）
        self.prev_predict_time_sec = None   # 上一帧预测时刻（秒，用于动态dt）
        self.latest_cam_pos = np.array([0.0, 0.0, 0.0])  # 相机世界位置（用于marker可视化）
        self.latest_cam_rot = np.eye(3)  # 相机旋转矩阵
        self.latest_base_pos = np.array([0.0, 0.0, 0.0])  # 机器人本体位置（用于坐标转换）
        self.latest_base_rot = np.eye(3)  # 机器人本体旋转矩阵（用于坐标转换）
        
        # === 图像变化检测 ===
        self.prev_rgb_id = None  # 上一帧RGB图像的id
        self.prev_depth_id = None  # 上一帧深度图像的id
        
        # === TF2 监听器 ===
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # === 相机参数 ===
        # 优先使用 profile 内的 intrinsics；否则回退到全局 intrinsics
        intr = profile_cfg.get('intrinsics', config['intrinsics'])
        self.camera_width = intr['width']
        self.camera_height = intr['height']
        self.camera_intrinsics = CameraIntrinsics(
            width=self.camera_width,
            height=self.camera_height,
            fx=intr['fx'],
            fy=intr['fy'],
            cx=intr['cx'],
            cy=intr['cy']
        )

        # 如果使用 raw 图像并进行了 fisheye 去畸变，
        # 则追踪所用内参必须与 remap 后图像一致（即 K_new）。
        if self.use_raw_undistort:
            fx_new = float(self.K_new[0, 0])
            fy_new = float(self.K_new[1, 1])
            cx_new = float(self.K_new[0, 2])
            cy_new = float(self.K_new[1, 2])

            self.camera_width = int(self.img_width)
            self.camera_height = int(self.img_height)
            self.camera_intrinsics = CameraIntrinsics(
                width=self.camera_width,
                height=self.camera_height,
                fx=fx_new,
                fy=fy_new,
                cx=cx_new,
                cy=cy_new,
            )
            self.get_logger().info(
                "Using image_raw with undistort intrinsics: "
                f"fx={fx_new:.3f}, fy={fy_new:.3f}, cx={cx_new:.3f}, cy={cy_new:.3f}"
            )
        
        # === 球追踪器 ===
        # 从配置文件加载tracker参数
        tracker_config = config['tracker']
        runtime_cfg = tracker_config.get('runtime', {})
        detector_cfg = tracker_config.get('detector', {})
        trajectory_cfg = tracker_config.get('trajectory', {})

        self.num_balls = int(runtime_cfg.get('num_balls', tracker_config.get('num_balls', 1)))
        self.dt = float(runtime_cfg.get('dt', tracker_config.get('dt', 1.0 / 60.0)))
        self.use_robot_data = bool(runtime_cfg.get('use_robot_data', tracker_config.get('use_robot_data', True)))
        self.dt_dynamic = None
        self.ball_tracker = BallTracker(tracker_config=tracker_config)
        self.center_border_pixels = int(detector_cfg.get('center_border_pixels', tracker_config.get('center_border_pixels', 50)))
        self.center_method = detector_cfg.get('center_method', tracker_config.get('center_method', 'min_depth'))
        self.ball_radius = float(detector_cfg.get('ball_radius', tracker_config.get('ball_radius', 0.0375)))
        # 轨迹保存坐标系：'body' 或 'world'
        self.trajectory_coord_frame = str(
            trajectory_cfg.get('coord_frame', tracker_config.get('trajectory_coord_frame', 'body'))
        ).lower()
        if self.trajectory_coord_frame not in ('body', 'world'):
            self.get_logger().warn(
                f"Invalid tracker.trajectory_coord_frame={self.trajectory_coord_frame}, fallback to 'body'"
            )
            self.trajectory_coord_frame = 'body'
        
        # === 地面高度阈值 ===
        self.ground_z_threshold = float(
            runtime_cfg.get('ground_z_threshold', tracker_config.get('ground_z_threshold', -0.2))
        )
        
        # === 卡尔曼滤波观测 ===
        self.kf_obs = [None] * self.num_balls
        self.kf_obs_body = [None] * self.num_balls
        self.max_velocity_uncertainty = float(
            runtime_cfg.get('max_velocity_uncertainty', tracker_config.get('max_velocity_uncertainty', 0.5))
        )
        
        # === 测试模式配置 ===
        test_mode_config = config.get('test_mode', {})
        self.test_mode_enabled = test_mode_config.get('enable', False)
        self.max_upgrades_after_valid = test_mode_config.get('max_upgrades_after_valid', 10)
        self.upgrade_counter = [0] * self.num_balls  # 记录每个tracker的upgrade次数
        self.get_logger().info(f"Test mode: {'Enabled' if self.test_mode_enabled else 'Disabled'}")
        if self.test_mode_enabled:
            self.get_logger().info(f"Max upgrades after valid: {self.max_upgrades_after_valid}")
        
        # === 轨迹记录 ===
        self.enable_trajectory_recording = True  # 是否启用轨迹记录
        self.trajectory_data = {}  # {tracker_id: [{timestamp, detection_pos, kf_pos, kf_vel, frame_idx}, ...]}
        self.trajectory_start_time = {}  # {tracker_id: start_timestamp}
        self.trajectory_counter = 0  # 轨迹计数器
        self.trajectory_save_dir = os.path.join(os.path.dirname(__file__), 'trajectory_data')
        if os.path.exists(self.trajectory_save_dir):
            shutil.rmtree(self.trajectory_save_dir)
        if self.enable_trajectory_recording:
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            self.get_logger().info(f"Trajectory data will be saved to: {self.trajectory_save_dir}")
        
        # === 可视化相关 ===
        self.enable_3D_visualization = False  # 是否启用3D可视化（matplotlib）
        self.enable_visualization = True  # 是否启用所有可视化
        self.latest_rgb_vis = None  # 带标注的 RGB 图像
        self.latest_depth_vis = None  # 带标注的深度图像
        self.latest_has_detection = {}  # 最新的检测状态
        self.latest_detection_results = []  # 最新的检测结果（用于marker可视化）
        
        # === 3D 可视化 ===
        if self.enable_3D_visualization:
            self.visualizer = BallTrackingVisualizer(
                num_balls=self.num_balls
            )
            pass
        
        if self.enable_visualization:
            cv2.namedWindow('Ball Tracking - RGB', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Ball Tracking - Depth', cv2.WINDOW_NORMAL)
            # 添加可视化更新定时器（33Hz）
            self.viz_timer = self.create_timer(0.03, self.update_visualization)
        
        # === ROS 订阅器（使用消息同步）===
        # 创建订阅器但不直接注册回调
        rgb_sub = message_filters.Subscriber(
            self,
            Image,
            self.camera_topic
        )
        
        depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic
        )
        
        # 使用近似时间同步器（允许5ms的时间差）
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=10,
            slop=0.001  # 1ms 时间容差
        )
        self.sync.registerCallback(self.images_callback)
        
        # 订阅 world_model_prediction
        self.world_model_pred_sub = self.create_subscription(
            Float32MultiArray,
            'world_model_prediction',
            self.world_model_pred_callback,
            1
        )

        
        # === ROS 发布器 ===
        # catch_ball_info: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        self.catch_ball_info_pub = self.create_publisher(
            Float32MultiArray,
            'catch_info',
            10
        )
        
        # marker publisher for visualization
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/ball_markers',
            1
        )
        
        # === 处理定时器（50Hz，与控制频率同步） ===
        # self.timer = self.create_timer(0.02, self.process_tracking)
        
        if self.use_robot_data:
            from robot_bridge_py.robot_client import RobotClient
            # === RobotClient（用于获取 IMU 数据）===
            self.robot = RobotClient(
                node=self,
                robot_type="H1",
                num_dof=20,
                control_frequency=50.0,
                interpolation_order=0.0
            )
            
        # === 统计信息 ===
        self.frame_count = 0
        self.last_process_time = time.time()
        self.get_logger().info("✅ Ball Tracking Node Started.")
        self.get_logger().info(f"Tracking {self.num_balls} balls at {1/self.dt:.0f}Hz")
        self.get_logger().info(f"Trajectory coordinate frame: {self.trajectory_coord_frame}")
        
        if self.use_robot_data:
            self.get_logger().info("Using RobotClient for IMU data")
        else:
            self.get_logger().info("Not using RobotClient for IMU data")
        # if self.enable_visualization:
        #     self.get_logger().info("Visualization enabled - Press 'q' to quit")
        
    def images_callback(self, rgb_msg, depth_msg):
        """
        同步接收 RGB 和深度图像（保证时间对齐）
        
        Args:
            rgb_msg: RGB 图像消息
            depth_msg: 深度图像消息
        """
        # 处理 RGB 图像
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        
        # 处理深度图像
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        
        if self.use_raw_undistort and self.map1 is not None and self.map2 is not None:
            #去畸变
            cv_image = cv2.remap(
            cv_image,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
            
        
        # 使用深度信息创建mask，只保留1米以内的区域
        cv_image_masked = cv_image.copy()
        
        # 创建深度mask：深度在0到1.0米之间的像素
        depth_mask = (cv_depth > 0) & (cv_depth < 1.0) & np.isfinite(cv_depth)
        
        # 对mask进行形态学处理以去除噪声
        kernel = np.ones((5, 5), np.uint8)
        # 开运算：先腐蚀后膨胀，去除小噪点
        depth_mask = depth_mask.astype(np.uint8) * 255
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        # 闭运算：先膨胀后腐蚀，填补空洞
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
        depth_mask = depth_mask > 0
        
        # 将深度大于1.0米的区域设置为黑色
        cv_image_masked[~depth_mask] = 0
        
        # 保存同步后的图像和时间戳
        self.latest_rgb = cv_image_masked
        self.latest_depth = cv_depth
        self.latest_image_timestamp = rgb_msg.header.stamp  # 保存时间戳用于marker同步
        # 直接在图像回调中处理追踪（确保处理与图像同步）
        self.process_tracking()
        
    
    def world_model_pred_callback(self, msg):
        """接收world model预测数据"""
        if len(msg.data) >= 3:
            self.latest_base_pos = np.array([msg.data[0], msg.data[1], msg.data[2]])
        else:
            self.latest_base_pos = np.array([0.0, 0.0, 0.0])
            
    def get_robot_imu_data(self):
        """
        从 RobotClient 获取机器人 IMU 姿态数据
        
        Returns:
            quat: 四元数 [w, x, y, z]
            angular_vel: 角速度 [x, y, z]
        """
        if not self.use_robot_data:
            return np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
        # 更新机器人状态
        self.robot.update_robot_state()
        
        # 获取四元数 (w, x, y, z)
        quat = self.robot.quat
        # print(f"Robot IMU Quaternion: {quat}")
        # 获取角速度
        angular_vel = self.robot.angular_velocity
        
        return quat, angular_vel
            
    @staticmethod
    def quat_to_rot_matrix(quat):
        """
        将四元数转换为旋转矩阵
        quat: [w, x, y, z]
        """
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    def get_camera_extrinsics(self):
        """
        使用 TF2 获取相机外参（相对变换）
        从 imu_link（机器人本体）到 zed_left_camera_optical_frame 的变换
        
        Returns:
            cam_pos_rel: 相机相对于机器人本体的位置 [x, y, z]
            cam_rot_rel: 相机相对于机器人本体的旋转矩阵 (3x3)
        """
        try:
            if self.use_robot_data:
                # 从配置文件读取相机外参
                config_path = Path(__file__).parent / 'Tracker_config.yaml'
                with open(config_path, 'r') as f:
                    camera_config = yaml.safe_load(f)
                extr = camera_config['extrinsics']
                cam_pos = np.array(extr['position'])
                cam_rot = np.array(extr['rotation'])
                return cam_pos, cam_rot
            else:
                # 使用已知标定外参（相机坐标系 -> body/imu_link 坐标系）
                cam_pos_rel = np.array([0.00, 0.00, 0.00], dtype=np.float64)
                cam_rot_rel = np.array([
                    [0.0,  0.0,  1.0],
                    [-1.0, 0.0,  0.0],
                    [0.0, -1.0,  0.0],
                ], dtype=np.float64)

                return cam_pos_rel, cam_rot_rel
            
        except Exception as e:
            self.get_logger().warn(f"Failed to get camera extrinsics: {e}")
            # 返回默认值（相机在基座正前方）
            return np.array([0.0, 0.0, 0.5]), np.eye(3)
    
    def get_base_rotation(self):
        """
        从 RobotClient 获取机器人本体当前旋转矩阵
        
        Returns:
            base_rot: 3x3 旋转矩阵（机器人本体姿态）
        """
        quat, _ = self.get_robot_imu_data()
        # 检查四元数是否有效
        if quat is None or (quat[0] == 0 and quat[1] == 0 and quat[2] == 0 and quat[3] == 0):
            return np.eye(3)
            
        return self.quat_to_rot_matrix(quat)
    
    def process_tracking(self):
    
        import time 
        start_time = time.perf_counter()
        

        # 预测步长优先使用前后两帧时间差；无有效时间差时回退固定 dt
        predict_time_sec = None
        if self.latest_image_timestamp is not None:
            stamp = self.latest_image_timestamp
            if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
                predict_time_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
                current_time = predict_time_sec

        if self.prev_predict_time_sec is not None:
            dt_dynamic = predict_time_sec - self.prev_predict_time_sec
            if not (np.isfinite(dt_dynamic) and dt_dynamic > 0.0):
                dt_dynamic = self.dt
        else:
            dt_dynamic = self.dt
        self.prev_predict_time_sec = predict_time_sec
        self.dt_dynamic = dt_dynamic
        print(f"Predict time: {dt_dynamic}")

        # 检查数据是否就绪
        if self.latest_rgb is None or self.latest_depth is None:
            print("Waiting for RGB and Depth data...")
            return
        
        self.frame_count += 1
        
        # === 获取当前机器人本体状态 ===
        # 从 world_model_prediction 获取位置
        base_pos_world = self.latest_base_pos.copy()  # 机器人本体世界位置（来自 world_model_prediction）
        base_rot = self.get_base_rotation()            # 机器人本体世界姿态（来自 RobotClient）
        
        # 转换到相对坐标（相对于初始位置）
        base_pos = base_pos_world 
        
        # === 获取相机外参（相对于机器人本体的变换） ===
        cam_pos_rel, cam_rot_rel = self.get_camera_extrinsics()
        
        # === 计算相机在世界坐标系中的位置和姿态 ===
        # 相机世界位置 = 机器人本体位置 + 旋转后的相机相对位置
        cam_pos = base_pos + base_rot @ cam_pos_rel
        # 相机世界姿态 = 机器人本体姿态 × 相机相对姿态
        cam_rot = base_rot @ cam_rot_rel
        # 保存相机位置和机器人本体位置用于marker可视化和坐标转换
        self.latest_cam_pos = cam_pos
        self.latest_cam_rot = cam_rot
        self.latest_base_pos = base_pos
        self.latest_base_rot = base_rot
        
        

        
        # === 预测步骤（每帧都执行） ===
        if any(self.ball_tracker.is_validated(i) for i in range(self.num_balls)):
            self.ball_tracker.predict_all(
                ground_z_threshold=self.ground_z_threshold,
                dt=dt_dynamic,
                base_site_pos=base_pos,
            )
        
        # === 保存已经落地的轨迹（在清理之前） ===
        if self.enable_trajectory_recording:
            for tracker_id in range(self.num_balls):
                if tracker_id in self.trajectory_data and self.ball_tracker.is_grounded(tracker_id):
                    self.save_trajectory(tracker_id)
        
        # 清理已经落地的球（每帧都执行）
        self.ball_tracker.cleanup_grounded_balls(self.kf_obs, self.kf_obs_body)
        
        # === 记录预测状态 ===
        self.ball_tracker.record_prediction_states(
            base_rot, base_pos,
            self.kf_obs, self.kf_obs_body,
            self.max_velocity_uncertainty
        )
        
        # === 图像变化检测 ===
        # 检查当前图像是否与上一帧不同
        current_rgb_id = id(self.latest_rgb)
        current_depth_id = id(self.latest_depth)
        image_changed = (current_rgb_id != self.prev_rgb_id or current_depth_id != self.prev_depth_id)
        
        # 转换 RGB 为 BGR（OpenCV 格式）
        rgb_bgr = self.latest_rgb.copy()
        depth_array = self.latest_depth.copy()
        
        # === 检测和更新追踪 ===
        detection_results = []
        has_detection = {}
        actually_updated = {}
        detection_assignments = {}
        
        if image_changed:
            # 图像发生变化，进行检测和更新
            has_detection, detection_results, actually_updated, detection_assignments, kf_obs_out, kf_obs_body_out = \
                self.ball_tracker.process_detection_and_update(
                    rgb_bgr, depth_array,
                    self.camera_intrinsics, cam_pos, cam_rot,
                    base_site_rot=base_rot,
                    base_site_pos=base_pos,
                    kf_obs=self.kf_obs,
                    kf_obs_body=self.kf_obs_body,
                    max_velocity_uncertainty=self.max_velocity_uncertainty,
                    test_mode_enabled=self.test_mode_enabled,
                    upgrade_counter=self.upgrade_counter,
                    max_upgrades_after_valid=self.max_upgrades_after_valid,
                    center_method=self.center_method,
                    ball_radius=self.ball_radius
                )
            self.kf_obs = kf_obs_out
            self.kf_obs_body = kf_obs_body_out
            
            # 在test模式下，根据actually_updated更新upgrade_counter
            if self.test_mode_enabled:
                for tracker_id in range(self.num_balls):
                    if actually_updated[tracker_id]:
                        self.upgrade_counter[tracker_id] += 1
            
            # 更新上一帧图像ID
            self.prev_rgb_id = current_rgb_id
            self.prev_depth_id = current_depth_id
            
            # 保存检测状态和结果用于可视化
            self.latest_has_detection = has_detection
            # 同时保存detection的body坐标系位置（在检测时刻转换）
            self.latest_detection_results = []
            for pos_world, det_info, ray_info in detection_results:
                pos_body = base_rot.T @ (pos_world - base_pos)
                self.latest_detection_results.append((pos_body, det_info, ray_info))
        else:
            # 图像未变化，初始化空字典
            for tracker_id in range(self.num_balls):
                has_detection[tracker_id] = False
                actually_updated[tracker_id] = False
            detection_assignments = {}
        
        # === 记录轨迹数据 ===
        if self.enable_trajectory_recording:
            self.record_trajectory_frame(
                has_detection,
                actually_updated,
                detection_results,
                detection_assignments,
                base_rot,
                base_pos,
                rgb_bgr,
                depth_array,
                current_time
            )
            
        # === 从 kf_obs_body 提取 catch_info ===
        catch_info = self.ball_tracker.catch_info_from_kf_obs_body(self.kf_obs_body)
        
        # === 生成可视化图像 ===
        if self.enable_visualization:
            self.generate_visualization_images(rgb_bgr, depth_array, detection_results)
        
        # === 发布球位置markers（包括catch_info） ===
        self.publish_ball_markers(catch_info)
        
        # === 发布 world_model_prediction ===
        if catch_info is not None:
            msg = Float32MultiArray()
            msg.data = [
                float(catch_info['position'][0]),
                float(catch_info['position'][1]),
                float(catch_info['position'][2]),
                float(catch_info['velocity'][0]),
                float(catch_info['velocity'][1]),
                float(catch_info['velocity'][2])
            ]
            self.catch_ball_info_pub.publish(msg)
            
            # 打印日志（降低频率）
            if self.frame_count % 10 == 0:
                self.get_logger().info(
                    f"Catch Ball Info - Pos: [{catch_info['position'][0]:.3f}, "
                    f"{catch_info['position'][1]:.3f}, {catch_info['position'][2]:.3f}], "
                    f"Vel: [{catch_info['velocity'][0]:.3f}, "
                    f"{catch_info['velocity'][1]:.3f}, {catch_info['velocity'][2]:.3f}]"
                )
        else:
            # 发布零值
            msg = Float32MultiArray()
            msg.data = [0.0] * 6
            self.catch_ball_info_pub.publish(msg)
            
            if self.frame_count % 25 == 0:
                self.get_logger().warn("No valid catch ball info available")
        
        end_time = time.perf_counter()
        # print(f"Frame {self.frame_count} processed in {(end_time - start_time)*1000:.1f} ms")
        
        
        # === 性能统计 ===
        if self.frame_count % 50 == 0:
            current_time = time.time()
            elapsed = current_time - self.last_process_time
            fps = 50.0 / elapsed if elapsed > 0 else 0.0
            self.get_logger().info(f"Tracking FPS: {fps:.1f}")
            self.last_process_time = current_time
    
    def record_trajectory_frame(self, has_detection, actually_updated, detection_results, detection_assignments, base_rot, base_pos, rgb_image, depth_image, current_time):
        """
        记录当前帧的轨迹数据，包括RGB和depth图像
        
        Args:
            has_detection: 检测状态字典（是否有检测匹配）
            actually_updated: 实际更新状态字典（是否真正执行了update操作）
            detection_results: 检测结果列表（world坐标系）
            detection_assignments: Tracker与检测索引匹配关系 {tracker_id: det_idx}
            base_rot: 基座旋转矩阵
            base_pos: 基座位置
            rgb_image: RGB图像（BGR格式）
            depth_image: 深度图像
            current_time: 当前时间
        """
        
        # 为每个tracker记录数据
        for tracker_id in range(self.num_balls):
            if self.ball_tracker.is_validated(tracker_id):
                # 检查是否是新的轨迹（首次验证通过）
                if tracker_id not in self.trajectory_data:
                    self.trajectory_data[tracker_id] = []
                    self.trajectory_start_time[tracker_id] = current_time
                    self.upgrade_counter[tracker_id] = 0  # 重置upgrade计数器
                    self.get_logger().info(
                        f"Started recording trajectory for tracker {tracker_id}, temp dir: "
                        f"{os.path.join(self.trajectory_save_dir, f'tracker{tracker_id}_temp')}"
                    )
                
                # 根据配置选择保存坐标系
                if self.trajectory_coord_frame == 'world':
                    kf_data = self.kf_obs[tracker_id]
                else:
                    kf_data = self.kf_obs_body[tracker_id]
                
                # 获取detection位置（如果有）
                detection_pos = None
                contour_area = None
                if has_detection.get(tracker_id, False):
                    # 使用Tracker内部匹配结果，避免重复做最近邻匹配
                    det_idx = detection_assignments.get(tracker_id, None)
                    if det_idx is not None and 0 <= det_idx < len(detection_results):
                        det_pos_world = detection_results[det_idx][0]
                        det_info = detection_results[det_idx][1]
                        if det_info is not None and det_info.get('area', None) is not None:
                            contour_area = float(det_info['area'])
                        if self.trajectory_coord_frame == 'world':
                            detection_pos = det_pos_world.tolist()
                        else:
                            det_pos_body = base_rot.T @ (det_pos_world - base_pos)
                            detection_pos = det_pos_body.tolist()
                
                if kf_data is not None:
                    # 当前轨迹的帧索引
                    frame_idx = len(self.trajectory_data[tracker_id])
                    
                    # 保存RGB和depth图像
                    traj_dir = os.path.join(self.trajectory_save_dir, f"tracker{tracker_id}_temp")
                    os.makedirs(traj_dir, exist_ok=True)
                    
                    rgb_filename = os.path.join(traj_dir, f"rgb_{frame_idx:06d}.png")
                    rgb_overlay_filename = os.path.join(traj_dir, f"rgb_overlay_{frame_idx:06d}.png")
                    depth_filename = os.path.join(traj_dir, f"depth_{frame_idx:06d}.npy")
                    
                    
                    # cv2.imwrite(rgb_filename, rgb_image)
                    # 保存带检测可视化标注的RGB图（center + contour）
                    rgb_to_save = rgb_image.copy()

                    # 在带标注图上也绘制 center 有效边框（仅用于center合法性判断）
                    h_ov, w_ov = rgb_to_save.shape[:2]
                    border_ov = int(max(0, min(self.center_border_pixels, h_ov // 2, w_ov // 2)))
                    if border_ov > 0:
                        cv2.rectangle(
                            rgb_to_save,
                            (border_ov, border_ov),
                            (w_ov - border_ov - 1, h_ov - border_ov - 1),
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            rgb_to_save,
                            f"center valid region (border={border_ov}px)",
                            (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )

                    if detection_results:
                        for det_idx, (_, det_info, _) in enumerate(detection_results):
                            if det_info is None:
                                continue

                            # 轮廓（绿色）
                            if 'contour' in det_info and det_info['contour'] is not None:
                                cv2.drawContours(rgb_to_save, [det_info['contour']], -1, (0, 255, 0), 2)

                            # 中心点（红色）和索引
                            if 'center' in det_info and det_info['center'] is not None:
                                center = det_info['center']
                                cv2.circle(rgb_to_save, center, 5, (0, 0, 255), -1)
                                cv2.putText(
                                    rgb_to_save,
                                    f"Det: {det_idx}",
                                    (center[0] + 8, center[1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 255),
                                    1,
                                )

                    # 叠加对应tracker的KF速度信息
                    # 1) 若当前tracker有匹配检测，则在该检测center附近显示速度
                    # 2) 同时在左上角显示一行速度文本，便于无检测时查看
                    if kf_data.get('velocity') is not None:
                        kf_vel = np.asarray(kf_data['velocity'], dtype=float)
                        kf_speed = float(np.linalg.norm(kf_vel))

                        # 左上角速度文本（按tracker区分行号）
                        text_y = 24 + 22 * int(tracker_id)
                        cv2.putText(
                            rgb_to_save,
                            f"KF{tracker_id} vel: [{kf_vel[0]:.2f}, {kf_vel[1]:.2f}, {kf_vel[2]:.2f}] m/s | |v|={kf_speed:.2f}",
                            (10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                        )

                        # 匹配检测点附近显示速度标注
                        matched_det_idx = detection_assignments.get(tracker_id, None)
                        if (
                            matched_det_idx is not None
                            and 0 <= matched_det_idx < len(detection_results)
                            and detection_results[matched_det_idx][1] is not None
                            and 'center' in detection_results[matched_det_idx][1]
                            and detection_results[matched_det_idx][1]['center'] is not None
                        ):
                            center = detection_results[matched_det_idx][1]['center']
                            cv2.putText(
                                rgb_to_save,
                                f"KF{tracker_id} |v|={kf_speed:.2f}m/s",
                                (center[0] + 8, center[1] + 16),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                            )

                    # 叠加不确定性归一化创新（vx, vy, vz）
                    nu_xyz = kf_data.get('normalized_innovation', None)
                    if nu_xyz is not None:
                        nu_xyz = np.asarray(nu_xyz, dtype=float).reshape(3)
                        nu_text_y = 24 + 22 * int(tracker_id) + 18
                        cv2.putText(
                            rgb_to_save,
                            f"nu[{tracker_id}]=[{nu_xyz[0]:.2f}, {nu_xyz[1]:.2f}, {nu_xyz[2]:.2f}]",
                            (10, nu_text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 200, 255),
                            1,
                        )

                    # 保留原始图（沿用现有命名）
                    cv2.imwrite(rgb_filename, rgb_image)
                    # 另存一张带标注图
                    cv2.imwrite(rgb_overlay_filename, rgb_to_save)
                    np.save(depth_filename, depth_image)
                    
                    # 确定KF状态：update或predict
                    # 基于actually_updated（是否真正执行了update操作）
                    kf_state = 'update' if actually_updated.get(tracker_id, False) else 'predict'
                    
                    frame_data = {
                        'frame_index': frame_idx,
                        'timestamp': current_time,
                        'relative_time': current_time - self.trajectory_start_time[tracker_id],
                        'body_pos': base_pos.tolist(),
                        'body_rot': base_rot.tolist(),
                        'detection_pos': detection_pos,
                        'contour_area': contour_area,
                        'kf_pos': kf_data['position'].tolist() if kf_data['position'] is not None else None,
                        'kf_vel': kf_data['velocity'].tolist() if kf_data['velocity'] is not None else None,
                        'kf_pos_var': kf_data['kf_pos_var'].tolist() if kf_data.get('kf_pos_var') is not None else None,
                        'kf_vel_var': kf_data['kf_vel_var'].tolist() if kf_data.get('kf_vel_var') is not None else None,
                        'innovation_r': kf_data['innovation_r'].tolist() if kf_data.get('innovation_r') is not None else None,
                        'innovation_S_diag': kf_data['innovation_S_diag'].tolist() if kf_data.get('innovation_S_diag') is not None else None,
                        'normalized_innovation': kf_data['normalized_innovation'].tolist() if kf_data.get('normalized_innovation') is not None else None,
                        'innovation_mahalanobis2': float(kf_data['innovation_mahalanobis2']) if kf_data.get('innovation_mahalanobis2') is not None else None,
                        'has_detection': has_detection.get(tracker_id, False),
                        'kf_state': kf_state,  # 'update' 或 'predict'
                        'rgb_file': f"rgb_{frame_idx:06d}.png",
                        'rgb_overlay_file': f"rgb_overlay_{frame_idx:06d}.png",
                        'depth_file': f"depth_{frame_idx:06d}.npy"
                    }

                    # 仅在 g 作为状态参与估计时记录重力状态
                    g_enabled = bool(kf_data.get('gravity_state_enabled', False))
                    g_value = kf_data.get('gravity', None)
                    if g_enabled and g_value is not None:
                        frame_data['kf_g'] = float(g_value)

                    self.trajectory_data[tracker_id].append(frame_data)
                    
                    # 定期打印记录状态
                    if len(self.trajectory_data[tracker_id]) % 30 == 0:
                        self.get_logger().info(
                            f"Tracker {tracker_id}: recorded {len(self.trajectory_data[tracker_id])} frames, "
                            f"latest files: rgb_{frame_idx:06d}.png, rgb_overlay_{frame_idx:06d}.png, depth_{frame_idx:06d}.npy"
                        )
    
    def save_trajectory(self, tracker_id):
        """
        保存单个tracker的轨迹数据到文件，并重命名临时图像目录
        
        Args:
            tracker_id: 追踪器ID
        """
        import json
        import shutil
        
        if tracker_id not in self.trajectory_data or len(self.trajectory_data[tracker_id]) == 0:
            return
        
        # 确保保存目录存在
        os.makedirs(self.trajectory_save_dir, exist_ok=True)
        
        # 生成文件名
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_name = f"trajectory_tracker{tracker_id}_{timestamp_str}_{self.trajectory_counter:04d}"
        json_filename = f"{traj_name}.json"
        json_filepath = os.path.join(self.trajectory_save_dir, json_filename)
        
        # 重命名临时图像目录
        temp_dir = os.path.join(self.trajectory_save_dir, f"tracker{tracker_id}_temp")
        final_dir = os.path.join(self.trajectory_save_dir, traj_name)
        if os.path.exists(temp_dir):
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            shutil.move(temp_dir, final_dir)
        
        # 保存数据
        trajectory_info = {
            'tracker_id': tracker_id,
            'start_timestamp': self.trajectory_start_time[tracker_id],
            'frame_count': len(self.trajectory_data[tracker_id]),
            'dt': self.dt_dynamic if self.dt_dynamic is not None else self.dt,
            'ground_z_threshold': self.ground_z_threshold,
            'coord_frame': self.trajectory_coord_frame,
            'image_dir': traj_name,
            'frames': self.trajectory_data[tracker_id]
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(trajectory_info, f, indent=2)
        
        self.get_logger().info(
            f"Saved trajectory for tracker {tracker_id}: json={json_filepath}, "
            f"image_dir={final_dir} ({len(self.trajectory_data[tracker_id])} frames)"
        )
        
        # 增加计数器
        self.trajectory_counter += 1
        
        # 清空数据
        if tracker_id in self.trajectory_data:
            del self.trajectory_data[tracker_id]
        if tracker_id in self.trajectory_start_time:
            del self.trajectory_start_time[tracker_id]

    def save_all_active_trajectories(self):
        """退出前强制保存所有仍在记录中的轨迹（即使还没落地）。"""
        if not self.enable_trajectory_recording:
            return
        active_ids = [tid for tid, frames in self.trajectory_data.items() if len(frames) > 0]
        for tid in active_ids:
            self.save_trajectory(tid)
    
    def generate_visualization_images(self, rgb_bgr, depth_array, detection_results):
        """
        生成带标注的可视化图像
        
        Args:
            rgb_bgr: BGR 格式的 RGB 图像
            depth_array: 深度图像
            detection_results: 检测结果
        """
        # === RGB 可视化 ===
        rgb_vis = rgb_bgr.copy()

        # 绘制 center 无效边框（仅用于center合法性判断，不屏蔽检测）
        h_rgb, w_rgb = rgb_vis.shape[:2]
        border = int(max(0, min(self.center_border_pixels, h_rgb // 2, w_rgb // 2)))
        if border > 0:
            cv2.rectangle(rgb_vis, (border, border), (w_rgb - border - 1, h_rgb - border - 1), (0, 255, 255), 2)
            cv2.putText(rgb_vis, f"center valid region (border={border}px)",
                        (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 绘制检测结果
        # detection_results 是 [(pos_body, det_info, ray_info), ...] 格式（body坐标系）
        if detection_results:
            for idx, (pos_world, det_info, ray_info) in enumerate(detection_results):
                if det_info is not None:
                    # 绘制轮廓（绿色）
                    if 'contour' in det_info:
                        cv2.drawContours(rgb_vis, [det_info['contour']], -1, (0, 255, 0), 2)
                    
                    # 绘制中心点（红色）
                    if 'center' in det_info:
                        center = det_info['center']
                        cv2.circle(rgb_vis, center, 5, (0, 0, 255), -1)
                        
                        # 显示索引
                        cv2.putText(rgb_vis, f"Det: {idx}", 
                                  (center[0] + 10, center[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # 绘制十字线
                    if 'vert_line' in det_info and det_info['vert_line']:
                        cv2.line(rgb_vis, det_info['vert_line'][0], det_info['vert_line'][1], 
                               (255, 255, 0), 1)
                    if 'hori_line' in det_info and det_info['hori_line']:
                        cv2.line(rgb_vis, det_info['hori_line'][0], det_info['hori_line'][1], 
                               (255, 0, 255), 1)

        
        # 绘制追踪状态信息
        y_offset = 30
        for tracker_id in range(self.num_balls):
            if self.ball_tracker.is_active(tracker_id):
                state = self.ball_tracker.get_state(tracker_id)
                if state:
                    pos = state['position']
                    vel = state['velocity']
                    status = "Active" if self.ball_tracker.is_validated(tracker_id) else "Validating"
                    
                    text = f"Ball {tracker_id}: {status} | Pos:[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]"
                    cv2.putText(rgb_vis, text, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y_offset += 20
        
        # 显示 catch_info
        catch_info = self.ball_tracker.catch_info_from_kf_obs_body(self.kf_obs_body)
        if catch_info:
            text = f"Catch: Pos:[{catch_info['position'][0]:.2f},{catch_info['position'][1]:.2f},{catch_info['position'][2]:.2f}]"
            cv2.putText(rgb_vis, text, (10, rgb_vis.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            text = f"Vel:[{catch_info['velocity'][0]:.2f},{catch_info['velocity'][1]:.2f},{catch_info['velocity'][2]:.2f}]"
            cv2.putText(rgb_vis, text, (10, rgb_vis.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 显示 FPS
        display_fps = (1.0 / self.dt_dynamic) if (self.dt_dynamic is not None and self.dt_dynamic > 0.0) else (1.0 / self.dt)
        cv2.putText(rgb_vis, f"FPS: {display_fps:.1f}", (rgb_vis.shape[1] - 110, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 旋转图像（顺时针90度）
        # rgb_vis = cv2.rotate(rgb_vis, cv2.ROTATE_90_CLOCKWISE)
        
        self.latest_rgb_vis = rgb_vis
        
        # === 深度可视化 ===
        # 生成伪彩色深度图
        valid_mask = np.isfinite(depth_array) & (depth_array > 0)
        if np.any(valid_mask):
            valid_depth = depth_array[valid_mask]
            min_depth = np.percentile(valid_depth, 1)
            max_depth = np.percentile(valid_depth, 99)
            depth_clipped = np.clip(depth_array, min_depth, max_depth)
            depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros(depth_array.shape, dtype=np.uint8)
        
        depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 在深度可视化上也画出同一 center 有效区域边框，便于对齐观察
        h_dep, w_dep = depth_vis.shape[:2]
        border_dep = int(max(0, min(self.center_border_pixels, h_dep // 2, w_dep // 2)))
        if border_dep > 0:
            cv2.rectangle(depth_vis, (border_dep, border_dep), (w_dep - border_dep - 1, h_dep - border_dep - 1), (0, 255, 255), 2)
        
        # 在深度图上绘制检测点
        # detection_results 是 [(pos_body, det_info, ray_info), ...] 格式（body坐标系）
        if detection_results:
            for idx, (pos_world, det_info, ray_info) in enumerate(detection_results):
                if det_info is not None and 'center' in det_info:
                    center = det_info['center']
                    cv2.circle(depth_vis, center, 5, (0, 0, 255), -1)
                    
                    # 显示深度值
                    cx, cy = center
                    if 0 <= cy < depth_array.shape[0] and 0 <= cx < depth_array.shape[1]:
                        depth_val = depth_array[cy, cx]
                        if np.isfinite(depth_val):
                            cv2.putText(depth_vis, f"{depth_val:.2f}m", 
                                      (cx + 10, cy - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 旋转图像（顺时针90度）
        # depth_vis = cv2.rotate(depth_vis, cv2.ROTATE_90_CLOCKWISE)
        
        self.latest_depth_vis = depth_vis
    
    def publish_ball_markers(self, catch_info):
        """发布球位置markers用于RViz可视化
        
        Args:
            catch_info: 接球信息字典，包含position和velocity
        """
        marker_array = MarkerArray()
        
        # 定义球的颜色
        colors = [
          
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # 绿色
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # 蓝色
        ]
        
        # for tracker_id in range(self.num_balls):
        #     # 只为已验证且活跃的追踪器创建marker
        #     if self.ball_tracker.is_validated(tracker_id) and self.ball_tracker.is_active(tracker_id):
        #         # 直接从kf_obs_body获取body坐标系下的观测
        #         obs_body = self.kf_obs_body[tracker_id]
        #         if obs_body is not None and obs_body['position'] is not None:
        #             # 位置已经在body坐标系中，直接使用
        #             pos_body = obs_body['position']
                    
        #             marker = Marker()
        #             marker.header.frame_id = "imu_link"  # 使用imu_link坐标系
        #             # 使用图像时间戳保证与相机数据时间对齐
        #             marker.header.stamp = self.latest_image_timestamp if self.latest_image_timestamp else self.get_clock().now().to_msg()
        #             marker.ns = "balls"
        #             marker.id = tracker_id
        #             marker.type = Marker.SPHERE
        #             marker.action = Marker.ADD
                    
        #             # 设置位置（body坐标系）
        #             marker.pose.position.x = float(pos_body[0])
        #             marker.pose.position.y = float(pos_body[1])
        #             marker.pose.position.z = float(pos_body[2])
        #             marker.pose.orientation.w = 1.0
                    
        #             # 设置球的大小（直径0.075米）
        #             marker.scale.x = 0.075
        #             marker.scale.y = 0.075
        #             marker.scale.z = 0.075
                    
        #             # 设置颜色
        #             marker.color = colors[tracker_id % len(colors)]
                    
        #             # marker生命周期
        #             marker.lifetime.sec = 0
        #             marker.lifetime.nanosec = 100000000  # 0.1秒
                    
        #             marker_array.markers.append(marker)
                    
        #             # 添加速度箭头marker
        #             if obs_body['velocity'] is not None:
        #                 vel_body = obs_body['velocity']
        #                 vel_magnitude = np.linalg.norm(vel_body)
                        
        #                 if vel_magnitude > 0.01:  # 只有速度足够大时才显示箭头
        #                     # 速度已经在body坐标系中，直接使用
        #                     arrow = Marker()
        #                     arrow.header.frame_id = "imu_link"
        #                     # 使用图像时间戳保证与相机数据时间对齐
        #                     arrow.header.stamp = self.latest_image_timestamp if self.latest_image_timestamp else self.get_clock().now().to_msg()
        #                     arrow.ns = "velocities"
        #                     arrow.id = tracker_id + 100  # 不同的ID避免冲突
        #                     arrow.type = Marker.ARROW
        #                     arrow.action = Marker.ADD
                            
        #                     # 箭头起点（球心，已经在body坐标系）
        #                     from geometry_msgs.msg import Point
        #                     arrow.points = []
        #                     start_point = Point()
        #                     start_point.x = float(pos_body[0])
        #                     start_point.y = float(pos_body[1])
        #                     start_point.z = float(pos_body[2])
        #                     arrow.points.append(start_point)
                            
        #                     # 箭头终点（速度方向，body坐标系）
        #                     end_point = Point()
        #                     end_point.x = start_point.x + vel_body[0] * 0.2  # 缩放因子0.2
        #                     end_point.y = start_point.y + vel_body[1] * 0.2
        #                     end_point.z = start_point.z + vel_body[2] * 0.2
        #                     arrow.points.append(end_point)
                            
        #                     # 箭头样式
        #                     arrow.scale.x = 0.01  # 箭杆直径
        #                     arrow.scale.y = 0.02  # 箭头宽度
        #                     arrow.scale.z = 0.03  # 箭头长度
                            
        #                     # 颜色（半透明）
        #                     arrow.color = ColorRGBA(
        #                         r=colors[tracker_id % len(colors)].r,
        #                         g=colors[tracker_id % len(colors)].g,
        #                         b=colors[tracker_id % len(colors)].b,
        #                         a=0.6
        #                     )
                            
        #                     arrow.lifetime.sec = 0
        #                     arrow.lifetime.nanosec = 100000000
                            
        #                     marker_array.markers.append(arrow)
        
        # === 添加检测位置markers（红色） ===
        if hasattr(self, 'latest_detection_results'):
            for det_idx, (pos_body, det_info, ray_info) in enumerate(self.latest_detection_results):
                # detection结果已经在body坐标系中，直接使用
                det_marker = Marker()
                det_marker.header.frame_id = "imu_link"
                det_marker.header.stamp = self.latest_image_timestamp if self.latest_image_timestamp else self.get_clock().now().to_msg()
                det_marker.ns = "detections"
                det_marker.id = det_idx + 200  # 不同的ID避免冲突
                det_marker.type = Marker.SPHERE
                det_marker.action = Marker.ADD
                
                # 设置位置（body坐标系）
                det_marker.pose.position.x = float(pos_body[0])
                det_marker.pose.position.y = float(pos_body[1])
                det_marker.pose.position.z = float(pos_body[2])
                det_marker.pose.orientation.w = 1.0
                
                # 设置球的大小（真实球半径0.0375米，直径0.075米）
                det_marker.scale.x = 0.075
                det_marker.scale.y = 0.075
                det_marker.scale.z = 0.075
                
                # 红色，半透明
                det_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)
                
                # marker生命周期
                det_marker.lifetime.sec = 0
                det_marker.lifetime.nanosec = 100000000  # 0.1秒
                
                marker_array.markers.append(det_marker)
                
                # === 添加射线marker（青色箭头） ===
                if ray_info is not None and 'ray_direction' in ray_info:
                    from geometry_msgs.msg import Point
                    ray_marker = Marker()
                    ray_marker.header.frame_id = "zed_left_camera_optical_frame"  # 基于相机坐标系
                    ray_marker.header.stamp = self.latest_image_timestamp if self.latest_image_timestamp else self.get_clock().now().to_msg()
                    ray_marker.ns = "rays"
                    ray_marker.id = det_idx + 300  # 不同的ID避免冲突
                    ray_marker.type = Marker.ARROW
                    ray_marker.action = Marker.ADD
                    
                    # 箭头起点（相机原点）
                    start_point = Point()
                    start_point.x = 0.0
                    start_point.y = 0.0
                    start_point.z = 0.0
                    
                    # 箭头终点（射线方向，缩放到合适长度）
                    ray_dir = ray_info['ray_direction']
                    ray_length = ray_info['actual_ray_length']
                    end_point = Point()
                    end_point.x = float(ray_dir[0] * ray_length)
                    end_point.y = float(ray_dir[1] * ray_length)
                    end_point.z = float(ray_dir[2] * ray_length)
                    
                    ray_marker.points = [start_point, end_point]
                    
                    # 箭头样式
                    ray_marker.scale.x = 0.005  # 箭杆直径
                    ray_marker.scale.y = 0.01   # 箭头宽度
                    ray_marker.scale.z = 0.015  # 箭头长度
                    
                    # 青色
                    ray_marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.8)
                    
                    # marker生命周期
                    ray_marker.lifetime.sec = 0
                    ray_marker.lifetime.nanosec = 100000000  # 0.1秒
                    
                    marker_array.markers.append(ray_marker)
        
        # === 添加相机位置marker（黄色球体） ===
        # 将相机世界坐标转换为imu_link坐标系
        cam_pos_imu = self.latest_base_rot.T @ (self.latest_cam_pos - self.latest_base_pos)
        
        cam_marker = Marker()
        cam_marker.header.frame_id = "imu_link"
        cam_marker.header.stamp = self.latest_image_timestamp if self.latest_image_timestamp else self.get_clock().now().to_msg()
        cam_marker.ns = "camera"
        cam_marker.id = 999
        cam_marker.type = Marker.SPHERE
        cam_marker.action = Marker.ADD
        
        # 设置相机位置（已转换到imu_link坐标系）
        cam_marker.pose.position.x = float(cam_pos_imu[0])
        cam_marker.pose.position.y = float(cam_pos_imu[1])
        cam_marker.pose.position.z = float(cam_pos_imu[2])
        cam_marker.pose.orientation.w = 1.0
        
        # 设置大小（小球）
        cam_marker.scale.x = 0.04
        cam_marker.scale.y = 0.04
        cam_marker.scale.z = 0.04
        
        # 黄色
        cam_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)
        
        # marker生命周期
        cam_marker.lifetime.sec = 0
        cam_marker.lifetime.nanosec = 100000000  # 0.1秒
        
        marker_array.markers.append(cam_marker)
        
        # # === 添加catch_info marker（洋红色球体，更大） ===
        # if catch_info is not None and catch_info['position'] is not None:
        #     # catch_info已经在body坐标系中，直接使用
        #     catch_pos = catch_info['position']
            
        #     catch_marker = Marker()
        #     catch_marker.header.frame_id = "imu_link"
        #     catch_marker.header.stamp = self.latest_image_timestamp if self.latest_image_timestamp else self.get_clock().now().to_msg()
        #     catch_marker.ns = "catch_ball"
        #     catch_marker.id = 1000
        #     catch_marker.type = Marker.SPHERE
        #     catch_marker.action = Marker.ADD
            
        #     # 设置位置（body坐标系）
        #     catch_marker.pose.position.x = float(catch_pos[0])
        #     catch_marker.pose.position.y = float(catch_pos[1])
        #     catch_marker.pose.position.z = float(catch_pos[2])
        #     catch_marker.pose.orientation.w = 1.0
            
        #     # 设置球的大小（稍大，直径0.10米）
        #     catch_marker.scale.x = 0.10
        #     catch_marker.scale.y = 0.10
        #     catch_marker.scale.z = 0.10
            
        #     # 洋红色（品红色），半透明
        #     catch_marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.8)
            
        #     # marker生命周期
        #     catch_marker.lifetime.sec = 0
        #     catch_marker.lifetime.nanosec = 100000000  # 0.1秒
            
        #     marker_array.markers.append(catch_marker)
            
        #     # 添加catch_info速度箭头（洋红色）
        #     if catch_info['velocity'] is not None:
        #         catch_vel = catch_info['velocity']
        #         catch_vel_magnitude = np.linalg.norm(catch_vel)

        #         if catch_vel_magnitude > 0.01:
        #             # catch_info速度已经在body坐标系中，直接使用
        #             catch_arrow = Marker()
        #             catch_arrow.header.frame_id = "imu_link"
        #             catch_arrow.header.stamp = self.latest_image_timestamp if self.latest_image_timestamp else self.get_clock().now().to_msg()
        #             catch_arrow.ns = "catch_velocity"
        #             catch_arrow.id = 1001
        #             catch_arrow.type = Marker.ARROW
        #             catch_arrow.action = Marker.ADD
                    
        #             # 箭头起点
        #             from geometry_msgs.msg import Point
        #             arrow_start = Point()
        #             arrow_start.x = float(catch_pos[0])
        #             arrow_start.y = float(catch_pos[1])
        #             arrow_start.z = float(catch_pos[2])
                    
        #             # 箭头终点
        #             arrow_end = Point()
        #             arrow_end.x = arrow_start.x + catch_vel[0] * 0.3  # 缩放因子0.3
        #             arrow_end.y = arrow_start.y + catch_vel[1] * 0.3
        #             arrow_end.z = arrow_start.z + catch_vel[2] * 0.3
                    
        #             catch_arrow.points = [arrow_start, arrow_end]
                    
        #             # 箭头样式（更粗）
        #             catch_arrow.scale.x = 0.015  # 箭杆直径
        #             catch_arrow.scale.y = 0.025  # 箭头宽度
        #             catch_arrow.scale.z = 0.035  # 箭头长度
                    
        #             # 洋红色
        #             catch_arrow.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9)
                    
        #             catch_arrow.lifetime.sec = 0
        #             catch_arrow.lifetime.nanosec = 100000000
                    
        #             marker_array.markers.append(catch_arrow)
        
        # 发布marker array
        self.marker_pub.publish(marker_array)
    
    def update_visualization(self):
        """更新可视化窗口"""
        # 更新matplotlib 3D可视化
        if hasattr(self, 'latest_has_detection') and self.enable_3D_visualization:
            self.visualizer.update_visualization(self.ball_tracker, self.latest_has_detection)
        
        # 更新2D图像可视化
        if self.latest_rgb_vis is not None:
            cv2.imshow('Ball Tracking - RGB', self.latest_rgb_vis)
        
        if self.latest_depth_vis is not None:
            cv2.imshow('Ball Tracking - Depth', self.latest_depth_vis)
        
        # 检查键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("退出程序...")
            self.save_all_active_trajectories()
            cv2.destroyAllWindows()
            if rclpy.ok():
                rclpy.shutdown()



def main(args=None):
    rclpy.init(args=args)
    
    # 创建球追踪节点
    node = BallTrackingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C 或异常退出时，确保把未落地但已记录的轨迹也落盘
        node.save_all_active_trajectories()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()




if __name__ == '__main__':
    # 默认启动球追踪节点
    # 如果需要图像保存功能，可以调用 main_saver()
    main()











class MultiRedBallDetector():
    """原有的图像保存节点（保留用于调试）"""
    """多红色球体检测器 - 同时检测多个球"""
    
    def __init__(self):
        """初始化检测器参数"""
        # HSV颜色范围 - 红色（降低饱和度下限以检测反光区域）
        # HSV说明：
        #   H (Hue/色调): 0-180，红色在0-10和165-180两个区间
        #   S (Saturation/饱和度): 0-255，0为白色/灰色，255为纯色
        #   V (Value/亮度): 0-255，0为黑色，255为最亮
        
        # 红色区间1: H=0-10（偏橙红色）
        # S下限=5: 允许检测低饱和度区域（反光处饱和度低）
        # V下限=20: 排除过暗区域
        self.lower_red1 = np.array([0, 5, 20])      
        self.upper_red1 = np.array([10, 255, 255])
        
        # 红色区间2: H=165-180（偏紫红色）
        self.lower_red2 = np.array([165, 5, 20])    
        self.upper_red2 = np.array([180, 255, 255])
        
        # # 形态学操作的核
        # self.kernel = np.ones((7, 7), np.uint8)
        
        # # 球体形状约束
        # self.min_area = 2000
        # self.max_area = 15000
        # self.min_circularity = 0.5
        # self.min_radius = 20
        # self.max_radius = 120
       
        # self.lower_red1 = np.array([0, 65, 65])
        # self.upper_red1 = np.array([8, 255, 255])
        # self.lower_red2 = np.array([170, 100, 100])
        # self.upper_red2 = np.array([180, 255, 255])
        
        # 形态学操作的核
        self.kernel = np.ones((5, 5), np.uint8)
        
        # 球体形状约束（放宽限制）
        self.min_area = 200      # 最小面积（像素）- 降低
        self.max_area = 20000     # 最大面积（像素）- 提高
        self.min_circularity = 0.3 # 最小圆度 (0-1, 1为完美圆形) - 降低到0.3
        self.min_radius = 15       # 最小半径 - 降低
        self.max_radius = 120     # 最大半径 - 提高
    
    def detect_all(self, image):
        """
        检测图像中的所有红色球体
        
        Args:
            image: BGR图像
            
        Returns:
            list of detections
        """
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作去除噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # 查找所有轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 在原图上绘制所有轮廓用于调试
        debug_image = image.copy()
        cv2.drawContours(debug_image, contours, -1, (255, 0, 255), 2)
        cv2.imshow('All Contours', debug_image)
        
        if not contours:
            return []
        
        # 收集所有符合条件的球体
        detections = []
        
        for contour in contours:
            # 面积过滤
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # 计算最小外接圆
            (_, _), radius = cv2.minEnclosingCircle(contour)
            
            # 半径过滤
            if radius < self.min_radius or radius > self.max_radius:
                continue
            
            # 圆度检测
            circle_area = np.pi * radius * radius
            circularity = area / circle_area if circle_area > 0 else 0
            
            if circularity < self.min_circularity:
                continue
            
            # 紧凑度
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
            else:
                compactness = 0
            
            # 综合评分
            area_score = 1.0 - abs(area - 300) / 10000
            area_score = max(0, area_score)
            score = circularity * 0.6 + compactness * 0.2 + area_score * 0.2
            
            # 如果符合条件，添加到检测结果
            if score > 0.3 and circularity > self.min_circularity:
                # 找边界点
                contour_points = contour.reshape(-1, 2)
                
                leftmost_idx = contour_points[:, 0].argmin()
                leftmost = tuple(contour_points[leftmost_idx])
                
                rightmost_idx = contour_points[:, 0].argmax()
                rightmost = tuple(contour_points[rightmost_idx])
                
                topmost_idx = contour_points[:, 1].argmin()
                topmost = tuple(contour_points[topmost_idx])
                
                bottommost_idx = contour_points[:, 1].argmax()
                bottommost = tuple(contour_points[bottommost_idx])
                
                hori_line = (leftmost, rightmost)
                vert_line = (topmost, bottommost)
                
                # 使用轮廓中心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)
                    intersection = center
                else:
                    continue
                
                # 添加检测结果
                detections.append({
                    'center': center,
                    'radius': int(radius),
                    'contour': contour,
                    'vert_line': vert_line,
                    'hori_line': hori_line,
                    'intersection': intersection,
                    'score': score,
                    'area': area
                })
        
        return detections
