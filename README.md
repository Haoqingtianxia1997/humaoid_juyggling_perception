# ZED Ball Tracking System

基于ZED相机的球体追踪系统，使用卡尔曼滤波进行3D轨迹预测。

## 系统概述

本系统实现了实时球体检测、追踪和轨迹记录功能，主要组件包括：
- **实时追踪节点**：订阅ZED相机的RGB和深度图像，检测并追踪球体运动
- **轨迹记录**：自动记录每个球从检测到落地的完整轨迹（检测位置 vs 卡尔曼滤波位置）
- **离线可视化**：对比分析检测位置和滤波位置的差异
- **数据采集工具**：保存图像数据用于离线分析和算法调试

---

## 文件说明

### 核心模块

#### `zed_tracker_deploy.py` - 主追踪节点
实时球体追踪ROS2节点

**功能：**
- 订阅ZED相机的RGB图像和深度图像
- 使用RobotClient获取IMU数据（机器人本体姿态）
- 通过TF2获取相机外参
- 多球追踪（支持最多2个球同时追踪）
- 基于卡尔曼滤波的3D轨迹预测
- 考虑重力和空气阻力的物理模型
- 发布可视化Marker（球位置、速度、预测轨迹）
- 发布catch_info话题（抓取信息）
- 自动记录轨迹数据（从球生到落地的完整轨迹）

**使用方法：**
```bash
# 1. 启动ZED相机节点
ros2 launch zed_wrapper zed2i.launch.py

# 2. 启动球追踪节点
cd ~/zed_ws/src/zed_subscriber
python3 zed_tracker_deploy.py
```

**参数调整：**

在代码中可调整的关键参数：
```python
# 卡尔曼滤波参数（第74-87行）
process_noise = 0.05          # 过程噪声，控制对预测的信任度
measurement_noise = 0.000001  # 测量噪声，控制对观测的信任度
drag_coefficient = 0.7        # 空气阻力系数

# 追踪参数
num_balls = 2                 # 最多追踪球数
dt = 1/60.0                  # 追踪频率（60Hz）
max_distance = 0.1           # 匹配最大距离阈值

# 地面高度（第95行）
ground_z_threshold = -0.144  # 米，z轴方向的地面高度
```

**输出：**
- **ROS话题：**
  - `/catch_info` (Float32MultiArray): 抓取信息 `[tracker_id, x, y, z, vx, vy, vz]`
  - `/ball_markers` (MarkerArray): 可视化Marker
  
- **轨迹文件：** `trajectory_data/trajectory_tracker{id}_{timestamp}_{counter}.json`
  - 每个球从检测到落地的完整轨迹
  - 包含每帧的检测位置、卡尔曼滤波位置和速度
  - 基于body坐标系（imu_link）
  - **同时保存每帧的RGB图像和深度数据**
  
- **图像数据：** `trajectory_data/trajectory_tracker{id}_{timestamp}_{counter}/`
  - `rgb_{frame_idx:06d}.png`: RGB图像
  - `depth_{frame_idx:06d}.npy`: 深度数据（NumPy格式）

**可视化Marker说明：**
- 🔴 红色球体：检测到的球位置（detection_pos）
- 🔵 蓝色球体：卡尔曼滤波估计的球位置（kf_pos）
- 🟢 绿色箭头：球速度向量
- 🟡 黄色线段：预测轨迹（0.5秒内的未来位置）

---

#### `visualize_trajectory_3d.py` - 3D轨迹可视化（Open3D）
使用Open3D逐帧可视化点云和球体追踪结果

**功能：**
- 加载轨迹JSON文件和对应的图像数据
- 从RGB和深度创建3D点云
- 显示检测位置（红色球体）和卡尔曼滤波位置（蓝色球体）
- 显示速度向量（绿色箭头）
- 显示body坐标系（imu_link）
- 关闭窗口后自动显示下一帧

**使用方法：**
```bash
# 可视化指定轨迹文件（所有帧）
python3 visualize_trajectory_3d.py trajectory_data/trajectory_tracker0_20260309_123456_0000.json

# 从第10帧开始可视化
python3 visualize_trajectory_3d.py trajectory_data/trajectory_tracker0_20260309_123456_0000.json --start 10

# 只可视化前20帧
python3 visualize_trajectory_3d.py trajectory_data/trajectory_tracker0_20260309_123456_0000.json --end 20

# 可视化第10-20帧
python3 visualize_trajectory_3d.py trajectory_data/trajectory_tracker0_20260309_123456_0000.json --start 10 --end 20
```

**参数：**
- `trajectory_file`: 轨迹JSON文件路径（必需）
- `--start`: 起始帧索引（默认0）
- `--end`: 结束帧索引（默认到最后一帧）

**可视化元素：**
- 点云：从RGB-D重建的3D场景
- 红色球体：检测位置（detection_pos）
- 蓝色球体：卡尔曼滤波位置（kf_pos）
- 绿色箭头：速度向量
- RGB坐标轴：body坐标系原点（imu_link）

**依赖安装：**
```bash
pip3 install open3d
```

#### `perception.py` - 感知核心模块
球体检测和追踪的核心算法

**主要类：**
- `CameraIntrinsics`: 相机内参管理
- `BallTracker`: 多球卡尔曼滤波追踪器
- `BallTrackingVisualizer`: 可视化工具

**算法特点：**
- 基于HSV颜色空间的球体检测
- 3D卡尔曼滤波（6维状态：位置+速度）
- 物理模型：考虑重力、空气阻力
- 匈牙利算法数据关联
- 遮挡处理和轨迹预测

**不需要单独运行**，作为库被 `zed_tracker_deploy.py` 调用。

---

### 数据采集工具

#### `zed_image_saver.py` - 图像数据采集
保存RGB和深度图像用于离线分析

**功能：**
- 同步采集RGB和深度图像
- 实时球体检测和标注
- 按时间戳命名保存
- 可选择是否显示检测结果

**使用方法：**
```bash
# 基础用法（保存到data/目录）
python3 zed_image_saver.py

# 指定保存路径
python3 zed_image_saver.py --data-root /path/to/save

# 显示检测可视化窗口
python3 zed_image_saver.py --display

# 组合使用
python3 zed_image_saver.py --data-root ../data --display
```

**参数：**
- `--data-root`: 数据保存根目录（默认：`../data`）
- `--display`: 是否显示实时检测结果窗口

**输出目录结构：**
```
data/
  rgb/              # RGB图像 (PNG)
  depth/            # 深度数据 (NPY)
  depth_image/      # 深度可视化图像 (PNG)
  processed_rgb/    # 标注后的RGB图像 (PNG)
```

---

#### `detection_process.py` - 离线批量检测
对已保存的图像数据进行批量球体检测和标注

**功能：**
- 批量处理RGB和深度图像
- 自动匹配时间戳最接近的图像对
- 在图像上标注检测到的球体
- 保存处理后的图像和检测统计

**使用方法：**
```bash
# 处理默认data/目录
python3 detection_process.py

# 指定数据目录
python3 detection_process.py --data-root /path/to/data

# 跳过已处理的图像
python3 detection_process.py --skip-existing
```

**参数：**
- `--data-root`: 数据根目录（默认：`../data`）
- `--skip-existing`: 跳过已存在的处理结果

**输出：**
- `processed_rgb/`: 标注后的图像（检测框、中心点、半径）
- `tracking_output/detection_stats.txt`: 检测统计（每张图检测到的球数）

---

### 可视化工具

#### `visualize_trajectory.py` - 轨迹对比可视化
对比分析检测位置和卡尔曼滤波位置的轨迹

**功能：**
- 加载JSON格式的轨迹数据
- 6子图对比分析：
  1. 3D轨迹对比
  2. XY平面投影（俯视图）
  3. XZ平面投影（侧视图）
  4. 位置误差随时间变化
  5. 高度随时间变化
  6. 速度分量随时间变化
- 支持单文件和批量处理

**使用方法：**
```bash
# 可视化单个轨迹文件
python3 visualize_trajectory.py trajectory_data/trajectory_tracker0_20260307_172510_0000.json

# 显示图像（不保存）
python3 visualize_trajectory.py trajectory_data/trajectory_tracker0_20260307_172510_0000.json --show

# 批量处理整个目录
python3 visualize_trajectory.py trajectory_data/

# 批量处理并显示
python3 visualize_trajectory.py trajectory_data/ --show

# 指定输出目录
python3 visualize_trajectory.py trajectory_data/ --output-dir results/
```

**参数：**
- `filepath`: 轨迹文件路径或目录路径
- `--show`: 显示图像窗口（默认只保存不显示）
- `--output-dir`: 保存图像的目录（默认：`trajectory_plots/`）

**输出：**
- `trajectory_plots/`: 每个轨迹的可视化图像（PNG）

**图表说明：**
- 🔴 红色：检测位置（detection）
- 🔵 蓝色：卡尔曼滤波位置（KF）
- 🟢 绿色：检测帧标记
- 灰色虚线：无检测时的插值

---

## 轨迹数据格式

轨迹JSON文件结构：
```json
{
  "tracker_id": 0,
  "start_timestamp": 1709812510.123456,
  "frame_count": 140,
  "dt": 0.016667,
  "ground_z_threshold": -0.144,
  "image_dir": "trajectory_tracker0_20260309_123456_0000",
  "frames": [
    {
      "frame_index": 0,
      "timestamp": 1709812510.123456,
      "relative_time": 0.0,
      "has_detection": true,
      "detection_pos": [0.234, -0.012, 0.456],
      "kf_pos": [0.234, -0.012, 0.456],
      "kf_vel": [0.123, -0.045, 0.678],
      "rgb_file": "rgb_000000.png",
      "depth_file": "depth_000000.npy"
    },
    ...
  ]
}
```

**字段说明：**
- `tracker_id`: 追踪器ID
- `start_timestamp`: 轨迹开始时间戳（Unix时间）
- `frame_count`: 总帧数
- `dt`: 时间步长（秒）
- `ground_z_threshold`: 地面高度阈值
- `image_dir`: 图像数据目录名
- `frames`: 每帧数据
  - `frame_index`: 帧索引
  - `timestamp`: 时间戳
  - `relative_time`: 相对时间（秒）
  - `has_detection`: 该帧是否有检测
  - `detection_pos`: 检测位置 [x, y, z] (body坐标系)
  - `kf_pos`: 卡尔曼滤波估计位置 [x, y, z]
  - `kf_vel`: 卡尔曼滤波估计速度 [vx, vy, vz]
  - `rgb_file`: RGB图像文件名
  - `depth_file`: 深度数据文件名

**坐标系：** 所有位置和速度基于 `imu_link`（body坐标系）

**目录结构：**
```
trajectory_data/
├── trajectory_tracker0_20260309_123456_0000.json
├── trajectory_tracker0_20260309_123456_0000/
│   ├── rgb_000000.png
│   ├── depth_000000.npy
│   ├── rgb_000001.png
│   ├── depth_000001.npy
│   └── ...
└── ...
```

---

## 完整工作流程

### 1. 实时追踪和数据采集

```bash
# Terminal 1: 启动ZED相机
ros2 launch zed_wrapper zed2i.launch.py

# Terminal 2: 启动球追踪节点
cd ~/zed_ws/src/zed_subscriber
python3 zed_tracker_deploy.py

# Terminal 3 (可选): 使用RViz可视化
rviz2
# 添加MarkerArray显示，订阅/ball_markers
```

进行球体投掷实验，系统会自动：
- 实时检测和追踪球体
- 发布可视化Marker和catch_info
- 当球落地时自动保存轨迹到 `trajectory_data/`

### 2. 轨迹分析

```bash
# 2D对比可视化（保存图像）
python3 visualize_trajectory.py trajectory_data/

# 或逐个查看（显示窗口）
python3 visualize_trajectory.py trajectory_data/trajectory_tracker0_20260309_123456_0000.json --show

# 3D点云可视化（逐帧显示）
python3 visualize_trajectory_3d.py trajectory_data/trajectory_tracker0_20260309_123456_0000.json
```

### 3. 图像数据采集（用于算法调试）

```bash
# 采集图像数据
python3 zed_image_saver.py --display

# 离线批量检测
python3 detection_process.py
```

---

## 依赖环境

### ROS2 包
```bash
sudo apt install ros-humble-cv-bridge
sudo apt install ros-humble-tf2-ros
sudo apt install ros-humble-visualization-msgs
```

### Python 包
```bash
pip3 install numpy opencv-python scipy matplotlib open3d
```

### 机器人接口
需要安装 `robot_bridge_py` 包（用于获取IMU数据）

---

## 常见问题

### 1. 检测不到球？
- 检查颜色阈值：修改 `perception.py` 中的HSV范围
- 调整形态学参数：`open_kernel`, `close_kernel` 大小
- 检查相机曝光和白平衡设置

### 2. 卡尔曼滤波不准？
- 调整 `process_noise` 和 `measurement_noise`
- `process_noise` 增大 → 更信任观测
- `measurement_noise` 减小 → 更信任观测

### 3. 轨迹没有保存？
- 检查 `trajectory_data/` 目录是否存在
- 查看终端输出：应有 "Started recording" 和 "Saved trajectory" 日志
- 确认球是否触发落地条件（z < ground_z_threshold）

### 4. 坐标系混淆？
- `detection_pos` / `kf_pos`: body坐标系（imu_link）
- Marker可视化：world坐标系
- 使用TF进行坐标转换

---

## 文件目录结构

```
zed_subscriber/
├── README.md                    # 本文档
├── zed_tracker_deploy.py        # 主追踪节点
├── perception.py                # 感知核心模块
├── detection_process.py         # 离线批量检测
├── zed_image_saver.py          # 图像数据采集
├── visualize_trajectory.py      # 2D轨迹可视化
├── visualize_trajectory_3d.py   # 3D点云可视化（Open3D）
├── trajectory_data/             # 轨迹数据目录
│   ├── trajectory_*.json        # 轨迹JSON文件
│   └── trajectory_*/            # 对应的图像数据
│       ├── rgb_*.png            # RGB图像
│       └── depth_*.npy          # 深度数据
└── trajectory_plots/            # 2D轨迹图像目录
    └── trajectory_*.png         # 可视化图像
```

---

## 参数速查表

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `process_noise` | zed_tracker_deploy.py:86 | 0.05 | 卡尔曼过程噪声 |
| `measurement_noise` | zed_tracker_deploy.py:87 | 0.000001 | 卡尔曼测量噪声 |
| `drag_coefficient` | zed_tracker_deploy.py:89 | 0.7 | 空气阻力系数 |
| `ground_z_threshold` | zed_tracker_deploy.py:95 | -0.144 | 地面高度阈值(m) |
| `num_balls` | zed_tracker_deploy.py:82 | 2 | 最大追踪球数 |
| `dt` | zed_tracker_deploy.py:83 | 1/60 | 追踪时间步长(s) |
| `max_distance` | zed_tracker_deploy.py:88 | 0.1 | 匹配距离阈值(m) |

---

## 版本信息

- **系统版本:** ROS2 Humble
- **Python版本:** 3.10+
- **ZED SDK:** 4.x
- **最后更新:** 2026-03-07

---

## 联系与贡献

如有问题或建议，请联系项目维护者。
