#!/usr/bin/env python3
"""
轨迹数据可视化脚本
对比显示detection位置和卡尔曼滤波位置的轨迹
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
from pathlib import Path


def load_trajectory(filepath):
    """加载轨迹数据文件"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def visualize_trajectory(trajectory_data, save_path=None):
    """
    可视化单个轨迹
    
    Args:
        trajectory_data: 轨迹数据字典
        save_path: 保存图像的路径（可选）
    """
    tracker_id = trajectory_data['tracker_id']
    frames = trajectory_data['frames']
    dt = trajectory_data['dt']
    
    # 提取数据
    times = []
    detection_positions = []
    kf_positions = []
    kf_velocities = []
    has_detections = []
    
    for frame in frames:
        times.append(frame['relative_time'])
        
        # Detection位置（可能为None）
        if frame['detection_pos'] is not None:
            detection_positions.append(frame['detection_pos'])
        else:
            detection_positions.append([None, None, None])
        
        # 卡尔曼滤波位置
        if frame['kf_pos'] is not None:
            kf_positions.append(frame['kf_pos'])
        else:
            kf_positions.append([None, None, None])
        
        # 卡尔曼滤波速度
        if frame['kf_vel'] is not None:
            kf_velocities.append(frame['kf_vel'])
        else:
            kf_velocities.append([None, None, None])
        
        has_detections.append(frame['has_detection'])
    
    # 转换为numpy数组
    times = np.array(times)
    detection_positions = np.array(detection_positions, dtype=float)
    kf_positions = np.array(kf_positions, dtype=float)
    kf_velocities = np.array(kf_velocities, dtype=float)
    has_detections = np.array(has_detections)
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # === 3D轨迹对比 ===
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # 绘制Detection轨迹（红色）
    det_mask = ~np.isnan(detection_positions[:, 0])
    if np.any(det_mask):
        det_pos = detection_positions[det_mask]
        ax1.plot(det_pos[:, 0], det_pos[:, 1], det_pos[:, 2], 
                'ro-', label='Detection', markersize=4, linewidth=2, alpha=0.7)
    
    # 绘制KF轨迹（蓝色）
    kf_mask = ~np.isnan(kf_positions[:, 0])
    if np.any(kf_mask):
        kf_pos = kf_positions[kf_mask]
        ax1.plot(kf_pos[:, 0], kf_pos[:, 1], kf_pos[:, 2], 
                'b.-', label='Kalman Filter', markersize=3, linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.set_title(f'Tracker {tracker_id}: 3D Trajectory Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # === XY平面投影 ===
    ax2 = fig.add_subplot(2, 3, 2)
    if np.any(det_mask):
        det_pos = detection_positions[det_mask]
        ax2.plot(det_pos[:, 0], det_pos[:, 1], 'ro-', label='Detection', markersize=4, alpha=0.7)
    if np.any(kf_mask):
        kf_pos = kf_positions[kf_mask]
        ax2.plot(kf_pos[:, 0], kf_pos[:, 1], 'b.-', label='Kalman Filter', markersize=3, alpha=0.8)
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_title('XY Plane Projection', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # === XZ平面投影 ===
    ax3 = fig.add_subplot(2, 3, 3)
    if np.any(det_mask):
        det_pos = detection_positions[det_mask]
        ax3.plot(det_pos[:, 0], det_pos[:, 2], 'ro-', label='Detection', markersize=4, alpha=0.7)
    if np.any(kf_mask):
        kf_pos = kf_positions[kf_mask]
        ax3.plot(kf_pos[:, 0], kf_pos[:, 2], 'b.-', label='Kalman Filter', markersize=3, alpha=0.8)
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Z (m)', fontsize=10)
    ax3.set_title('XZ Plane Projection', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # === 位置误差随时间变化 ===
    ax4 = fig.add_subplot(2, 3, 4)
    
    # 计算位置误差（只在有detection的帧）
    errors = []
    error_times = []
    for i, has_det in enumerate(has_detections):
        if has_det and not np.any(np.isnan(detection_positions[i])) and not np.any(np.isnan(kf_positions[i])):
            error = np.linalg.norm(detection_positions[i] - kf_positions[i])
            errors.append(error)
            error_times.append(times[i])
    
    if len(errors) > 0:
        ax4.plot(error_times, errors, 'g.-', linewidth=1.5, markersize=4)
        ax4.axhline(y=np.mean(errors), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(errors)*1000:.2f}mm')
        ax4.fill_between(error_times, 0, errors, alpha=0.3)
    
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Position Error (m)', fontsize=10)
    ax4.set_title('Detection vs KF Position Error', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # === Z坐标随时间变化 ===
    ax5 = fig.add_subplot(2, 3, 5)
    if np.any(det_mask):
        det_times = times[det_mask]
        det_z = detection_positions[det_mask, 2]
        ax5.plot(det_times, det_z, 'ro-', label='Detection Z', markersize=4, alpha=0.7)
    if np.any(kf_mask):
        kf_times = times[kf_mask]
        kf_z = kf_positions[kf_mask, 2]
        ax5.plot(kf_times, kf_z, 'b.-', label='KF Z', markersize=3, alpha=0.8)
    
    # 绘制地面高度线
    if 'ground_z_threshold' in trajectory_data:
        ax5.axhline(y=trajectory_data['ground_z_threshold'], color='k', 
                   linestyle='--', linewidth=2, label='Ground')
    
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Z Position (m)', fontsize=10)
    ax5.set_title('Height vs Time', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # === 速度随时间变化 ===
    ax6 = fig.add_subplot(2, 3, 6)
    vel_mask = ~np.isnan(kf_velocities[:, 0])
    if np.any(vel_mask):
        vel_times = times[vel_mask]
        vx = kf_velocities[vel_mask, 0]
        vy = kf_velocities[vel_mask, 1]
        vz = kf_velocities[vel_mask, 2]
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        
        ax6.plot(vel_times, vx, 'r.-', label='Vx', markersize=2, alpha=0.7)
        ax6.plot(vel_times, vy, 'g.-', label='Vy', markersize=2, alpha=0.7)
        ax6.plot(vel_times, vz, 'b.-', label='Vz', markersize=2, alpha=0.7)
        ax6.plot(vel_times, v_mag, 'k-', label='|V|', linewidth=2, alpha=0.8)
    
    ax6.set_xlabel('Time (s)', fontsize=10)
    ax6.set_ylabel('Velocity (m/s)', fontsize=10)
    ax6.set_title('KF Velocity Components', fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 添加总标题
    info_text = (f"Tracker {tracker_id} | Frames: {len(frames)} | "
                f"Duration: {times[-1]:.2f}s | dt: {dt:.4f}s")
    fig.suptitle(info_text, fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_directory(directory_path, output_dir=None):
    """
    可视化目录中的所有轨迹文件
    
    Args:
        directory_path: 轨迹数据目录
        output_dir: 输出图像目录（可选，默认为directory_path/visualizations）
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return
    
    # 获取所有JSON文件
    json_files = sorted(directory.glob("trajectory_*.json"))
    
    if len(json_files) == 0:
        print(f"No trajectory files found in {directory_path}")
        return
    
    print(f"Found {len(json_files)} trajectory files")
    
    # 创建输出目录
    if output_dir is None:
        output_dir = directory / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 处理每个文件
    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")
        try:
            trajectory_data = load_trajectory(json_file)
            
            # 生成输出文件名
            output_name = json_file.stem + ".png"
            output_path = output_dir / output_name
            
            # 可视化并保存
            visualize_trajectory(trajectory_data, save_path=output_path)
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    print(f"\nAll visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize ball tracking trajectory data')
    parser.add_argument('path', type=str, nargs='?', default=None,
                       help='Path to trajectory JSON file or directory containing trajectory files (default: trajectory_data/)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for visualization images (only for directory mode)')
    parser.add_argument('--show', '-s', action='store_true',
                       help='Show plots interactively instead of saving (only for single file mode)')
    
    args = parser.parse_args()
    
    # 如果未提供路径，默认使用trajectory_data目录
    if args.path is None:
        script_dir = Path(__file__).parent
        path = script_dir / 'trajectory_data'
        if not path.exists():
            print(f"Error: Default directory {path} does not exist")
            return
    else:
        path = Path(args.path)
    
    if path.is_file():
        # 处理单个文件
        print(f"Loading trajectory from {path}")
        trajectory_data = load_trajectory(path)
        
        if args.show:
            visualize_trajectory(trajectory_data, save_path=None)
        else:
            output_path = path.parent / (path.stem + ".png")
            visualize_trajectory(trajectory_data, save_path=output_path)
    
    elif path.is_dir():
        # 处理目录
        print(f"Processing directory {path}")
        visualize_directory(path, output_dir=args.output)
    
    else:
        print(f"Error: {path} is not a valid file or directory")


if __name__ == '__main__':
    main()
