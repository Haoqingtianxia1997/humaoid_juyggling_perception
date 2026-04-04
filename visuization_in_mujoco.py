from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class FrameData:
	frame_index: int
	timestamp: float
	base_pos: np.ndarray
	base_rot: np.ndarray
	detection_pos: np.ndarray | None
	gt_pos: np.ndarray | None
	kf_pos_body: np.ndarray | None
	kf_vel_body: np.ndarray | None
	kf_state: str
	has_detection: bool


@dataclass
class PredictSample:
	frame_index: int
	timestamp: float
	imu_world_pos: np.ndarray
	base_rot: np.ndarray
	kf_world_pos: np.ndarray
	kf_world_vel: np.ndarray
	fit_world_pos: np.ndarray
	fit_world_vel: np.ndarray
	gt_fit_world_pos: np.ndarray
	gt_fit_world_vel: np.ndarray
	sim_ball_world_pos: np.ndarray
	sim_ball_world_vel: np.ndarray
	pos_error_xyz: np.ndarray
	vel_error_xyz: np.ndarray
	fit_pos_error_xyz: np.ndarray
	fit_vel_error_xyz: np.ndarray


@dataclass
class TrajectoryData:
	name: str
	frames: list[FrameData]
	base_pos_ref: np.ndarray
	init_update_index: int
	predict_samples: list[PredictSample]
	fit_info: dict | None


def _compute_stats(values: np.ndarray) -> dict[str, float] | None:
	arr = np.asarray(values, dtype=float).reshape(-1)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return None
	abs_arr = np.abs(arr)
	return {
		"max": float(np.max(arr)),
		"min": float(np.min(arr)),
		"mean": float(np.mean(arr)),
		"std": float(np.std(arr)),
		"abs_mean": float(np.mean(abs_arr)),
		"abs_std": float(np.std(abs_arr)),
		"count": int(arr.size),
	}


def _collect_error_stats_for_trajectory(
	traj: TrajectoryData,
	error_source: str = "kf",
) -> dict[str, dict[str, float] | None]:
	if not traj.predict_samples:
		return {k: None for k in ["x", "y", "z", "vx", "vy", "vz"]}

	if error_source == "kinematic":
		pos = np.asarray([s.fit_pos_error_xyz for s in traj.predict_samples], dtype=float)
		vel = np.asarray([s.fit_vel_error_xyz for s in traj.predict_samples], dtype=float)
	elif error_source == "kf_gt":
		pos = np.asarray([s.kf_world_pos - s.gt_fit_world_pos for s in traj.predict_samples], dtype=float)
		vel = np.asarray([s.kf_world_vel - s.gt_fit_world_vel for s in traj.predict_samples], dtype=float)
	elif error_source == "kinematic_gt":
		pos = np.asarray([s.fit_world_pos - s.gt_fit_world_pos for s in traj.predict_samples], dtype=float)
		vel = np.asarray([s.fit_world_vel - s.gt_fit_world_vel for s in traj.predict_samples], dtype=float)
	else:
		pos = np.asarray([s.pos_error_xyz for s in traj.predict_samples], dtype=float)
		vel = np.asarray([s.vel_error_xyz for s in traj.predict_samples], dtype=float)
	return {
		"x": _compute_stats(pos[:, 0]),
		"y": _compute_stats(pos[:, 1]),
		"z": _compute_stats(pos[:, 2]),
		"vx": _compute_stats(vel[:, 0]),
		"vy": _compute_stats(vel[:, 1]),
		"vz": _compute_stats(vel[:, 2]),
	}


def _collect_error_stats_all(
	trajectories: list[TrajectoryData],
	error_source: str = "kf",
) -> dict[str, dict[str, float] | None]:
	pos_all = []
	vel_all = []
	for traj in trajectories:
		if not traj.predict_samples:
			continue
		if error_source == "kinematic":
			pos_all.append(np.asarray([s.fit_pos_error_xyz for s in traj.predict_samples], dtype=float))
			vel_all.append(np.asarray([s.fit_vel_error_xyz for s in traj.predict_samples], dtype=float))
		elif error_source == "kf_gt":
			pos_all.append(np.asarray([s.kf_world_pos - s.gt_fit_world_pos for s in traj.predict_samples], dtype=float))
			vel_all.append(np.asarray([s.kf_world_vel - s.gt_fit_world_vel for s in traj.predict_samples], dtype=float))
		elif error_source == "kinematic_gt":
			pos_all.append(np.asarray([s.fit_world_pos - s.gt_fit_world_pos for s in traj.predict_samples], dtype=float))
			vel_all.append(np.asarray([s.fit_world_vel - s.gt_fit_world_vel for s in traj.predict_samples], dtype=float))
		else:
			pos_all.append(np.asarray([s.pos_error_xyz for s in traj.predict_samples], dtype=float))
			vel_all.append(np.asarray([s.vel_error_xyz for s in traj.predict_samples], dtype=float))

	if not pos_all or not vel_all:
		return {k: None for k in ["x", "y", "z", "vx", "vy", "vz"]}

	pos_cat = np.concatenate(pos_all, axis=0)
	vel_cat = np.concatenate(vel_all, axis=0)
	return {
		"x": _compute_stats(pos_cat[:, 0]),
		"y": _compute_stats(pos_cat[:, 1]),
		"z": _compute_stats(pos_cat[:, 2]),
		"vx": _compute_stats(vel_cat[:, 0]),
		"vy": _compute_stats(vel_cat[:, 1]),
		"vz": _compute_stats(vel_cat[:, 2]),
	}


def _print_error_stats(title: str, stats: dict[str, dict[str, float] | None]) -> None:
	print(f"\n===== {title} =====")
	for key in ["x", "y", "z", "vx", "vy", "vz"]:
		s = stats.get(key, None)
		if s is None:
			print(f"{key}: N/A")
			continue
		print(
			f"{key}: "
			f"max={s['max']:+.6f}, min={s['min']:+.6f}, "
			f"mean={s['mean']:+.6f}, std={s['std']:.6f}, "
			f"abs_mean={s['abs_mean']:.6f}, abs_std={s['abs_std']:.6f}, n={s['count']}"
		)


def _save_traj_error_plot(
	traj: TrajectoryData,
	output_dir: Path,
	show_plot: bool = False,
	error_source: str = "kf",
) -> Path | None:
	if not traj.predict_samples:
		return None

	ts = np.asarray([s.timestamp for s in traj.predict_samples], dtype=float)
	if error_source == "kinematic":
		pos_err = np.asarray([s.fit_pos_error_xyz for s in traj.predict_samples], dtype=float)
		vel_err = np.asarray([s.fit_vel_error_xyz for s in traj.predict_samples], dtype=float)
		error_title = "Kinematic - Sim"
	else:
		pos_err = np.asarray([s.pos_error_xyz for s in traj.predict_samples], dtype=float)
		vel_err = np.asarray([s.vel_error_xyz for s in traj.predict_samples], dtype=float)
		error_title = "KF - Sim"

	if ts.size == 0:
		return None

	# 使用相对时间方便阅读，但严格对应原时间戳顺序
	t0 = float(ts[0])
	t_rel = ts - t0

	fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
	keys = ["x", "y", "z", "vx", "vy", "vz"]
	labels = ["x err (m)", "y err (m)", "z err (m)", "vx err (m/s)", "vy err (m/s)", "vz err (m/s)"]
	series = [
		pos_err[:, 0],
		pos_err[:, 1],
		pos_err[:, 2],
		vel_err[:, 0],
		vel_err[:, 1],
		vel_err[:, 2],
	]

	for i, ax in enumerate(axes.reshape(-1)):
		y = np.asarray(series[i], dtype=float)
		valid = np.isfinite(t_rel) & np.isfinite(y)
		if np.any(valid):
			ax.plot(t_rel[valid], y[valid], color="tab:blue", linewidth=1.5)
			ax.scatter(t_rel[valid], y[valid], color="tab:blue", s=10, alpha=0.8)
		ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
		ax.set_title(keys[i])
		ax.set_ylabel(labels[i])
		ax.grid(True, alpha=0.3)

	axes[1, 0].set_xlabel("t (s), aligned to trajectory start timestamp")
	axes[1, 1].set_xlabel("t (s), aligned to trajectory start timestamp")
	axes[1, 2].set_xlabel("t (s), aligned to trajectory start timestamp")
	fig.suptitle(f"{traj.name} | error ({error_title}) vs timestamp")
	fig.tight_layout(rect=[0, 0, 1, 0.96])

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{Path(traj.name).stem}_error_xyz_vxyz_vs_time.png"
	fig.savefig(out_path, dpi=160)
	if show_plot:
		# 阻塞显示：关闭该窗口后，继续回到 MuJoCo 回放并可切换到下一条轨迹
		plt.show(block=True)
	plt.close(fig)
	return out_path


def _save_traj_state_plot(traj: TrajectoryData, output_dir: Path, show_plot: bool = False) -> Path | None:
	"""保存并可选展示轨迹状态对比图：x/y/z 与 vx/vy/vz（KF vs Sim）。"""
	if not traj.predict_samples:
		return None

	ts = np.asarray([s.timestamp for s in traj.predict_samples], dtype=float)
	if ts.size == 0:
		return None

	t_rel = ts - float(ts[0])
	kf_pos = np.asarray([s.kf_world_pos for s in traj.predict_samples], dtype=float)
	sim_pos = np.asarray([s.sim_ball_world_pos for s in traj.predict_samples], dtype=float)
	kf_vel = np.asarray([s.kf_world_vel for s in traj.predict_samples], dtype=float)
	sim_vel = np.asarray([s.sim_ball_world_vel for s in traj.predict_samples], dtype=float)
	fit_pos = np.asarray([s.fit_world_pos for s in traj.predict_samples], dtype=float)
	fit_vel = np.asarray([s.fit_world_vel for s in traj.predict_samples], dtype=float)
	gt_fit_pos = np.asarray([s.gt_fit_world_pos for s in traj.predict_samples], dtype=float)
	gt_fit_vel = np.asarray([s.gt_fit_world_vel for s in traj.predict_samples], dtype=float)

	# 一致性自检：确保误差定义没有线条混淆
	stored_pos_err = np.asarray([s.pos_error_xyz for s in traj.predict_samples], dtype=float)
	stored_vel_err = np.asarray([s.vel_error_xyz for s in traj.predict_samples], dtype=float)
	calc_pos_err = kf_pos - sim_pos
	calc_vel_err = kf_vel - sim_vel
	if np.nanmax(np.abs(stored_pos_err - calc_pos_err)) > 1e-6 or np.nanmax(np.abs(stored_vel_err - calc_vel_err)) > 1e-6:
		print(f"[warn] {traj.name}: stored error != (KF - Sim)，请检查数据来源")

	stored_fit_pos_err = np.asarray([s.fit_pos_error_xyz for s in traj.predict_samples], dtype=float)
	stored_fit_vel_err = np.asarray([s.fit_vel_error_xyz for s in traj.predict_samples], dtype=float)
	calc_fit_pos_err = fit_pos - sim_pos
	calc_fit_vel_err = fit_vel - sim_vel
	if np.nanmax(np.abs(stored_fit_pos_err - calc_fit_pos_err)) > 1e-6 or np.nanmax(np.abs(stored_fit_vel_err - calc_fit_vel_err)) > 1e-6:
		print(f"[warn] {traj.name}: stored fit_error != (Kinematic - Sim)，请检查数据来源")

	fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
	keys = ["x", "y", "z", "vx", "vy", "vz"]
	ylabels = ["x (m)", "y (m)", "z (m)", "vx (m/s)", "vy (m/s)", "vz (m/s)"]
	series_kf = [kf_pos[:, 0], kf_pos[:, 1], kf_pos[:, 2], kf_vel[:, 0], kf_vel[:, 1], kf_vel[:, 2]]
	series_sim = [sim_pos[:, 0], sim_pos[:, 1], sim_pos[:, 2], sim_vel[:, 0], sim_vel[:, 1], sim_vel[:, 2]]
	series_fit = [fit_pos[:, 0], fit_pos[:, 1], fit_pos[:, 2], fit_vel[:, 0], fit_vel[:, 1], fit_vel[:, 2]]
	series_gt_fit = [gt_fit_pos[:, 0], gt_fit_pos[:, 1], gt_fit_pos[:, 2], gt_fit_vel[:, 0], gt_fit_vel[:, 1], gt_fit_vel[:, 2]]

	for i, ax in enumerate(axes.reshape(-1)):
		y_kf = np.asarray(series_kf[i], dtype=float)
		y_sim = np.asarray(series_sim[i], dtype=float)
		y_fit = np.asarray(series_fit[i], dtype=float)
		y_gt_fit = np.asarray(series_gt_fit[i], dtype=float)
		valid_kf = np.isfinite(t_rel) & np.isfinite(y_kf)
		valid_sim = np.isfinite(t_rel) & np.isfinite(y_sim)
		valid_fit = np.isfinite(t_rel) & np.isfinite(y_fit)
		valid_gt_fit = np.isfinite(t_rel) & np.isfinite(y_gt_fit)
		if np.any(valid_kf):
			ax.plot(t_rel[valid_kf], y_kf[valid_kf], color="tab:blue", linewidth=1.6, label="KF")
			ax.scatter(t_rel[valid_kf], y_kf[valid_kf], color="tab:blue", s=10, alpha=0.7)
		if np.any(valid_sim):
			ax.plot(t_rel[valid_sim], y_sim[valid_sim], color="tab:green", linewidth=1.6, linestyle="--", label="Sim")
			ax.scatter(t_rel[valid_sim], y_sim[valid_sim], color="tab:green", s=10, alpha=0.7)
		if np.any(valid_fit):
			ax.plot(t_rel[valid_fit], y_fit[valid_fit], color="tab:orange", linewidth=1.6, linestyle="-.", label="Kinematic")
			ax.scatter(t_rel[valid_fit], y_fit[valid_fit], color="tab:orange", s=10, alpha=0.7)
		if np.any(valid_gt_fit):
			ax.plot(t_rel[valid_gt_fit], y_gt_fit[valid_gt_fit], color="tab:red", linewidth=1.6, linestyle=":", label="GT-Kinematic")
			ax.scatter(t_rel[valid_gt_fit], y_gt_fit[valid_gt_fit], color="tab:red", s=10, alpha=0.7)
		ax.set_title(keys[i])
		ax.set_ylabel(ylabels[i])
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=8)

	axes[1, 0].set_xlabel("t (s), aligned to trajectory start timestamp")
	axes[1, 1].set_xlabel("t (s), aligned to trajectory start timestamp")
	axes[1, 2].set_xlabel("t (s), aligned to trajectory start timestamp")
	fig.suptitle(f"{traj.name} | state vs timestamp  [Blue=KF, Green=Sim, Orange=Kinematic, Red=GT-Kinematic]")
	fig.tight_layout(rect=[0, 0, 1, 0.96])

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{Path(traj.name).stem}_state_xyz_vxyz_vs_time.png"
	fig.savefig(out_path, dpi=160)
	if show_plot:
		plt.show(block=True)
	plt.close(fig)
	return out_path


def resolve_model_path(user_path: str | None) -> Path:
	if user_path:
		p = Path(user_path).expanduser().resolve()
		if not p.exists():
			raise FileNotFoundError(f"模型文件不存在: {p}")
		return p
	default_model = Path(__file__).resolve().parent / "assets" / "mjcf" / "h1_juggling_camera.xml"
	if not default_model.exists():
		raise FileNotFoundError(f"默认模型文件不存在: {default_model}")
	return default_model


def resolve_trajectory_dir(user_path: str | None) -> Path:
	if user_path:
		p = Path(user_path).expanduser().resolve()
		if not p.exists():
			raise FileNotFoundError(f"轨迹目录不存在: {p}")
		return p
	default_dir = Path(__file__).resolve().parent / "trajectory_data"
	if not default_dir.exists():
		raise FileNotFoundError(f"默认轨迹目录不存在: {default_dir}")
	return default_dir


def rotmat_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
	m = np.asarray(r, dtype=float).reshape(3, 3)
	tr = np.trace(m)
	if tr > 0.0:
		s = np.sqrt(tr + 1.0) * 2.0
		w = 0.25 * s
		x = (m[2, 1] - m[1, 2]) / s
		y = (m[0, 2] - m[2, 0]) / s
		z = (m[1, 0] - m[0, 1]) / s
	else:
		if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
			s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
			w = (m[2, 1] - m[1, 2]) / s
			x = 0.25 * s
			y = (m[0, 1] + m[1, 0]) / s
			z = (m[0, 2] + m[2, 0]) / s
		elif m[1, 1] > m[2, 2]:
			s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
			w = (m[0, 2] - m[2, 0]) / s
			x = (m[0, 1] + m[1, 0]) / s
			y = 0.25 * s
			z = (m[1, 2] + m[2, 1]) / s
		else:
			s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
			w = (m[1, 0] - m[0, 1]) / s
			x = (m[0, 2] + m[2, 0]) / s
			y = (m[1, 2] + m[2, 1]) / s
			z = 0.25 * s
	q = np.array([w, x, y, z], dtype=float)
	q /= np.linalg.norm(q) + 1e-12
	return q


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
	w, x, y, z = np.asarray(q, dtype=float).reshape(4)
	return np.array(
		[
			[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
			[2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
			[2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
		],
		dtype=float,
	)


def body_to_world(base_pos: np.ndarray, base_rot: np.ndarray, p_body: np.ndarray) -> np.ndarray:
	return base_rot @ p_body + base_pos


def vel_body_to_world(base_rot: np.ndarray, v_body: np.ndarray) -> np.ndarray:
	return base_rot @ v_body


def map_base_pos_to_imu_world(base_pos: np.ndarray, base_pos_ref: np.ndarray, imu_pos_ref: np.ndarray) -> np.ndarray:
	"""将轨迹中的 base_pos(imu位置，常为相对量)映射到 MuJoCo 世界坐标 imu 位置。"""
	return imu_pos_ref + (base_pos - base_pos_ref)


def root_world_from_imu_pose(imu_world_pos: np.ndarray, imu_world_rot: np.ndarray, imu_local_offset: np.ndarray) -> np.ndarray:
	"""根据 imu 世界位姿与 imu 在 root 坐标下偏置，反算 root 世界位置。"""
	return imu_world_pos - imu_world_rot @ imu_local_offset


def parse_trajectory_json(json_path: Path) -> tuple[list[FrameData], str]:
	with open(json_path, "r", encoding="utf-8") as f:
		obj = json.load(f)
	meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}
	coord_frame = str(obj.get("coord_frame", meta.get("coord_frame", "body"))).lower()
	if coord_frame not in ("body", "world"):
		coord_frame = "body"
	frames_raw = obj.get("frames", [])
	frames: list[FrameData] = []
	for i, fr in enumerate(frames_raw):
		base_pos_raw = fr.get("body_pos", [0.0, 0.0, 0.0])
		if base_pos_raw is None:
			base_pos_raw = [0.0, 0.0, 0.0]
		base_pos = np.asarray(base_pos_raw, dtype=float).reshape(3)
		base_rot_raw = fr.get("body_rot", np.eye(3))
		if base_rot_raw is None:
			base_rot_raw = np.eye(3)
		base_rot = np.asarray(base_rot_raw, dtype=float).reshape(3, 3)
		det_pos = fr.get("detection_pos", None)
		gt_pos = fr.get("gt_pos", None)
		# 兼容在线轨迹字段（kf_pos/kf_vel）与离线输出字段（kf_main_pos/kf_main_vel）
		kf_pos = fr.get("kf_pos", fr.get("kf_main_pos", None))
		kf_vel = fr.get("kf_vel", fr.get("kf_main_vel", None))
		kf_state_raw = fr.get("kf_state", None)
		if kf_state_raw is None:
			# 离线输出没有 kf_state，用 main_did_upgrade 推断
			if bool(fr.get("main_did_upgrade", False)):
				kf_state_raw = "update"
			elif kf_pos is not None or kf_vel is not None:
				kf_state_raw = "predict"
			else:
				kf_state_raw = ""

		# detection_pos 保持“原坐标系”存储：
		# - coord_frame=world: 直接存 world
		# - coord_frame=body: 存 body
		# 这样拟合时可以明确先统一到 world 再做拟合。
		det_body = None
		if det_pos is not None:
			det_arr = np.asarray(det_pos, dtype=float).reshape(3)
			det_body = det_arr

		gt_body = None
		if gt_pos is not None:
			gt_arr = np.asarray(gt_pos, dtype=float).reshape(3)
			gt_body = gt_arr

		kf_pos_body = None
		if kf_pos is not None:
			kp_arr = np.asarray(kf_pos, dtype=float).reshape(3)
			if coord_frame == "world":
				kf_pos_body = base_rot.T @ (kp_arr - base_pos)
			else:
				kf_pos_body = kp_arr

		kf_vel_body = None
		if kf_vel is not None:
			kv_arr = np.asarray(kf_vel, dtype=float).reshape(3)
			if coord_frame == "world":
				kf_vel_body = base_rot.T @ kv_arr
			else:
				kf_vel_body = kv_arr

		frames.append(
			FrameData(
				frame_index=int(fr.get("frame_index", i)),
				timestamp=float(fr.get("timestamp", 0.0)),
				base_pos=base_pos,
				base_rot=base_rot,
				detection_pos=det_body,
				gt_pos=gt_body,
				kf_pos_body=kf_pos_body,
				kf_vel_body=kf_vel_body,
				kf_state=str(kf_state_raw).lower(),
				has_detection=bool(fr.get("has_detection", det_pos is not None)),
			)
		)
	frames.sort(key=lambda x: x.frame_index)
	return frames, coord_frame


def _gt_world_from_frame(
	fr: FrameData,
	coord_frame: str,
	base_pos_ref: np.ndarray,
	imu_pos_ref: np.ndarray,
) -> np.ndarray | None:
	if fr.gt_pos is None:
		return None
	if coord_frame == "world":
		gt_body = fr.base_rot.T @ (fr.gt_pos.copy() - fr.base_pos)
	else:
		gt_body = fr.gt_pos.copy()
	imu_world_pos = map_base_pos_to_imu_world(fr.base_pos, base_pos_ref, imu_pos_ref)
	return body_to_world(imu_world_pos, fr.base_rot, gt_body)


def _fit_gt_kinematic_model(
	frames: list[FrameData],
	coord_frame: str,
	base_pos_ref: np.ndarray,
	imu_pos_ref: np.ndarray,
	anchor_update_idx: int,
) -> dict | None:
	"""用 GT 点拟合 x/y 直线、z 抛物线，并在 anchor_update_idx 时刻取拟合初值。"""
	ts = []
	gt_world = []

	if anchor_update_idx < 0 or anchor_update_idx >= len(frames):
		return None
	t0 = float(frames[anchor_update_idx].timestamp)

	anchor_gt_world = None

	for i, fr in enumerate(frames):
		if i > anchor_update_idx or float(fr.timestamp) > t0 + 1e-9:
			continue

		p_world = _gt_world_from_frame(
			fr,
			coord_frame=coord_frame,
			base_pos_ref=base_pos_ref,
			imu_pos_ref=imu_pos_ref,
		)
		if p_world is None:
			continue

		if i == anchor_update_idx:
			anchor_gt_world = p_world.copy()
		ts.append(float(fr.timestamp))
		gt_world.append(p_world)

	if len(ts) < 3:
		return None

	t = np.asarray(ts, dtype=float)
	p = np.asarray(gt_world, dtype=float)
	tau = t - t0
	if np.ptp(tau) <= 1e-12:
		return None

	coef_x = np.polyfit(tau, p[:, 0], 1)
	coef_y = np.polyfit(tau, p[:, 1], 1)
	coef_z = np.polyfit(tau, p[:, 2], 2)

	if anchor_gt_world is not None:
		p0 = anchor_gt_world.astype(float)
	else:
		p0 = np.array([
			float(np.polyval(coef_x, 0.0)),
			float(np.polyval(coef_y, 0.0)),
			float(np.polyval(coef_z, 0.0)),
		], dtype=float)
	v0 = np.array([
		float(coef_x[0]),
		float(coef_y[0]),
		float(coef_z[1]),
	], dtype=float)

	return {
		"coef_x": coef_x,
		"coef_y": coef_y,
		"coef_z": coef_z,
		"last_update_idx": int(anchor_update_idx),
		"t0": t0,
		"p0": p0,
		"v0": v0,
		"fit_count": int(len(t)),
	}


def _fit_detection_kinematic_model(
	frames: list[FrameData],
	coord_frame: str,
	base_pos_ref: np.ndarray,
	imu_pos_ref: np.ndarray,
	anchor_update_idx: int,
) -> dict | None:
	"""用 detection 点拟合 x/y 直线、z 抛物线，并在 anchor_update_idx 时刻取拟合初值。"""
	ts = []
	det_world = []

	if anchor_update_idx < 0 or anchor_update_idx >= len(frames):
		return None
	t0 = float(frames[anchor_update_idx].timestamp)

	anchor_det_world = None

	for i, fr in enumerate(frames):
		# 仅使用回放起点（最后一次update）及其之前的检测点进行拟合
		if i > anchor_update_idx or float(fr.timestamp) > t0 + 1e-9:
			continue
		if not fr.has_detection or fr.detection_pos is None:
			continue

		# 先统一到 MuJoCo 世界坐标，再拟合：
		# - 原始 world 先转 body（去掉记录坐标原点影响）
		# - 再用 imu 映射后的世界位姿转到 MuJoCo world
		if coord_frame == "world":
			det_body = fr.base_rot.T @ (fr.detection_pos.copy() - fr.base_pos)
		else:
			det_body = fr.detection_pos.copy()
		imu_world_pos = map_base_pos_to_imu_world(fr.base_pos, base_pos_ref, imu_pos_ref)
		p_world = body_to_world(imu_world_pos, fr.base_rot, det_body)

		if i == anchor_update_idx:
			anchor_det_world = p_world.copy()
		ts.append(float(fr.timestamp))
		det_world.append(p_world)

	if len(ts) < 3:
		return None

	t = np.asarray(ts, dtype=float)
	p = np.asarray(det_world, dtype=float)
	# 关键：使用相对锚点时间，避免绝对时间戳（~1e9）导致 polyfit 数值病态
	tau = t - t0
	if np.ptp(tau) <= 1e-12:
		return None

	coef_x = np.polyfit(tau, p[:, 0], 1)
	coef_y = np.polyfit(tau, p[:, 1], 1)
	coef_z = np.polyfit(tau, p[:, 2], 2)

	# 起始位置优先锚点真实检测，避免 OLS 截距导致“起点对不上”
	if anchor_det_world is not None:
		p0 = anchor_det_world.astype(float)
	else:
		p0 = np.array([
			float(np.polyval(coef_x, 0.0)),
			float(np.polyval(coef_y, 0.0)),
			float(np.polyval(coef_z, 0.0)),
		], dtype=float)
	v0 = np.array([
		float(coef_x[0]),
		float(coef_y[0]),
		float(coef_z[1]),
	], dtype=float)

	return {
		"coef_x": coef_x,
		"coef_y": coef_y,
		"coef_z": coef_z,
		"last_update_idx": int(anchor_update_idx),
		"t0": t0,
		"p0": p0,
		"v0": v0,
		"fit_count": int(len(t)),
	}


def _predict_kinematic_from_fit(fit_info: dict | None, t: float, g_abs: float) -> tuple[np.ndarray, np.ndarray]:
	if fit_info is None:
		nan3 = np.full(3, np.nan, dtype=float)
		return nan3.copy(), nan3.copy()
	t0 = float(fit_info["t0"])
	if t < t0 - 1e-9:
		nan3 = np.full(3, np.nan, dtype=float)
		return nan3.copy(), nan3.copy()
	dt = float(t - t0)
	p0 = np.asarray(fit_info["p0"], dtype=float)
	v0 = np.asarray(fit_info["v0"], dtype=float)
	p = np.array([
		p0[0] + v0[0] * dt,
		p0[1] + v0[1] * dt,
		p0[2] + v0[2] * dt - 0.5 * g_abs * dt * dt,
	], dtype=float)
	v = np.array([
		v0[0],
		v0[1],
		v0[2] - g_abs * dt,
	], dtype=float)
	return p, v


def _detection_world_from_frame(
	fr: FrameData,
	coord_frame: str,
	base_pos_ref: np.ndarray,
	imu_pos_ref: np.ndarray,
) -> np.ndarray | None:
	if fr.detection_pos is None:
		return None
	if coord_frame == "world":
		det_body = fr.base_rot.T @ (fr.detection_pos.copy() - fr.base_pos)
	else:
		det_body = fr.detection_pos.copy()
	imu_world_pos = map_base_pos_to_imu_world(fr.base_pos, base_pos_ref, imu_pos_ref)
	return body_to_world(imu_world_pos, fr.base_rot, det_body)


def find_init_update_index(frames: list[FrameData], first_predict_idx: int) -> int | None:
	# 严格模式：优先在“首个predict之前”寻找下降段最后一次update
	descending_updates = [
		i
		for i in range(first_predict_idx)
		if frames[i].kf_state == "update"
		and frames[i].kf_vel_body is not None
		and frames[i].kf_vel_body[2] < 0.0
	]
	if descending_updates:
		return descending_updates[-1]

	# 兜底：有些轨迹会先经历一段predict，再在后半段重新检测到并进入下降update。
	# 这种情况下回退到“全轨迹最后一个下降update”，避免整条轨迹被跳过。
	global_descending_updates = [
		i
		for i, fr in enumerate(frames)
		if fr.kf_state == "update"
		and fr.kf_vel_body is not None
		and fr.kf_vel_body[2] < 0.0
	]
	if global_descending_updates:
		return global_descending_updates[-1]

	return None


def set_freejoint_pose_vel(
	data: mujoco.MjData,
	model: mujoco.MjModel,
	joint_id: int,
	pos: np.ndarray,
	quat_wxyz: np.ndarray,
	linvel: np.ndarray | None = None,
	angvel: np.ndarray | None = None,
) -> None:
	qadr = int(model.jnt_qposadr[joint_id])
	dadr = int(model.jnt_dofadr[joint_id])
	data.qpos[qadr : qadr + 3] = pos
	data.qpos[qadr + 3 : qadr + 7] = quat_wxyz
	lv = np.zeros(3, dtype=float) if linvel is None else np.asarray(linvel, dtype=float).reshape(3)
	av = np.zeros(3, dtype=float) if angvel is None else np.asarray(angvel, dtype=float).reshape(3)
	data.qvel[dadr : dadr + 3] = lv
	data.qvel[dadr + 3 : dadr + 6] = av


def set_hinge_joint_value(data: mujoco.MjData, model: mujoco.MjModel, joint_id: int, value: float) -> None:
	qadr = int(model.jnt_qposadr[joint_id])
	dadr = int(model.jnt_dofadr[joint_id])
	data.qpos[qadr] = float(value)
	data.qvel[dadr] = 0.0


def apply_fixed_arm_pose(data: mujoco.MjData, model: mujoco.MjModel, fixed_joint_targets: dict[int, float]) -> None:
	for jid, val in fixed_joint_targets.items():
		set_hinge_joint_value(data, model, jid, val)


def park_unused_balls(data: mujoco.MjData, model: mujoco.MjModel, joint_ids: list[int]) -> None:
	"""把未使用的小球停到远处，避免影响仿真与观感。"""
	for jid in joint_ids:
		set_freejoint_pose_vel(
			data,
			model,
			jid,
			pos=np.array([50.0, 50.0, -10.0], dtype=float),
			quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
			linvel=np.zeros(3, dtype=float),
			angvel=np.zeros(3, dtype=float),
		)


def step_exact_dt(
	model: mujoco.MjModel,
	data: mujoco.MjData,
	dt: float,
	base_timestep: float,
	fixed_joint_targets: dict[int, float] | None = None,
) -> None:
	if dt <= 0.0:
		return
	n = int(np.floor(dt / base_timestep))
	rem = float(dt - n * base_timestep)
	for _ in range(max(0, n)):
		if fixed_joint_targets:
			apply_fixed_arm_pose(data, model, fixed_joint_targets)
		mujoco.mj_step(model, data)
	if rem > 1e-10:
		old = float(model.opt.timestep)
		model.opt.timestep = rem
		if fixed_joint_targets:
			apply_fixed_arm_pose(data, model, fixed_joint_targets)
		mujoco.mj_step(model, data)
		model.opt.timestep = old
	if fixed_joint_targets:
		apply_fixed_arm_pose(data, model, fixed_joint_targets)


def build_one_trajectory(
	json_path: Path,
	model_template: mujoco.MjModel,
	root_joint_id: int,
	ball_joint_id: int,
	extra_ball_joint_ids: list[int],
	imu_pos_ref: np.ndarray,
	imu_local_offset: np.ndarray,
	fixed_joint_targets: dict[int, float],
	init_source: str,
) -> TrajectoryData | None:
	frames, coord_frame = parse_trajectory_json(json_path)
	if len(frames) < 2:
		return None

	predict_indices = [
		i
		for i, fr in enumerate(frames)
		if fr.kf_state == "predict" and fr.kf_pos_body is not None and fr.kf_vel_body is not None
	]
	if not predict_indices:
		return None

	first_predict = predict_indices[0]
	init_idx = find_init_update_index(frames, first_predict)
	if init_idx is None:
		print(f"[skip] {json_path.name}: 未找到下降段最后一次 update(vz<0)，严格模式下跳过")
		return None
	if init_idx >= first_predict:
		print(f"[warn] {json_path.name}: 首个predict前无下降update，回退到全轨迹最后一个下降update(idx={init_idx})")
	init_fr = frames[init_idx]
	base_pos_ref = frames[0].base_pos.copy()
	if init_fr.kf_pos_body is None or init_fr.kf_vel_body is None:
		return None

	fit_info = _fit_detection_kinematic_model(
		frames=frames,
		coord_frame=coord_frame,
		base_pos_ref=base_pos_ref,
		imu_pos_ref=imu_pos_ref,
		anchor_update_idx=init_idx,
	)
	gt_fit_info = _fit_gt_kinematic_model(
		frames=frames,
		coord_frame=coord_frame,
		base_pos_ref=base_pos_ref,
		imu_pos_ref=imu_pos_ref,
		anchor_update_idx=init_idx,
	)
	g_abs = float(max(0.0, -model_template.opt.gravity[2]))

	# 只回放初始化之后的predict帧，避免时间倒序导致的dt<=0无效步进
	predict_indices = [
		i
		for i in predict_indices
		if i >= init_idx and frames[i].timestamp >= init_fr.timestamp
	]

	model = model_template
	data = mujoco.MjData(model)
	mujoco.mj_resetData(model, data)
	park_unused_balls(data, model, extra_ball_joint_ids)

	base_q = rotmat_to_quat_wxyz(init_fr.base_rot)
	init_imu_world_pos = map_base_pos_to_imu_world(init_fr.base_pos, base_pos_ref, imu_pos_ref)
	init_root_pos = root_world_from_imu_pose(init_imu_world_pos, init_fr.base_rot, imu_local_offset)
	set_freejoint_pose_vel(data, model, root_joint_id, init_root_pos, base_q)
	apply_fixed_arm_pose(data, model, fixed_joint_targets)
	park_unused_balls(data, model, extra_ball_joint_ids)

	init_ball_world_pos = body_to_world(init_imu_world_pos, init_fr.base_rot, init_fr.kf_pos_body)
	init_ball_world_vel = vel_body_to_world(init_fr.base_rot, init_fr.kf_vel_body)
	if init_source == "detection":
		det_world = _detection_world_from_frame(
			init_fr,
			coord_frame=coord_frame,
			base_pos_ref=base_pos_ref,
			imu_pos_ref=imu_pos_ref,
		)
		fit_p0_ok = fit_info is not None and np.all(np.isfinite(fit_info.get("p0", np.full(3, np.nan, dtype=float))))
		fit_v0_ok = fit_info is not None and np.all(np.isfinite(fit_info.get("v0", np.full(3, np.nan, dtype=float))))

		if fit_p0_ok:
			init_ball_world_pos = np.asarray(fit_info["p0"], dtype=float).reshape(3)
		elif det_world is not None:
			init_ball_world_pos = det_world
			print(f"[warn] {json_path.name}: init-source=detection 但无可用拟合位置，回退使用detection位置初始化")
		else:
			print(f"[warn] {json_path.name}: init-source=detection 但锚点无detection，回退使用KF初始化")

		if fit_v0_ok:
			init_ball_world_vel = np.asarray(fit_info["v0"], dtype=float).reshape(3)
		else:
			print(f"[warn] {json_path.name}: init-source=detection 但无可用拟合速度，回退使用KF速度初始化")
	set_freejoint_pose_vel(
		data,
		model,
		ball_joint_id,
		init_ball_world_pos,
		np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
		linvel=init_ball_world_vel,
		angvel=np.zeros(3, dtype=float),
	)
	apply_fixed_arm_pose(data, model, fixed_joint_targets)
	park_unused_balls(data, model, extra_ball_joint_ids)
	mujoco.mj_forward(model, data)

	base_dt = float(model.opt.timestep)
	sim_time = float(init_fr.timestamp)
	predict_samples: list[PredictSample] = []

	ball_qadr = int(model.jnt_qposadr[ball_joint_id])
	ball_dadr = int(model.jnt_dofadr[ball_joint_id])

	# 严格第0帧：下降段最后一次 update（用于初始化 set）
	init_sim_ball_world_pos = data.qpos[ball_qadr : ball_qadr + 3].copy()
	init_sim_ball_world_vel = data.qvel[ball_dadr : ball_dadr + 3].copy()
	init_kf_world_pos = body_to_world(init_imu_world_pos, init_fr.base_rot, init_fr.kf_pos_body)
	init_kf_world_vel = vel_body_to_world(init_fr.base_rot, init_fr.kf_vel_body)
	init_fit_world_pos, init_fit_world_vel = _predict_kinematic_from_fit(fit_info, float(init_fr.timestamp), g_abs)
	init_gt_fit_world_pos, init_gt_fit_world_vel = _predict_kinematic_from_fit(gt_fit_info, float(init_fr.timestamp), g_abs)
	predict_samples.append(
		PredictSample(
			frame_index=init_fr.frame_index,
			timestamp=init_fr.timestamp,
			imu_world_pos=init_imu_world_pos.copy(),
			base_rot=init_fr.base_rot.copy(),
			kf_world_pos=init_kf_world_pos,
			kf_world_vel=init_kf_world_vel,
			fit_world_pos=init_fit_world_pos,
			fit_world_vel=init_fit_world_vel,
			gt_fit_world_pos=init_gt_fit_world_pos,
			gt_fit_world_vel=init_gt_fit_world_vel,
			sim_ball_world_pos=init_sim_ball_world_pos,
			sim_ball_world_vel=init_sim_ball_world_vel,
			pos_error_xyz=(init_kf_world_pos - init_sim_ball_world_pos).copy(),
			vel_error_xyz=(init_kf_world_vel - init_sim_ball_world_vel).copy(),
			fit_pos_error_xyz=(init_fit_world_pos - init_sim_ball_world_pos).copy(),
			fit_vel_error_xyz=(init_fit_world_vel - init_sim_ball_world_vel).copy(),
		)
	)

	for idx in predict_indices:
		fr = frames[idx]
		q = rotmat_to_quat_wxyz(fr.base_rot)
		imu_world_pos = map_base_pos_to_imu_world(fr.base_pos, base_pos_ref, imu_pos_ref)
		root_world_pos = root_world_from_imu_pose(imu_world_pos, fr.base_rot, imu_local_offset)
		# 每一个仿真阶段都按该帧 imu 位姿设置机器人
		set_freejoint_pose_vel(data, model, root_joint_id, root_world_pos, q)
		apply_fixed_arm_pose(data, model, fixed_joint_targets)
		park_unused_balls(data, model, extra_ball_joint_ids)
		mujoco.mj_forward(model, data)

		dt = float(fr.timestamp - sim_time)
		step_exact_dt(model, data, dt=dt, base_timestep=base_dt, fixed_joint_targets=fixed_joint_targets)
		sim_time = float(fr.timestamp)

		# 在输出时再次对齐该帧 robot base
		set_freejoint_pose_vel(data, model, root_joint_id, root_world_pos, q)
		apply_fixed_arm_pose(data, model, fixed_joint_targets)
		park_unused_balls(data, model, extra_ball_joint_ids)
		mujoco.mj_forward(model, data)

		sim_ball_world_pos = data.qpos[ball_qadr : ball_qadr + 3].copy()
		sim_ball_world_vel = data.qvel[ball_dadr : ball_dadr + 3].copy()
		kf_world_pos = body_to_world(imu_world_pos, fr.base_rot, fr.kf_pos_body)
		kf_world_vel = vel_body_to_world(fr.base_rot, fr.kf_vel_body)
		fit_world_pos, fit_world_vel = _predict_kinematic_from_fit(fit_info, float(fr.timestamp), g_abs)
		gt_fit_world_pos, gt_fit_world_vel = _predict_kinematic_from_fit(gt_fit_info, float(fr.timestamp), g_abs)

		pos_error_xyz = (kf_world_pos - sim_ball_world_pos).copy()
		vel_error_xyz = (kf_world_vel - sim_ball_world_vel).copy()

		predict_samples.append(
			PredictSample(
				frame_index=fr.frame_index,
				timestamp=fr.timestamp,
				imu_world_pos=imu_world_pos.copy(),
				base_rot=fr.base_rot.copy(),
				kf_world_pos=kf_world_pos,
				kf_world_vel=kf_world_vel,
				fit_world_pos=fit_world_pos,
				fit_world_vel=fit_world_vel,
				gt_fit_world_pos=gt_fit_world_pos,
				gt_fit_world_vel=gt_fit_world_vel,
				sim_ball_world_pos=sim_ball_world_pos,
				sim_ball_world_vel=sim_ball_world_vel,
				pos_error_xyz=pos_error_xyz,
				vel_error_xyz=vel_error_xyz,
				fit_pos_error_xyz=(fit_world_pos - sim_ball_world_pos).copy(),
				fit_vel_error_xyz=(fit_world_vel - sim_ball_world_vel).copy(),
			)
		)

	return TrajectoryData(
		name=json_path.name,
		frames=frames,
		base_pos_ref=base_pos_ref,
		init_update_index=init_idx,
		predict_samples=predict_samples,
		fit_info=fit_info,
	)


def add_marker_sphere(viewer: mujoco.viewer.Handle, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
	scene = viewer.user_scn
	if scene.ngeom >= scene.maxgeom:
		return
	g = scene.geoms[scene.ngeom]
	mujoco.mjv_initGeom(
		g,
		type=mujoco.mjtGeom.mjGEOM_SPHERE,
		size=np.array([radius, 0.0, 0.0], dtype=np.float64),
		pos=np.asarray(pos, dtype=np.float64),
		mat=np.eye(3, dtype=np.float64).reshape(-1),
		rgba=np.asarray(rgba, dtype=np.float32),
	)
	scene.ngeom += 1


def load_all_trajectories(
	trajectory_dir: Path,
	model: mujoco.MjModel,
	root_joint_id: int,
	ball_joint_id: int,
	extra_ball_joint_ids: list[int],
	imu_pos_ref: np.ndarray,
	imu_local_offset: np.ndarray,
	fixed_joint_targets: dict[int, float],
	init_source: str,
) -> list[TrajectoryData]:
	json_files = sorted([p for p in trajectory_dir.glob("trajectory_*.json") if not p.name.endswith(".bak")])
	out: list[TrajectoryData] = []
	for p in json_files:
		td = build_one_trajectory(
			p,
			model,
			root_joint_id,
			ball_joint_id,
			imu_pos_ref=imu_pos_ref,
			extra_ball_joint_ids=extra_ball_joint_ids,
			imu_local_offset=imu_local_offset,
			fixed_joint_targets=fixed_joint_targets,
			init_source=init_source,
		)
		if td is not None and td.predict_samples:
			out.append(td)
	return out


def run(model_path: Path, trajectory_dir: Path, init_source: str = "kf") -> None:
	model = mujoco.MjModel.from_xml_path(str(model_path))
	data = mujoco.MjData(model)
	mujoco.mj_resetData(model, data)

	root_joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root"))
	ball_joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_0_joint"))
	if root_joint_id < 0:
		raise RuntimeError("未找到 root freejoint，请检查 XML 是否包含 <freejoint name=\"root\" />")
	if ball_joint_id < 0:
		raise RuntimeError("未找到 ball_0_joint，请检查 XML 中 ball_0 配置")

	extra_ball_joint_ids: list[int] = []
	for jname in ("ball_1_joint", "ball_2_joint"):
		jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname))
		if jid >= 0:
			extra_ball_joint_ids.append(jid)

	root_qadr = int(model.jnt_qposadr[root_joint_id])
	root_pos_ref = data.qpos[root_qadr : root_qadr + 3].copy()
	root_quat_ref = data.qpos[root_qadr + 3 : root_qadr + 7].copy()
	root_rot_ref = quat_wxyz_to_rotmat(root_quat_ref)

	imu_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu"))
	if imu_site_id < 0:
		raise RuntimeError("未找到 site 'imu'，请检查 XML")
	mujoco.mj_forward(model, data)
	imu_pos_ref = data.site_xpos[imu_site_id].copy()
	imu_local_offset = root_rot_ref.T @ (imu_pos_ref - root_pos_ref)

	fixed_joint_name_to_value = {
		"left_shoulder_roll": 0.35,
		"right_shoulder_roll": -0.35,
		"left_elbow": 1.51,
		"right_elbow": 1.51,
	}
	fixed_joint_targets: dict[int, float] = {}
	for jname, jval in fixed_joint_name_to_value.items():
		jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname))
		if jid < 0:
			raise RuntimeError(f"未找到关节: {jname}")
		fixed_joint_targets[jid] = float(jval)

	trajectories = load_all_trajectories(
		trajectory_dir,
		model,
		root_joint_id,
		ball_joint_id,
		extra_ball_joint_ids=extra_ball_joint_ids,
		imu_pos_ref=imu_pos_ref,
		imu_local_offset=imu_local_offset,
		fixed_joint_targets=fixed_joint_targets,
		init_source=init_source,
	)
	if not trajectories:
		raise RuntimeError(f"未找到可用轨迹：{trajectory_dir}")

	print(f"已加载模型: {model_path}")
	print(f"轨迹目录: {trajectory_dir}")
	print(f"可用轨迹数: {len(trajectories)}")
	print(f"初始化赋值来源: {init_source}")
	print("按键: ←/→=前后帧, ↑/↓=前后轨迹, Space=暂停/播放（默认暂停）")

	error_source = "kinematic" if init_source == "detection" else "kf"
	error_title = "Kinematic - Sim" if error_source == "kinematic" else "KF - Sim"
	kf_gt_title = "KF - GT-Kinematic"
	kine_gt_title = "Kinematic - GT-Kinematic"
	per_traj_stats = [_collect_error_stats_for_trajectory(t, error_source=error_source) for t in trajectories]
	all_stats = _collect_error_stats_all(trajectories, error_source=error_source)
	all_stats_kf_gt = _collect_error_stats_all(trajectories, error_source="kf_gt")
	all_stats_kine_gt = _collect_error_stats_all(trajectories, error_source="kinematic_gt")
	_print_error_stats(f"所有轨迹汇总误差统计 (error = {kf_gt_title})", all_stats_kf_gt)
	_print_error_stats(f"所有轨迹汇总误差统计 (error = {kine_gt_title})", all_stats_kine_gt)
	_print_error_stats(f"所有轨迹汇总误差统计 (error = {error_title})", all_stats)
	plot_output_dir = trajectory_dir / "mujoco_error_plots"

	state = {
		"traj_i": 0,
		"sample_i": 0,
		"playing": False,
		"pending": [],
		"last_wall": time.monotonic(),
		"request_exit": False,
		"traj_end_printed": set(),
	}

	def maybe_print_traj_end_stats(traj_i: int) -> None:
		if traj_i < 0 or traj_i >= len(trajectories):
			return
		if traj_i in state["traj_end_printed"]:
			return
		traj = trajectories[traj_i]
		if not traj.predict_samples:
			return
		if state["sample_i"] != len(traj.predict_samples) - 1:
			return
		_print_error_stats(
			f"轨迹结束统计 [{traj_i+1}/{len(trajectories)}] {traj.name} (error = {error_title})",
			per_traj_stats[traj_i],
		)
		if traj.fit_info is not None:
			v0 = np.asarray(traj.fit_info["v0"], dtype=float)
			t0 = float(traj.fit_info["t0"])
			print(
				"[fit] detection拟合: "
				f"samples={traj.fit_info.get('fit_count', 0)}, "
				f"last_update_idx={traj.fit_info.get('last_update_idx', -1)}, t0={t0:.6f}, "
				f"v0_fit=({v0[0]:+.4f}, {v0[1]:+.4f}, {v0[2]:+.4f}) m/s"
			)
		plot_path = _save_traj_error_plot(traj, plot_output_dir, show_plot=True, error_source=error_source)
		if plot_path is not None:
			print(f"[plot] 已展示并保存轨迹误差时序图: {plot_path}")
		state_plot_path = _save_traj_state_plot(traj, plot_output_dir, show_plot=True)
		if state_plot_path is not None:
			print(f"[plot] 已展示并保存轨迹状态时序图(KF vs Sim): {state_plot_path}")
		state["traj_end_printed"].add(traj_i)

	def apply_sample_to_scene(traj: TrajectoryData, smp_i: int) -> None:
		smp = traj.predict_samples[smp_i]

		root_q = rotmat_to_quat_wxyz(smp.base_rot)
		root_world_pos = root_world_from_imu_pose(smp.imu_world_pos, smp.base_rot, imu_local_offset)
		set_freejoint_pose_vel(data, model, root_joint_id, root_world_pos, root_q)
		set_freejoint_pose_vel(
			data,
			model,
			ball_joint_id,
			smp.sim_ball_world_pos,
			np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
			linvel=smp.sim_ball_world_vel,
			angvel=np.zeros(3, dtype=float),
		)
		apply_fixed_arm_pose(data, model, fixed_joint_targets)
		park_unused_balls(data, model, extra_ball_joint_ids)
		data.time = float(smp.timestamp)
		mujoco.mj_forward(model, data)

		print(
			f"[traj {state['traj_i']+1}/{len(trajectories)}: {traj.name}] "
			f"frame={smp.frame_index} ts={smp.timestamp:.6f}"
		)
		print(
			"  world KF    pos=("
			f"{smp.kf_world_pos[0]:+.4f}, {smp.kf_world_pos[1]:+.4f}, {smp.kf_world_pos[2]:+.4f}) m, "
			"vel=("
			f"{smp.kf_world_vel[0]:+.4f}, {smp.kf_world_vel[1]:+.4f}, {smp.kf_world_vel[2]:+.4f}) m/s"
		)
		print(
			"  world Sim   pos=("
			f"{smp.sim_ball_world_pos[0]:+.4f}, {smp.sim_ball_world_pos[1]:+.4f}, {smp.sim_ball_world_pos[2]:+.4f}) m, "
			"vel=("
			f"{smp.sim_ball_world_vel[0]:+.4f}, {smp.sim_ball_world_vel[1]:+.4f}, {smp.sim_ball_world_vel[2]:+.4f}) m/s"
		)
		if np.all(np.isfinite(smp.fit_world_pos)) and np.all(np.isfinite(smp.fit_world_vel)):
			print(
				"  world Kin   pos=("
				f"{smp.fit_world_pos[0]:+.4f}, {smp.fit_world_pos[1]:+.4f}, {smp.fit_world_pos[2]:+.4f}) m, "
				"vel=("
				f"{smp.fit_world_vel[0]:+.4f}, {smp.fit_world_vel[1]:+.4f}, {smp.fit_world_vel[2]:+.4f}) m/s"
			)
		print(
			"  error (KF - Sim): "
			"pos_err_xyz=("
			f"{smp.pos_error_xyz[0]:+.4f}, {smp.pos_error_xyz[1]:+.4f}, {smp.pos_error_xyz[2]:+.4f}) m, "
			"vel_err_xyz=("
			f"{smp.vel_error_xyz[0]:+.4f}, {smp.vel_error_xyz[1]:+.4f}, {smp.vel_error_xyz[2]:+.4f}) m/s"
		)
		if np.all(np.isfinite(smp.fit_pos_error_xyz)) and np.all(np.isfinite(smp.fit_vel_error_xyz)):
			print(
				"  error (Kin - Sim): "
				"pos_err_xyz=("
				f"{smp.fit_pos_error_xyz[0]:+.4f}, {smp.fit_pos_error_xyz[1]:+.4f}, {smp.fit_pos_error_xyz[2]:+.4f}) m, "
				"vel_err_xyz=("
				f"{smp.fit_vel_error_xyz[0]:+.4f}, {smp.fit_vel_error_xyz[1]:+.4f}, {smp.fit_vel_error_xyz[2]:+.4f}) m/s"
			)

	def key_callback(keycode: int) -> None:
		state["pending"].append(keycode)

	def is_left_key(k: int) -> bool:
		return k == 263  # GLFW_KEY_LEFT

	def is_right_key(k: int) -> bool:
		return k == 262  # GLFW_KEY_RIGHT

	def is_up_key(k: int) -> bool:
		return k == 265  # GLFW_KEY_UP

	def is_down_key(k: int) -> bool:
		return k == 264  # GLFW_KEY_DOWN

	def draw_current_markers() -> None:
		viewer.user_scn.ngeom = 0
		smp = trajectories[state["traj_i"]].predict_samples[state["sample_i"]]
		add_marker_sphere(viewer, smp.kf_world_pos, radius=0.025, rgba=np.array([0.2, 0.5, 1.0, 0.95]))
		add_marker_sphere(viewer, smp.sim_ball_world_pos, radius=0.02, rgba=np.array([0.1, 0.9, 0.2, 0.8]))
		if np.all(np.isfinite(smp.fit_world_pos)):
			add_marker_sphere(viewer, smp.fit_world_pos, radius=0.018, rgba=np.array([1.0, 0.6, 0.1, 0.9]))

	with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
		apply_sample_to_scene(trajectories[state["traj_i"]], state["sample_i"])
		draw_current_markers()
		viewer.sync()

		while viewer.is_running():
			if state["request_exit"]:
				print("已完成所有轨迹可视化，收到下一步指令，退出。")
				break

			while state["pending"]:
				key = state["pending"].pop(0)
				if is_left_key(key):
					state["sample_i"] = max(0, state["sample_i"] - 1)
					state["playing"] = False
					state["last_wall"] = time.monotonic()
					apply_sample_to_scene(trajectories[state["traj_i"]], state["sample_i"])
				elif is_right_key(key):
					traj = trajectories[state["traj_i"]]
					if state["sample_i"] >= len(traj.predict_samples) - 1:
						maybe_print_traj_end_stats(state["traj_i"])
						# 当前轨迹末帧再按“下一帧”：切下一轨迹；若已是最后一条则退出
						if state["traj_i"] >= len(trajectories) - 1:
							state["request_exit"] = True
							break
						state["traj_i"] += 1
						state["sample_i"] = 0
					else:
						state["sample_i"] += 1
					state["playing"] = False
					state["last_wall"] = time.monotonic()
					apply_sample_to_scene(trajectories[state["traj_i"]], state["sample_i"])
				elif is_up_key(key):
					state["traj_i"] = max(0, state["traj_i"] - 1)
					state["sample_i"] = 0
					state["playing"] = False
					state["last_wall"] = time.monotonic()
					apply_sample_to_scene(trajectories[state["traj_i"]], state["sample_i"])
				elif is_down_key(key):
					maybe_print_traj_end_stats(state["traj_i"])
					if state["traj_i"] >= len(trajectories) - 1:
						# 最后一条再按“下一轨迹”直接退出
						state["request_exit"] = True
						break
					state["traj_i"] += 1
					state["sample_i"] = 0
					state["playing"] = False
					state["last_wall"] = time.monotonic()
					apply_sample_to_scene(trajectories[state["traj_i"]], state["sample_i"])
				elif key == ord(" "):
					state["playing"] = not state["playing"]
					state["last_wall"] = time.monotonic()

			traj = trajectories[state["traj_i"]]
			if state["playing"] and state["sample_i"] < len(traj.predict_samples) - 1:
				curr = traj.predict_samples[state["sample_i"]]
				nxt = traj.predict_samples[state["sample_i"] + 1]
				wait_dt = max(0.0, float(nxt.timestamp - curr.timestamp))
				now = time.monotonic()
				if now - state["last_wall"] >= wait_dt:
					state["sample_i"] += 1
					state["last_wall"] = now
					apply_sample_to_scene(traj, state["sample_i"])

			maybe_print_traj_end_stats(state["traj_i"])

			smp = trajectories[state["traj_i"]].predict_samples[state["sample_i"]]
			apply_fixed_arm_pose(data, model, fixed_joint_targets)
			park_unused_balls(data, model, extra_ball_joint_ids)
			mujoco.mj_forward(model, data)
			draw_current_markers()
			viewer.sync()
			time.sleep(0.001)


def main() -> None:
	parser = argparse.ArgumentParser(description="MuJoCo 回放 KF 预测与重力仿真对比")
	parser.add_argument("--model", type=str, default=None, help="MJCF/XML 模型路径")
	parser.add_argument("--trajectory-dir", type=str, default=None, help="轨迹 JSON 目录")
	parser.add_argument(
		"--init-source",
		type=str,
		default="kf",
		choices=["kf", "detection"],
		help="小球初始化赋值来源：kf 或 detection(默认)",
	)
	args = parser.parse_args()

	model_path = resolve_model_path(args.model)
	trajectory_dir = resolve_trajectory_dir(args.trajectory_dir)
	run(model_path=model_path, trajectory_dir=trajectory_dir, init_source=args.init_source)


if __name__ == "__main__":
	main()
