#!/usr/bin/env python3
"""
将 trajectory_data 中每条轨迹的 detection 位置逐条画出来。

默认行为：
- 扫描 trajectory_data/trajectory_tracker*_*.json
- 每条轨迹先绘制 detection 的 X/Y/Z 随时间 t 变化曲线（3张子图）
- 再绘制 detection_pos 的 3D 轨迹（线+点）
- 输出到 trajectory_data/detection_plots/*.png

可选：
- --no-show：不弹窗，仅保存图片
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def _safe_vec3(v):
	if v is None or len(v) < 3:
		return None
	return np.asarray(v[:3], dtype=float)


def _safe_rot3x3(r):
	if r is None:
		return None
	arr = np.asarray(r, dtype=float)
	if arr.size != 9:
		return None
	return arr.reshape(3, 3)


def _to_world_pos(pos, body_pos, body_rot, coord_frame):
	if pos is None:
		return None
	if coord_frame == "world":
		return np.asarray(pos, dtype=float)
	if body_pos is None or body_rot is None:
		return None
	return body_rot @ np.asarray(pos, dtype=float) + body_pos


def _to_world_vel(vel, body_rot, coord_frame):
	if vel is None:
		return None
	if coord_frame == "world":
		return np.asarray(vel, dtype=float)
	if body_rot is None:
		return None
	return body_rot @ np.asarray(vel, dtype=float)


def _to_world_var_diag(var_diag, body_rot, coord_frame):
	if var_diag is None:
		return None
	# 按约定：方差字段直接使用记录值，不做坐标系旋转转换
	return np.asarray(var_diag, dtype=float)


def _load_detection_series(json_path: Path):
	"""读取单条轨迹 detection/gt 点，返回与 KF update 对齐的序列。"""
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	frames = sorted(data.get("frames", []), key=lambda x: x.get("frame_index", 0))
	meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}
	coord_frame = str(data.get("coord_frame", meta.get("coord_frame", "body"))).lower()
	if coord_frame not in ("body", "world"):
		coord_frame = "body"
	dt = float(data.get("dt", meta.get("dt", 1.0 / 60.0)))
	start_ts = data.get("start_timestamp", None)

	points = []
	gt_points = []
	frame_ids = []
	times = []
	kf_update_pos = []
	kf_update_vel = []
	kf_update_pos_var = []
	kf_update_vel_var = []
	normalized_innovation = []
	innovation_mahalanobis2 = []
	contour_areas = []
	online_det_fit_pos = []
	online_det_fit_vel = []
	base_positions = []
	for fr in frames:
		body_pos = _safe_vec3(fr.get("body_pos", None))
		body_rot = _safe_rot3x3(fr.get("body_rot", None))

		det_raw = _safe_vec3(fr.get("detection_pos", None))
		det = _to_world_pos(det_raw, body_pos, body_rot, coord_frame)
		if det is None:
			continue
		points.append(np.asarray(det, dtype=float))

		gt_raw = _safe_vec3(fr.get("gt_pos", None))
		gt = _to_world_pos(gt_raw, body_pos, body_rot, coord_frame)
		if gt is not None:
			gt_points.append(np.asarray(gt, dtype=float))
		else:
			gt_points.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))
		fid = int(fr.get("frame_index", len(frame_ids)))
		frame_ids.append(fid)

		rel_t = fr.get("relative_time", None)
		if rel_t is not None:
			times.append(float(rel_t))
		else:
			ts = fr.get("timestamp", None)
			if ts is not None and start_ts is not None:
				times.append(float(ts) - float(start_ts))
			else:
				times.append(float(fid) * dt)

		if body_pos is not None:
			base_positions.append(np.asarray(body_pos, dtype=float))
		else:
			base_positions.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))

		contour_areas.append(float(fr.get("contour_area")) if fr.get("contour_area", None) is not None else np.nan)

		# 若 JSON 已提供在线拟合结果，优先直接读取
		online_p_raw = _safe_vec3(fr.get("online_detection_fit_pos", None))
		online_v_raw = _safe_vec3(fr.get("online_detection_fit_vel", None))
		online_p = _to_world_pos(online_p_raw, body_pos, body_rot, coord_frame)
		online_v = _to_world_vel(online_v_raw, body_rot, coord_frame)
		if online_p is not None:
			online_det_fit_pos.append(np.asarray(online_p, dtype=float))
		else:
			online_det_fit_pos.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))
		if online_v is not None:
			online_det_fit_vel.append(np.asarray(online_v, dtype=float))
		else:
			online_det_fit_vel.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))

		# detection 对应的 KF update（仅 kf_state=update 的点）
		state = fr.get("kf_state", None)
		kp_raw = _safe_vec3(fr.get("kf_pos", fr.get("kf_main_pos", None)))
		kv_raw = _safe_vec3(fr.get("kf_vel", fr.get("kf_main_vel", None)))
		if state is None:
			# offline_kf 输出无 kf_state，按 main_did_upgrade 推断
			if bool(fr.get("main_did_upgrade", False)):
				state = "update"
			elif kp_raw is not None or kv_raw is not None:
				state = "predict"
			else:
				state = ""
		kp = _to_world_pos(kp_raw, body_pos, body_rot, coord_frame)
		kv = _to_world_vel(kv_raw, body_rot, coord_frame)
		if state == "update" and kp is not None and kv is not None:
			kf_update_pos.append(np.asarray(kp, dtype=float))
			kf_update_vel.append(np.asarray(kv, dtype=float))

			kpv_raw = _safe_vec3(fr.get("kf_pos_var", None))
			kvv_raw = _safe_vec3(fr.get("kf_vel_var", None))
			kpv = _to_world_var_diag(kpv_raw, body_rot, coord_frame)
			kvv = _to_world_var_diag(kvv_raw, body_rot, coord_frame)
			if kpv is not None:
				kf_update_pos_var.append(np.asarray(kpv, dtype=float))
			else:
				kf_update_pos_var.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))
			if kvv is not None:
				kf_update_vel_var.append(np.asarray(kvv, dtype=float))
			else:
				kf_update_vel_var.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))

			nu = fr.get("normalized_innovation", None)
			if nu is not None and len(nu) >= 3:
				normalized_innovation.append(np.asarray(nu[:3], dtype=float))
			else:
				normalized_innovation.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))

			d2 = fr.get("innovation_mahalanobis2", None)
			innovation_mahalanobis2.append(float(d2) if d2 is not None else np.nan)
		else:
			kf_update_pos.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))
			kf_update_vel.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))
			kf_update_pos_var.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))
			kf_update_vel_var.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))
			normalized_innovation.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))
			innovation_mahalanobis2.append(np.nan)

	return (
		points,
		gt_points,
		frame_ids,
		times,
		kf_update_pos,
		kf_update_vel,
		kf_update_pos_var,
		kf_update_vel_var,
		normalized_innovation,
		innovation_mahalanobis2,
		contour_areas,
		online_det_fit_pos,
		online_det_fit_vel,
		base_positions,
	)


def _plot_base_pos_vs_time(
	json_path: Path,
	output_dir: Path,
	times: np.ndarray,
	base_pos: np.ndarray,
	show: bool,
):
	"""单独绘制 base_pos 的 X/Y/Z 随时间曲线（与当前 times 严格对齐）。"""
	if base_pos.size == 0 or not np.isfinite(base_pos).any():
		print(f"[跳过] {json_path.name}: 没有可用的 base_pos")
		return

	fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
	labels = ["base_x", "base_y", "base_z"]
	colors = ["tab:blue", "tab:orange", "tab:green"]

	for i in range(3):
		valid = np.isfinite(times) & np.isfinite(base_pos[:, i])
		if np.any(valid):
			axes[i].plot(times[valid], base_pos[valid, i], color=colors[i], linewidth=1.8, label=labels[i])
			axes[i].scatter(times[valid], base_pos[valid, i], color=colors[i], s=14, alpha=0.9)
		axes[i].set_ylabel(labels[i])
		axes[i].grid(True, alpha=0.3)
		axes[i].legend(loc="best")

	axes[-1].set_xlabel("t (s)")
	fig.suptitle(f"Figure0 | {json_path.stem} | base_pos (timestamp aligned)")
	fig.tight_layout(rect=[0, 0, 1, 0.97])

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{json_path.stem}_figure0_base_pos_vs_t.png"
	fig.savefig(out_path, dpi=160)
	print(f"[已保存] {out_path}")

	if show:
		plt.show()

	plt.close(fig)


def _set_equal_aspect_3d(ax, pts: np.ndarray):
	"""让 3D 坐标轴尽量等比例，避免轨迹形状失真。"""
	mins = pts.min(axis=0)
	maxs = pts.max(axis=0)
	center = 0.5 * (mins + maxs)
	radius = 0.5 * float(np.max(maxs - mins))
	radius = max(radius, 1e-3)

	ax.set_xlim(center[0] - radius, center[0] + radius)
	ax.set_ylim(center[1] - radius, center[1] + radius)
	ax.set_zlim(center[2] - radius, center[2] + radius)


def _poly_formula_str(axis_name: str, coef: np.ndarray):
	if len(coef) == 2:
		k, b = float(coef[0]), float(coef[1])
		return f"{axis_name} = {k:.4f} t + {b:.4f}"
	a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
	return f"{axis_name} = {a:.4f} t² + {b:.4f} t + {c:.4f}"


def _fit_poly_ols(times: np.ndarray, values: np.ndarray, degree: int):
	return np.polyfit(times, values, degree)


def _fit_poly_huber(times: np.ndarray, values: np.ndarray, degree: int):
	coef0 = np.polyfit(times, values, degree)
	res0 = values - np.polyval(coef0, times)
	mad = np.median(np.abs(res0 - np.median(res0)))
	f_scale = max(1e-3, 1.4826 * mad)

	def _resid(c):
		return np.polyval(c, times) - values

	ret = least_squares(_resid, x0=coef0, loss="huber", f_scale=f_scale)
	return ret.x


def _fit_poly_ransac(times: np.ndarray, values: np.ndarray, degree: int, max_trials: int = 250):
	min_samples = degree + 1
	if len(times) < min_samples:
		return np.polyfit(times, values, degree)

	rng = np.random.default_rng(42)
	coef_ref = np.polyfit(times, values, degree)
	res_ref = np.abs(values - np.polyval(coef_ref, times))
	mad = np.median(np.abs(res_ref - np.median(res_ref)))
	threshold = max(1e-3, 2.5 * 1.4826 * mad)

	best_inliers = None
	best_count = -1
	best_err = np.inf
	best_coef = coef_ref

	idx_all = np.arange(len(times))
	for _ in range(int(max_trials)):
		sub_idx = rng.choice(idx_all, size=min_samples, replace=False)
		t_sub = times[sub_idx]
		y_sub = values[sub_idx]
		if np.ptp(t_sub) <= 1e-12:
			continue
		try:
			coef = np.polyfit(t_sub, y_sub, degree)
		except Exception:
			continue

		res = np.abs(values - np.polyval(coef, times))
		inliers = res <= threshold
		count = int(np.sum(inliers))
		err = float(np.mean(res[inliers] ** 2)) if count > 0 else np.inf

		if count > best_count or (count == best_count and err < best_err):
			best_count = count
			best_err = err
			best_inliers = inliers
			best_coef = coef

	if best_inliers is not None and np.sum(best_inliers) >= min_samples and np.ptp(times[best_inliers]) > 1e-12:
		best_coef = np.polyfit(times[best_inliers], values[best_inliers], degree)

	return best_coef


def _fit_poly(times: np.ndarray, values: np.ndarray, degree: int, method: str):
	if method == "ols":
		return _fit_poly_ols(times, values, degree)
	if method == "ransac":
		return _fit_poly_ransac(times, values, degree)
	if method == "huber":
		return _fit_poly_huber(times, values, degree)
	raise ValueError(f"未知拟合方法: {method}")


def _eval_poly_velocity(coef: list[float] | np.ndarray, t: np.ndarray):
	"""对多项式位置函数求导并计算速度值。"""
	coef_arr = np.asarray(coef, dtype=float)
	if coef_arr.size <= 1:
		return np.full_like(t, np.nan, dtype=float)
	deriv_coef = np.polyder(coef_arr)
	return np.polyval(deriv_coef, t)


def _velocity_formula_str(axis_name: str, coef: list[float] | np.ndarray):
	coef_arr = np.asarray(coef, dtype=float)
	if coef_arr.size == 2:
		k = float(coef_arr[0])
		return f"v{axis_name.lower()} = {k:.4f}"
	if coef_arr.size == 3:
		a, b = float(coef_arr[0]), float(coef_arr[1])
		return f"v{axis_name.lower()} = {2*a:.4f} t + {b:.4f}"
	return f"v{axis_name.lower()} = N/A"


def _compute_stats(values: np.ndarray):
	arr = np.asarray(values, dtype=float)
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


def _compute_fit_points_and_velocity(times: np.ndarray, pts: np.ndarray, fit_method: str):
	"""根据给定点计算拟合位置点与拟合速度点（与 times 对齐；支持缺失值）。"""
	fit_pts = np.full_like(pts, np.nan, dtype=float)
	fit_coef = {}

	for i, key in enumerate(["x", "y", "z"]):
		valid = np.isfinite(times) & np.isfinite(pts[:, i])
		tv = times[valid]
		yv = pts[valid, i]

		if i in (0, 1):
			if len(tv) >= 2 and np.ptp(tv) > 1e-12:
				coef = _fit_poly(tv, yv, degree=1, method=fit_method)
				fit_pts[valid, i] = np.polyval(coef, tv)
				fit_coef[key] = [float(v) for v in coef]
		else:
			if len(tv) >= 3 and np.ptp(tv) > 1e-12:
				coef = _fit_poly(tv, yv, degree=2, method=fit_method)
				fit_pts[valid, i] = np.polyval(coef, tv)
				fit_coef[key] = [float(v) for v in coef]

	fit_vel = np.full((len(times), 3), np.nan, dtype=float)
	for i, key in enumerate(["x", "y", "z"]):
		coef = fit_coef.get(key, None)
		if coef is None:
			continue
		fit_vel[:, i] = _eval_poly_velocity(coef, times)

	return fit_pts, fit_vel, fit_coef


def _compute_online_fit_points_and_velocity(times: np.ndarray, pts: np.ndarray, fit_method: str):
	"""在线拟合：第 n 时刻仅使用 [0..n] 的点做拟合，并在 t_n 处读出位置/速度。"""
	n = len(times)
	fit_pts = np.full_like(pts, np.nan, dtype=float)
	fit_vel = np.full((n, 3), np.nan, dtype=float)

	for k in range(n):
		tk = float(times[k])
		for i, key in enumerate(["x", "y", "z"]):
			valid_prefix = np.isfinite(times[: k + 1]) & np.isfinite(pts[: k + 1, i])
			tv = times[: k + 1][valid_prefix]
			yv = pts[: k + 1, i][valid_prefix]

			degree = 1 if key in ("x", "y") else 2
			min_samples = degree + 1
			if len(tv) < min_samples or np.ptp(tv) <= 1e-12:
				continue

			coef = _fit_poly(tv, yv, degree=degree, method=fit_method)
			fit_pts[k, i] = float(np.polyval(coef, tk))
			fit_vel[k, i] = float(_eval_poly_velocity(coef, np.asarray([tk], dtype=float))[0])

	return fit_pts, fit_vel


def _build_err_bundle(kf_pos: np.ndarray, kf_vel: np.ndarray, fit_pts: np.ndarray, fit_vel: np.ndarray):
	err_series = {
		"x": kf_pos[:, 0] - fit_pts[:, 0],
		"y": kf_pos[:, 1] - fit_pts[:, 1],
		"z": kf_pos[:, 2] - fit_pts[:, 2],
		"vx": kf_vel[:, 0] - fit_vel[:, 0],
		"vy": kf_vel[:, 1] - fit_vel[:, 1],
		"vz": kf_vel[:, 2] - fit_vel[:, 2],
	}
	desc_mask = np.isfinite(fit_vel[:, 2]) & (fit_vel[:, 2] <= 0.0)
	err_series_down = {
		"x": err_series["x"][desc_mask],
		"y": err_series["y"][desc_mask],
		"z": err_series["z"][desc_mask],
		"vx": err_series["vx"][desc_mask],
		"vy": err_series["vy"][desc_mask],
		"vz": err_series["vz"][desc_mask],
	}
	return {"all": err_series, "down": err_series_down}


def _collect_global_error_stats(json_files: list[Path], fit_method: str, fit_source: str):
	global_err = {"x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": []}
	global_err_down = {"x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": []}

	for js in json_files:
		points, gt_points, _, times, kf_update_pos, kf_update_vel, _, _, _, _, _, _, _, _ = _load_detection_series(js)
		if not points:
			continue
		det_pts = np.asarray(points, dtype=float)
		gt_pts = np.asarray(gt_points, dtype=float)
		ts = np.asarray(times, dtype=float)
		kf_pos = np.asarray(kf_update_pos, dtype=float)
		kf_vel = np.asarray(kf_update_vel, dtype=float)

		fit_det_pts, fit_det_vel, _ = _compute_fit_points_and_velocity(ts, det_pts, fit_method)
		fit_gt_pts, fit_gt_vel, _ = _compute_fit_points_and_velocity(ts, gt_pts, fit_method)

		if fit_source == "gt" and np.isfinite(fit_gt_pts).any():
			fit_pts, fit_vel = fit_gt_pts, fit_gt_vel
		else:
			fit_pts, fit_vel = fit_det_pts, fit_det_vel

		err_bundle = _build_err_bundle(kf_pos, kf_vel, fit_pts, fit_vel)

		for k in global_err:
			arr = np.asarray(err_bundle["all"].get(k, []), dtype=float)
			arr = arr[np.isfinite(arr)]
			if arr.size > 0:
				global_err[k].extend(arr.tolist())

			arr_down = np.asarray(err_bundle["down"].get(k, []), dtype=float)
			arr_down = arr_down[np.isfinite(arr_down)]
			if arr_down.size > 0:
				global_err_down[k].extend(arr_down.tolist())

	return global_err, global_err_down


def _collect_global_online_det_vs_gt_fit_stats(json_files: list[Path], fit_method: str):
	"""汇总 online detection fit 与 gt fit 的误差统计。"""
	global_err = {"x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": []}
	global_err_down = {"x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": []}

	for js in json_files:
		points, gt_points, _, times, _, _, _, _, _, _, _, online_det_fit_pos, online_det_fit_vel, _ = _load_detection_series(js)
		if not points:
			continue

		det_pts = np.asarray(points, dtype=float)
		gt_pts = np.asarray(gt_points, dtype=float)
		ts = np.asarray(times, dtype=float)

		online_det_fit_pts = np.asarray(online_det_fit_pos, dtype=float)
		online_det_fit_vel = np.asarray(online_det_fit_vel, dtype=float)
		if not np.isfinite(online_det_fit_pts).any() or not np.isfinite(online_det_fit_vel).any():
			online_det_fit_pts, online_det_fit_vel = _compute_online_fit_points_and_velocity(ts, det_pts, fit_method)
		fit_gt_pts, fit_gt_vel, _ = _compute_fit_points_and_velocity(ts, gt_pts, fit_method)

		err_bundle = _build_err_bundle(online_det_fit_pts, online_det_fit_vel, fit_gt_pts, fit_gt_vel)

		for k in global_err:
			arr = np.asarray(err_bundle["all"].get(k, []), dtype=float)
			arr = arr[np.isfinite(arr)]
			if arr.size > 0:
				global_err[k].extend(arr.tolist())

			arr_down = np.asarray(err_bundle["down"].get(k, []), dtype=float)
			arr_down = arr_down[np.isfinite(arr_down)]
			if arr_down.size > 0:
				global_err_down[k].extend(arr_down.tolist())

	return global_err, global_err_down


def _stats_text(stats: dict | None, prefix: str = "err"):
	if stats is None:
		return f"{prefix}: N/A"
	return (
		f"{prefix}={prefix}_kf-fit\n"
		f"max={stats['max']:.4f}\n"
		f"min={stats['min']:.4f}\n"
		f"mean={stats['mean']:.4f}\n"
		f"std={stats['std']:.4f}\n"
		f"|{prefix}| mean={stats['abs_mean']:.4f}\n"
		f"|{prefix}| std={stats['abs_std']:.4f}\n"
		f"n={stats['count']}"
	)


def _stats_text_with_down(stats_all: dict | None, stats_down: dict | None, prefix: str = "err"):
	base = _stats_text(stats_all, prefix=prefix)
	if stats_down is None:
		return base + "\n----\ndown(vz<=0): N/A"
	return (
		base
		+ "\n----\n"
		+ f"down(vz<=0) max={stats_down['max']:.4f}\n"
		+ f"down min={stats_down['min']:.4f}\n"
		+ f"down mean={stats_down['mean']:.4f}\n"
		+ f"down std={stats_down['std']:.4f}\n"
		+ f"down |{prefix}| mean={stats_down['abs_mean']:.4f}\n"
		+ f"down |{prefix}| std={stats_down['abs_std']:.4f}\n"
		+ f"down n={stats_down['count']}"
	)


def _format_stats_console_line(key: str, stats: dict | None):
	"""统一终端统计行格式。"""
	label = f"{key:<2}"
	if stats is None:
		return f"{label}: N/A"
	return (
		f"{label}: "
		f"max={stats['max']:+.6f}, min={stats['min']:+.6f}, "
		f"mean={stats['mean']:+.6f}, std={stats['std']:.6f}, "
		f"abs_mean={stats['abs_mean']:.6f}, abs_std={stats['abs_std']:.6f}, n={stats['count']}"
	)


def _plot_xyz_vs_time(
	json_path: Path,
	output_dir: Path,
	det_pts: np.ndarray,
	gt_pts: np.ndarray,
	kf_update_pos: np.ndarray,
	online_det_fit_pts: np.ndarray,
	normalized_innovation: np.ndarray,
	innovation_mahalanobis2: np.ndarray,
	contour_areas: np.ndarray,
	times: np.ndarray,
	fit_det_pts: np.ndarray,
	fit_det_coef: dict,
	fit_gt_pts: np.ndarray,
	fit_gt_coef: dict,
	fit_source: str,
	fit_method: str,
	show: bool,
):
	fig, axes = plt.subplots(8, 1, figsize=(9, 16), sharex=True)
	labels = ["X", "Y", "Z"]
	det_colors = ["tab:blue", "tab:orange", "tab:green"]
	gt_colors = ["tab:cyan", "goldenrod", "limegreen"]
	fit_det_colors = ["navy", "darkorange", "darkgreen"]
	fit_gt_colors = ["purple", "saddlebrown", "forestgreen"]

	# 用更平滑的 t 网格来画拟合曲线
	t_min = float(np.min(times))
	t_max = float(np.max(times))
	t_fit = np.linspace(t_min, t_max, 200) if t_max > t_min else np.asarray(times, dtype=float)

	fit_formula_text = ["", "", ""]
	pos_axes = [axes[0], axes[2], axes[4]]
	nu_axes = [axes[1], axes[3], axes[5]]

	for i in range(3):
		ax = pos_axes[i]
		valid_det = np.isfinite(det_pts[:, i])
		if np.any(valid_det):
			ax.plot(times[valid_det], det_pts[valid_det, i], color=det_colors[i], linewidth=1.8)
			ax.scatter(times[valid_det], det_pts[valid_det, i], color=det_colors[i], s=14, alpha=0.85, label="detection")

		valid_gt = np.isfinite(gt_pts[:, i])
		if np.any(valid_gt):
			ax.plot(times[valid_gt], gt_pts[valid_gt, i], color=gt_colors[i], linewidth=1.6, alpha=0.9)
			ax.scatter(times[valid_gt], gt_pts[valid_gt, i], color=gt_colors[i], s=14, alpha=0.85, label="gt")

		# 叠加 detection 对应的 KF update 点（时间严格对齐）
		valid_kf = np.isfinite(kf_update_pos[:, i])
		if np.any(valid_kf):
			ax.plot(
				times[valid_kf],
				kf_update_pos[valid_kf, i],
				"-.",
				color="tab:red",
				linewidth=1.6,
				alpha=0.9,
				label="kf update",
			)
			ax.scatter(
				times[valid_kf],
				kf_update_pos[valid_kf, i],
				marker="s",
				color="tab:red",
				s=20,
				alpha=0.9,
			)

		# 仅绘制“当前选择来源”的拟合线
		coef_det = fit_det_coef.get(labels[i].lower(), None)
		coef_gt = fit_gt_coef.get(labels[i].lower(), None)
		if fit_source == "gt":
			coef_for_text = coef_gt
			fit_pts_axis = fit_gt_pts[:, i]
			fit_color = fit_gt_colors[i]
			fit_ls = "-."
			fit_label = f"gt fit ({fit_method})"
			fit_marker = "+"
		else:
			coef_for_text = coef_det
			fit_pts_axis = fit_det_pts[:, i]
			fit_color = fit_det_colors[i]
			fit_ls = "--"
			fit_label = f"det fit ({fit_method})"
			fit_marker = "x"

		if coef_for_text is not None:
			y_fit = np.polyval(coef_for_text, t_fit)
			ax.plot(t_fit, y_fit, fit_ls, color=fit_color, linewidth=2.0, label=fit_label)
			valid_fit = np.isfinite(fit_pts_axis)
			if np.any(valid_fit):
				ax.scatter(times[valid_fit], fit_pts_axis[valid_fit], marker=fit_marker, color=fit_color, s=24, alpha=0.9)

		# 在线 detection 拟合（第 n 时刻用前 n 时刻数据）
		online_axis = np.asarray(online_det_fit_pts[:, i], dtype=float)
		valid_online = np.isfinite(online_axis)
		if np.any(valid_online):
			ax.plot(
				times[valid_online],
				online_axis[valid_online],
				color="tab:pink",
				linewidth=1.4,
				linestyle=":",
				label="det online fit",
			)
			ax.scatter(
				times[valid_online],
				online_axis[valid_online],
				color="tab:pink",
				s=18,
				alpha=0.8,
			)

		if coef_for_text is not None:
			fit_formula_text[i] = f"fit({fit_source}, {fit_method}): {_poly_formula_str(labels[i], np.asarray(coef_for_text, dtype=float))}"

		ax.set_ylabel(labels[i])
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best")

		nu_ax = nu_axes[i]
		valid_nu = np.isfinite(normalized_innovation[:, i])
		if np.any(valid_nu):
			nu_ax.plot(times[valid_nu], normalized_innovation[valid_nu, i], color="tab:purple", linewidth=1.6, label=f"nu_{labels[i].lower()}")
			nu_ax.scatter(times[valid_nu], normalized_innovation[valid_nu, i], color="tab:purple", s=16, alpha=0.9)
		nu_ax.axhline(3.0, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.7, label="3σ gate")
		nu_ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
		nu_ax.set_ylabel(f"nu_{labels[i].lower()}")
		nu_ax.grid(True, alpha=0.3)
		nu_ax.legend(loc="best")

	valid_d2 = np.isfinite(innovation_mahalanobis2)
	if np.any(valid_d2):
		axes[6].plot(times[valid_d2], innovation_mahalanobis2[valid_d2], color="tab:brown", linewidth=1.8, label="d²")
		axes[6].scatter(times[valid_d2], innovation_mahalanobis2[valid_d2], color="tab:brown", s=18, alpha=0.9)
	axes[6].axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
	axes[6].set_ylabel("d²")
	axes[6].grid(True, alpha=0.3)
	axes[6].legend(loc="best")

	valid_area = np.isfinite(contour_areas)
	if np.any(valid_area):
		axes[7].plot(times[valid_area], contour_areas[valid_area], color="tab:olive", linewidth=1.8, label="contour area")
		axes[7].scatter(times[valid_area], contour_areas[valid_area], color="tab:olive", s=14, alpha=0.9)
	axes[7].set_ylabel("area(px²)")
	axes[7].grid(True, alpha=0.3)
	axes[7].legend(loc="best")

	# 位置误差统计：err = kf_update - fit(按fit_source选择)
	fit_at_times = fit_gt_pts if fit_source == "gt" and np.isfinite(fit_gt_pts).any() else fit_det_pts
	desc_mask = np.zeros(len(times), dtype=bool)
	z_coef = (fit_gt_coef.get("z", None) if fit_source == "gt" else fit_det_coef.get("z", None))
	if z_coef is None and fit_source == "gt":
		z_coef = fit_det_coef.get("z", None)
	if z_coef is not None:
		vz_fit = _eval_poly_velocity(z_coef, times)
		desc_mask = np.isfinite(vz_fit) & (vz_fit <= 0.0)

	pos_err_stats = {}
	side_blocks = []
	for i, key in enumerate(["x", "y", "z"]):
		err = kf_update_pos[:, i] - fit_at_times[:, i]
		valid = np.isfinite(err)
		valid_down = valid & desc_mask
		stats_all = _compute_stats(err[valid])
		stats_down = _compute_stats(err[valid_down])
		pos_err_stats[key] = {"all": stats_all, "down": stats_down}
		side_blocks.append(
			f"{labels[i]}\n{fit_formula_text[i] if fit_formula_text[i] else 'fit: N/A'}\n"
			f"{_stats_text_with_down(stats_all, stats_down, prefix='err')}"
		)

	axes[-1].set_xlabel("t (s)")
	fig.suptitle(f"Figure1 | {json_path.stem} | X/nu_x/Y/nu_y/Z/nu_z/d²/area vs t ({fit_method}, stat={fit_source})")
	fig.tight_layout(rect=[0, 0, 0.76, 0.98])
	for i, txt in enumerate(side_blocks):
		y = 0.985 - i * 0.23
		fig.text(
			0.78,
			y,
			txt,
			fontsize=7.5,
			va="top",
			ha="left",
			bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
		)

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{json_path.stem}_figure1_xyz_nu_d2_vs_t_{fit_method}_{fit_source}.png"
	fig.savefig(out_path, dpi=160)
	print(f"[已保存] {out_path}")

	if show:
		plt.show()

	plt.close(fig)
	coef_selected = fit_gt_coef if fit_source == "gt" and len(fit_gt_coef) > 0 else fit_det_coef
	return fit_at_times, coef_selected, pos_err_stats


def _plot_velocity_vs_time(
	json_path: Path,
	output_dir: Path,
	times: np.ndarray,
	fit_det_coef: dict,
	fit_gt_coef: dict,
	kf_update_vel: np.ndarray,
	online_det_fit_vel: np.ndarray,
	fit_det_vel: np.ndarray,
	fit_gt_vel: np.ndarray,
	fit_source: str,
	fit_method: str,
	show: bool,
):
	fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
	labels = ["Vx", "Vy", "Vz"]
	coef_keys = ["x", "y", "z"]
	colors = ["tab:blue", "tab:orange", "tab:green"]

	t_min = float(np.min(times))
	t_max = float(np.max(times))
	t_fit = np.linspace(t_min, t_max, 200) if t_max > t_min else np.asarray(times, dtype=float)

	vel_at_times = fit_gt_vel if fit_source == "gt" and np.isfinite(fit_gt_vel).any() else fit_det_vel
	vel_formula_text = ["", "", ""]

	for i, key in enumerate(coef_keys):
		# 先叠加 detection 对应的 KF update 速度点（时间严格对齐）
		valid_kf = np.isfinite(kf_update_vel[:, i])
		if np.any(valid_kf):
			axes[i].plot(
				times[valid_kf],
				kf_update_vel[valid_kf, i],
				"-.",
				color="tab:red",
				linewidth=1.6,
				alpha=0.9,
				label="kf update",
			)
			axes[i].scatter(
				times[valid_kf],
				kf_update_vel[valid_kf, i],
				marker="s",
				color="tab:red",
				s=20,
				alpha=0.9,
			)

		coef_det = fit_det_coef.get(key, None)
		coef_gt = fit_gt_coef.get(key, None)
		if coef_det is None and coef_gt is None:
			axes[i].text(0.02, 0.90, f"{labels[i]}: 拟合不可用", transform=axes[i].transAxes, fontsize=9)
			axes[i].set_ylabel(labels[i])
			axes[i].grid(True, alpha=0.3)
			continue

		# 仅绘制“当前选择来源”的速度拟合线
		if fit_source == "gt":
			coef_for_text = coef_gt
			fit_color = "purple"
			fit_ls = "-."
			fit_label = f"gt vel ({fit_method})"
			fit_marker = "+"
		else:
			coef_for_text = coef_det
			fit_color = colors[i]
			fit_ls = "--"
			fit_label = f"det vel ({fit_method})"
			fit_marker = "x"

		if coef_for_text is not None:
			v_fit = _eval_poly_velocity(coef_for_text, t_fit)
			v_hat = _eval_poly_velocity(coef_for_text, times)
			axes[i].plot(t_fit, v_fit, color=fit_color, linewidth=2.0, linestyle=fit_ls, label=fit_label)
			valid_fit = np.isfinite(v_hat)
			if np.any(valid_fit):
				axes[i].scatter(times[valid_fit], v_hat[valid_fit], marker=fit_marker, color=fit_color, s=24, alpha=0.9)

		# 在线 detection 拟合速度（第 n 时刻用前 n 时刻数据）
		online_v = np.asarray(online_det_fit_vel[:, i], dtype=float)
		valid_online_v = np.isfinite(online_v)
		if np.any(valid_online_v):
			axes[i].plot(
				times[valid_online_v],
				online_v[valid_online_v],
				color="tab:pink",
				linewidth=1.4,
				linestyle=":",
				label="det online fit vel",
			)
			axes[i].scatter(
				times[valid_online_v],
				online_v[valid_online_v],
				color="tab:pink",
				s=18,
				alpha=0.8,
			)

		if coef_for_text is not None:
			vel_formula_text[i] = _velocity_formula_str(key.upper(), coef_for_text)
		axes[i].set_ylabel(labels[i])
		axes[i].grid(True, alpha=0.3)
		axes[i].legend(loc="best")

	# 速度误差统计：err = kf_update - fit
	desc_mask = np.isfinite(vel_at_times[:, 2]) & (vel_at_times[:, 2] <= 0.0)
	vel_err_stats = {}
	side_blocks = []
	for i, key in enumerate(["vx", "vy", "vz"]):
		err = kf_update_vel[:, i] - vel_at_times[:, i]
		valid = np.isfinite(err)
		valid_down = valid & desc_mask
		stats_all = _compute_stats(err[valid])
		stats_down = _compute_stats(err[valid_down])
		vel_err_stats[key] = {"all": stats_all, "down": stats_down}
		side_blocks.append(
			f"{labels[i]}\n{vel_formula_text[i] if vel_formula_text[i] else 'fit vel: N/A'}\n"
			f"{_stats_text_with_down(stats_all, stats_down, prefix='err')}"
		)

	axes[-1].set_xlabel("t (s)")
	fig.suptitle(f"Figure2 | {json_path.stem} | fitted velocity: Vx/Vy/Vz vs t ({fit_method}, stat={fit_source})")
	fig.tight_layout(rect=[0, 0, 0.76, 0.97])
	for i, txt in enumerate(side_blocks):
		y = 0.97 - i * 0.32
		fig.text(
			0.78,
			y,
			txt,
			fontsize=7.5,
			va="top",
			ha="left",
			bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
		)

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{json_path.stem}_figure2_vxyz_vs_t_{fit_method}_{fit_source}.png"
	fig.savefig(out_path, dpi=160)
	print(f"[已保存] {out_path}")

	if show:
		plt.show()

	plt.close(fig)
	return vel_at_times, vel_err_stats


def _save_fit_records(
	json_path: Path,
	output_dir: Path,
	fit_method: str,
	frame_ids: list[int],
	times: np.ndarray,
	pts: np.ndarray,
	contour_areas: np.ndarray,
	kf_update_pos: np.ndarray,
	kf_update_vel: np.ndarray,
	kf_update_pos_var: np.ndarray,
	kf_update_vel_var: np.ndarray,
	fit_pts: np.ndarray,
	fit_vel: np.ndarray,
	fit_coef: dict,
	err_stats: dict,
	online_det_vs_gt_fit_err_stats: dict | None = None,
):
	record = {
		"source_json": str(json_path),
		"fit_method": fit_method,
		"coefficients": fit_coef,
		"error_stats": err_stats,
		"online_det_vs_gt_fit_error_stats": online_det_vs_gt_fit_err_stats,
		"samples": [],
	}

	for i in range(len(frame_ids)):
		record["samples"].append(
			{
				"frame_index": int(frame_ids[i]),
				"t": float(times[i]),
				"detection_xyz": [float(v) for v in pts[i]],
				"contour_area": float(contour_areas[i]) if np.isfinite(contour_areas[i]) else None,
				"kf_update_xyz": [
					float(kf_update_pos[i, 0]) if np.isfinite(kf_update_pos[i, 0]) else None,
					float(kf_update_pos[i, 1]) if np.isfinite(kf_update_pos[i, 1]) else None,
					float(kf_update_pos[i, 2]) if np.isfinite(kf_update_pos[i, 2]) else None,
				],
				"kf_update_vxyz": [
					float(kf_update_vel[i, 0]) if np.isfinite(kf_update_vel[i, 0]) else None,
					float(kf_update_vel[i, 1]) if np.isfinite(kf_update_vel[i, 1]) else None,
					float(kf_update_vel[i, 2]) if np.isfinite(kf_update_vel[i, 2]) else None,
				],
				"kf_update_pos_var": [
					float(kf_update_pos_var[i, 0]) if np.isfinite(kf_update_pos_var[i, 0]) else None,
					float(kf_update_pos_var[i, 1]) if np.isfinite(kf_update_pos_var[i, 1]) else None,
					float(kf_update_pos_var[i, 2]) if np.isfinite(kf_update_pos_var[i, 2]) else None,
				],
				"kf_update_vel_var": [
					float(kf_update_vel_var[i, 0]) if np.isfinite(kf_update_vel_var[i, 0]) else None,
					float(kf_update_vel_var[i, 1]) if np.isfinite(kf_update_vel_var[i, 1]) else None,
					float(kf_update_vel_var[i, 2]) if np.isfinite(kf_update_vel_var[i, 2]) else None,
				],
				"fit_x": float(fit_pts[i, 0]) if np.isfinite(fit_pts[i, 0]) else None,
				"fit_y": float(fit_pts[i, 1]) if np.isfinite(fit_pts[i, 1]) else None,
				"fit_z": float(fit_pts[i, 2]) if np.isfinite(fit_pts[i, 2]) else None,
				"fit_xyz": [
					float(fit_pts[i, 0]) if np.isfinite(fit_pts[i, 0]) else None,
					float(fit_pts[i, 1]) if np.isfinite(fit_pts[i, 1]) else None,
					float(fit_pts[i, 2]) if np.isfinite(fit_pts[i, 2]) else None,
				],
				"fit_vx": float(fit_vel[i, 0]) if np.isfinite(fit_vel[i, 0]) else None,
				"fit_vy": float(fit_vel[i, 1]) if np.isfinite(fit_vel[i, 1]) else None,
				"fit_vz": float(fit_vel[i, 2]) if np.isfinite(fit_vel[i, 2]) else None,
				"fit_vxyz": [
					float(fit_vel[i, 0]) if np.isfinite(fit_vel[i, 0]) else None,
					float(fit_vel[i, 1]) if np.isfinite(fit_vel[i, 1]) else None,
					float(fit_vel[i, 2]) if np.isfinite(fit_vel[i, 2]) else None,
				],
			}
		)

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{json_path.stem}_fit_points_{fit_method}.json"
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(record, f, ensure_ascii=False, indent=2)
	print(f"[已保存] {out_path}")


def _plot_error_vs_kf_variance(
	json_path: Path,
	output_dir: Path,
	times: np.ndarray,
	err_bundle: dict,
	kf_pos_var: np.ndarray,
	kf_vel_var: np.ndarray,
	fit_method: str,
	fit_source: str,
	show: bool,
):
	"""绘制误差与KF方差(3σ)对照图（同轴对照）。"""
	if not np.isfinite(kf_pos_var).any() and not np.isfinite(kf_vel_var).any():
		print(f"[跳过] {json_path.name}: 无 kf_pos_var/kf_vel_var，未绘制误差-方差对照图")
		return

	fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
	pos_keys = ["x", "y", "z"]
	vel_keys = ["vx", "vy", "vz"]

	for i, key in enumerate(pos_keys):
		ax = axes[i, 0]
		err = np.asarray(err_bundle["all"][key], dtype=float)
		var = np.asarray(kf_pos_var[:, i], dtype=float)
		sigma3 = 3.0 * np.sqrt(np.clip(var, a_min=0.0, a_max=None))

		valid_e = np.isfinite(err)
		if np.any(valid_e):
			ax.plot(times[valid_e], err[valid_e], color="tab:blue", linewidth=1.6, label="err=kf-fit")
		valid_s = np.isfinite(sigma3)
		if np.any(valid_s):
			ax.plot(times[valid_s], sigma3[valid_s], "--", color="tab:red", linewidth=1.2, label="+3σ")
			ax.plot(times[valid_s], -sigma3[valid_s], "--", color="tab:red", linewidth=1.2, label="-3σ")

		ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
		ax.set_ylabel(f"{key} (m)")
		ax.set_title(f"pos {key.upper()}")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=8)

	for i, key in enumerate(vel_keys):
		ax = axes[i, 1]
		err = np.asarray(err_bundle["all"][key], dtype=float)
		var = np.asarray(kf_vel_var[:, i], dtype=float)
		sigma3 = 3.0 * np.sqrt(np.clip(var, a_min=0.0, a_max=None))

		valid_e = np.isfinite(err)
		if np.any(valid_e):
			ax.plot(times[valid_e], err[valid_e], color="tab:blue", linewidth=1.6, label="err=kf-fit")
		valid_s = np.isfinite(sigma3)
		if np.any(valid_s):
			ax.plot(times[valid_s], sigma3[valid_s], "--", color="tab:red", linewidth=1.2, label="+3σ")
			ax.plot(times[valid_s], -sigma3[valid_s], "--", color="tab:red", linewidth=1.2, label="-3σ")

		ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
		ax.set_ylabel(f"{key} (m/s)")
		ax.set_title(f"vel {key.upper()}")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=8)

	axes[-1, 0].set_xlabel("t (s)")
	axes[-1, 1].set_xlabel("t (s)")
	fig.suptitle(f"{json_path.stem}  err vs KF variance(3σ) ({fit_method}, stat={fit_source})")
	fig.tight_layout(rect=[0, 0, 1, 0.96])

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{json_path.stem}_err_vs_kf_sigma_{fit_method}_{fit_source}.png"
	fig.savefig(out_path, dpi=160)
	print(f"[已保存] {out_path}")

	if show:
		plt.show()

	plt.close(fig)


def _plot_kf_variance_vs_time(
	json_path: Path,
	output_dir: Path,
	times: np.ndarray,
	kf_pos_var: np.ndarray,
	kf_vel_var: np.ndarray,
	fit_method: str,
	fit_source: str,
	show: bool,
):
	"""绘制每条轨迹的 KF 方差时序图：x/y/z/vx/vy/vz。"""
	if not np.isfinite(kf_pos_var).any() and not np.isfinite(kf_vel_var).any():
		print(f"[跳过] {json_path.name}: 无 kf_pos_var/kf_vel_var，未绘制方差时序图")
		return

	fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
	pos_keys = ["x", "y", "z"]
	vel_keys = ["vx", "vy", "vz"]

	for i, key in enumerate(pos_keys):
		ax = axes[i, 0]
		var = np.asarray(kf_pos_var[:, i], dtype=float)
		valid = np.isfinite(var)
		if np.any(valid):
			ax.plot(times[valid], var[valid], color="tab:purple", linewidth=1.8, label="kf_pos_var")
			ax.scatter(times[valid], var[valid], color="tab:purple", s=12, alpha=0.85)
		ax.set_ylabel(f"{key} var (m²)")
		ax.set_title(f"pos {key.upper()} variance")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=8)

	for i, key in enumerate(vel_keys):
		ax = axes[i, 1]
		var = np.asarray(kf_vel_var[:, i], dtype=float)
		valid = np.isfinite(var)
		if np.any(valid):
			ax.plot(times[valid], var[valid], color="tab:brown", linewidth=1.8, label="kf_vel_var")
			ax.scatter(times[valid], var[valid], color="tab:brown", s=12, alpha=0.85)
		ax.set_ylabel(f"{key} var ((m/s)²)")
		ax.set_title(f"vel {key.upper()} variance")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=8)

	axes[-1, 0].set_xlabel("t (s)")
	axes[-1, 1].set_xlabel("t (s)")
	fig.suptitle(f"{json_path.stem}  KF variance vs t ({fit_method}, stat={fit_source})")
	fig.tight_layout(rect=[0, 0, 1, 0.96])

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{json_path.stem}_kf_variance_vs_t_{fit_method}_{fit_source}.png"
	fig.savefig(out_path, dpi=160)
	print(f"[已保存] {out_path}")

	if show:
		plt.show()

	plt.close(fig)


def _plot_one_trajectory(json_path: Path, output_dir: Path, fit_method: str, fit_source: str, show: bool = True):
	(
		points,
		gt_points,
		frame_ids,
		times,
		kf_update_pos,
		kf_update_vel,
		kf_update_pos_var,
		kf_update_vel_var,
		normalized_innovation,
		innovation_mahalanobis2,
		contour_areas,
		online_det_fit_pos,
		online_det_fit_vel,
		base_positions,
	) = _load_detection_series(json_path)

	if not points:
		print(f"[跳过] {json_path.name}: 没有 detection_pos")
		return

	det_pts = np.asarray(points, dtype=float)
	gt_pts = np.asarray(gt_points, dtype=float)
	ts = np.asarray(times, dtype=float)
	kf_pos = np.asarray(kf_update_pos, dtype=float)
	kf_vel = np.asarray(kf_update_vel, dtype=float)
	kf_pos_var = np.asarray(kf_update_pos_var, dtype=float)
	kf_vel_var = np.asarray(kf_update_vel_var, dtype=float)
	contour_areas = np.asarray(contour_areas, dtype=float)

	fit_det_pts, fit_det_vel, fit_det_coef = _compute_fit_points_and_velocity(ts, det_pts, fit_method)
	fit_gt_pts, fit_gt_vel, fit_gt_coef = _compute_fit_points_and_velocity(ts, gt_pts, fit_method)
	online_det_fit_pts = np.asarray(online_det_fit_pos, dtype=float)
	online_det_fit_vel = np.asarray(online_det_fit_vel, dtype=float)
	base_pos = np.asarray(base_positions, dtype=float)
	if not np.isfinite(online_det_fit_pts).any() or not np.isfinite(online_det_fit_vel).any():
		online_det_fit_pts, online_det_fit_vel = _compute_online_fit_points_and_velocity(ts, det_pts, fit_method)

	# 0) 单独绘制 base_pos（时间戳与当前序列严格对齐）
	_plot_base_pos_vs_time(
		json_path,
		output_dir=output_dir,
		times=ts,
		base_pos=base_pos,
		show=show,
	)

	use_gt = fit_source == "gt" and np.isfinite(fit_gt_pts).any()
	selected_source = "gt" if use_gt else "detection"
	if fit_source == "gt" and not use_gt:
		print(f"[提示] {json_path.name}: gt_pos不足，误差统计回退到 detection 拟合")

	fit_pts = fit_gt_pts if use_gt else fit_det_pts
	fit_vel = fit_gt_vel if use_gt else fit_det_vel
	fit_coef = fit_gt_coef if use_gt else fit_det_coef

	# 1) 先显示 X/Y/Z 随时间图
	fit_pts, fit_coef, pos_err_stats = _plot_xyz_vs_time(
		json_path,
		output_dir=output_dir,
		det_pts=det_pts,
		gt_pts=gt_pts,
		kf_update_pos=kf_pos,
		online_det_fit_pts=online_det_fit_pts,
		normalized_innovation=np.asarray(normalized_innovation, dtype=float),
		innovation_mahalanobis2=np.asarray(innovation_mahalanobis2, dtype=float),
		contour_areas=contour_areas,
		times=ts,
		fit_det_pts=fit_det_pts,
		fit_det_coef=fit_det_coef,
		fit_gt_pts=fit_gt_pts,
		fit_gt_coef=fit_gt_coef,
		fit_source=selected_source,
		fit_method=fit_method,
		show=show,
	)
	fit_vel, vel_err_stats = _plot_velocity_vs_time(
		json_path,
		output_dir=output_dir,
		times=ts,
		fit_det_coef=fit_det_coef,
		fit_gt_coef=fit_gt_coef,
		kf_update_vel=kf_vel,
		online_det_fit_vel=online_det_fit_vel,
		fit_det_vel=fit_det_vel,
		fit_gt_vel=fit_gt_vel,
		fit_source=selected_source,
		fit_method=fit_method,
		show=show,
	)

	# 在线 detection 拟合 vs gt 拟合 误差统计（全时段 + 下降段）
	online_vs_gt_err_bundle = _build_err_bundle(online_det_fit_pts, online_det_fit_vel, fit_gt_pts, fit_gt_vel)
	online_vs_gt_err_stats = {}
	for key in ["x", "y", "z", "vx", "vy", "vz"]:
		stats_all = _compute_stats(np.asarray(online_vs_gt_err_bundle["all"].get(key, []), dtype=float))
		stats_down = _compute_stats(np.asarray(online_vs_gt_err_bundle["down"].get(key, []), dtype=float))
		online_vs_gt_err_stats[key] = {"all": stats_all, "down": stats_down}

	err_stats = {
		"x": pos_err_stats.get("x", {}),
		"y": pos_err_stats.get("y", {}),
		"z": pos_err_stats.get("z", {}),
		"vx": vel_err_stats.get("vx", {}),
		"vy": vel_err_stats.get("vy", {}),
		"vz": vel_err_stats.get("vz", {}),
	}
	_save_fit_records(
		json_path,
		output_dir=output_dir,
		fit_method=f"{fit_method}:{selected_source}",
		frame_ids=frame_ids,
		times=ts,
		pts=det_pts,
		contour_areas=contour_areas,
		kf_update_pos=kf_pos,
		kf_update_vel=kf_vel,
		kf_update_pos_var=kf_pos_var,
		kf_update_vel_var=kf_vel_var,
		fit_pts=fit_pts,
		fit_vel=fit_vel,
		fit_coef=fit_coef,
		err_stats=err_stats,
		online_det_vs_gt_fit_err_stats=online_vs_gt_err_stats,
	)

	err_bundle = _build_err_bundle(kf_pos, kf_vel, fit_pts, fit_vel)
	_plot_error_vs_kf_variance(
		json_path,
		output_dir=output_dir,
		times=ts,
		err_bundle=err_bundle,
		kf_pos_var=kf_pos_var,
		kf_vel_var=kf_vel_var,
		fit_method=fit_method,
		fit_source=selected_source,
		show=show,
	)
	_plot_kf_variance_vs_time(
		json_path,
		output_dir=output_dir,
		times=ts,
		kf_pos_var=kf_pos_var,
		kf_vel_var=kf_vel_var,
		fit_method=fit_method,
		fit_source=selected_source,
		show=show,
	)

	# 2) 再显示 3D detection 轨迹图
	fig = plt.figure(figsize=(7, 6))
	ax = fig.add_subplot(111, projection="3d")

	# 按时间顺序逐条连接
	ax.plot(det_pts[:, 0], det_pts[:, 1], det_pts[:, 2], color="tab:blue", linewidth=1.8, label="detection path")
	ax.scatter(det_pts[:, 0], det_pts[:, 1], det_pts[:, 2], color="tab:blue", s=18, alpha=0.9, label="detection points")

	valid_gt3 = np.isfinite(gt_pts).all(axis=1)
	if np.any(valid_gt3):
		gt3 = gt_pts[valid_gt3]
		ax.plot(gt3[:, 0], gt3[:, 1], gt3[:, 2], color="tab:green", linewidth=1.6, alpha=0.9, label="gt path")
		ax.scatter(gt3[:, 0], gt3[:, 1], gt3[:, 2], color="tab:green", s=16, alpha=0.85, marker="o", label="gt points")

	# 叠加 detection 对应的 KF update 3D 点（时间严格对齐）
	valid_kf3 = np.isfinite(kf_pos).all(axis=1)
	if np.any(valid_kf3):
		kf3 = kf_pos[valid_kf3]
		ax.plot(
			kf3[:, 0],
			kf3[:, 1],
			kf3[:, 2],
			"-.",
			color="tab:red",
			linewidth=1.8,
			label="kf update",
		)
		ax.scatter(
			kf3[:, 0],
			kf3[:, 1],
			kf3[:, 2],
			marker="s",
			color="tab:red",
			s=24,
			alpha=0.9,
		)

	# 仅绘制“当前选择来源”的3D拟合轨迹
	fit_plot_pts = fit_gt_pts if selected_source == "gt" else fit_det_pts
	fit_plot_color = "purple" if selected_source == "gt" else "magenta"
	fit_plot_ls = "-." if selected_source == "gt" else "--"
	fit_plot_label = f"{selected_source} fit xyz ({fit_method})"
	valid_fit_sel = np.isfinite(fit_plot_pts).all(axis=1)
	if np.any(valid_fit_sel):
		fit3 = fit_plot_pts[valid_fit_sel]
		ax.plot(
			fit3[:, 0],
			fit3[:, 1],
			fit3[:, 2],
			fit_plot_ls,
			color=fit_plot_color,
			linewidth=2.0,
			label=fit_plot_label,
		)

	# 标记起点和终点
	ax.scatter(det_pts[0, 0], det_pts[0, 1], det_pts[0, 2], color="green", s=45, marker="o", label="start")
	ax.scatter(det_pts[-1, 0], det_pts[-1, 1], det_pts[-1, 2], color="red", s=45, marker="^", label="end")

	# 仅少量标注，避免图太乱
	step = max(1, len(frame_ids) // 8)
	for i in range(0, len(frame_ids), step):
		ax.text(det_pts[i, 0], det_pts[i, 1], det_pts[i, 2], f"f{frame_ids[i]}", fontsize=8)

	ax.set_title(f"{json_path.stem} ({fit_method}, stat={selected_source})")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.legend(loc="best")
	ax.grid(True, alpha=0.3)
	_set_equal_aspect_3d(ax, det_pts)
	fig.tight_layout()

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{json_path.stem}_detection_3d_{fit_method}_{selected_source}.png"
	fig.savefig(out_path, dpi=160)
	print(f"[已保存] {out_path}")

	if show:
		plt.show()

	plt.close(fig)

	# 返回每条轨迹用于全局统计的误差序列（err = kf_update - fit）
	return err_bundle


def _print_global_error_stats(
	global_err: dict[str, list[float]],
	title: str,
):
	print(f"\n===== {title} =====", flush=True)
	for key in ["x", "y", "z", "vx", "vy", "vz"]:
		arr = np.asarray(global_err.get(key, []), dtype=float)
		arr = arr[np.isfinite(arr)]
		stats = _compute_stats(arr)
		print(_format_stats_console_line(key, stats), flush=True)


def main():
	parser = argparse.ArgumentParser(description="逐条绘制 trajectory_data 中每条轨迹的 detection 位置")
	parser.add_argument(
		"--trajectory-dir",
		type=Path,
		default=Path("trajectory_data"),
		help="轨迹目录（默认 trajectory_data）",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="输出目录（默认 trajectory_data/detection_plots）",
	)
	parser.add_argument(
		"--fit-method",
		type=str,
		default="ols",
		choices=["ols", "ransac", "huber"],
		help="拟合方法：ols / ransac / huber（默认 ols）",
	)
	parser.add_argument(
		"--fit-source",
		type=str,
		default="gt",
		choices=["detection", "gt"],
		help="选择拟合来源：detection 或 gt（每个图只画所选来源的一条拟合线）",
	)
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="不弹窗，仅保存图片",
	)
	args = parser.parse_args()

	script_dir = Path(__file__).parent
	trajectory_dir = args.trajectory_dir
	if not trajectory_dir.is_absolute():
		trajectory_dir = script_dir / trajectory_dir
	output_dir = args.output_dir or (trajectory_dir / "detection_plots")
	if not output_dir.is_absolute():
		output_dir = script_dir / output_dir

	if not trajectory_dir.exists():
		raise FileNotFoundError(f"轨迹目录不存在: {trajectory_dir}")

	json_files = sorted(trajectory_dir.glob("trajectory_tracker*_*.json"))
	if not json_files:
		print(f"未找到轨迹文件: {trajectory_dir}")
		return

	print(f"共找到 {len(json_files)} 条轨迹。先打印全局汇总，再逐条绘制。", flush=True)
	# 先计算并打印全局统计，再逐条出图
	global_err, global_err_down = _collect_global_error_stats(json_files, args.fit_method, "gt")
	_print_global_error_stats(
		global_err,
		title="所有轨迹汇总误差统计（kf_update - gt_fit，全时段）",
	)
	_print_global_error_stats(
		global_err_down,
		title="所有轨迹汇总误差统计（kf_update - gt_fit，仅下降段 vz<=0）",
	)
	global_online_gt_err, global_online_gt_err_down = _collect_global_online_det_vs_gt_fit_stats(
		json_files, args.fit_method
	)
	_print_global_error_stats(
		global_online_gt_err,
		title="所有轨迹汇总误差统计（online_det_fit - gt_fit，全时段）",
	)
	_print_global_error_stats(
		global_online_gt_err_down,
		title="所有轨迹汇总误差统计（online_det_fit - gt_fit，仅下降段 vz<=0）",
	)
	print("\n[开始逐条绘制并保存图像]", flush=True)

	for js in json_files:
		_plot_one_trajectory(
			js,
			output_dir=output_dir,
			fit_method=args.fit_method,
			fit_source=args.fit_source,
			show=not args.no_show,
		)

	print("全部完成。")


if __name__ == "__main__":
	main()
