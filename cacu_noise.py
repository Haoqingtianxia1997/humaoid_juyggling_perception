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


def _load_detection_series(json_path: Path):
	"""读取单条轨迹 detection/gt 点，返回与 KF update 对齐的序列。"""
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	frames = sorted(data.get("frames", []), key=lambda x: x.get("frame_index", 0))
	dt = float(data.get("dt", 1.0 / 60.0))
	start_ts = data.get("start_timestamp", None)

	points = []
	gt_points = []
	frame_ids = []
	times = []
	kf_update_pos = []
	kf_update_vel = []
	for fr in frames:
		det = fr.get("detection_pos", None)
		if det is None:
			continue
		if len(det) < 3:
			continue
		points.append(np.asarray(det[:3], dtype=float))
		gt = fr.get("gt_pos", None)
		if gt is not None and len(gt) >= 3:
			gt_points.append(np.asarray(gt[:3], dtype=float))
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

		# detection 对应的 KF update（仅 kf_state=update 的点）
		state = fr.get("kf_state", None)
		kp = fr.get("kf_pos", None)
		kv = fr.get("kf_vel", None)
		if state == "update" and kp is not None and kv is not None and len(kp) >= 3 and len(kv) >= 3:
			kf_update_pos.append(np.asarray(kp[:3], dtype=float))
			kf_update_vel.append(np.asarray(kv[:3], dtype=float))
		else:
			kf_update_pos.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))
			kf_update_vel.append(np.asarray([np.nan, np.nan, np.nan], dtype=float))

	return points, gt_points, frame_ids, times, kf_update_pos, kf_update_vel


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
		points, gt_points, _, times, kf_update_pos, kf_update_vel = _load_detection_series(js)
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


def _plot_xyz_vs_time(
	json_path: Path,
	output_dir: Path,
	det_pts: np.ndarray,
	gt_pts: np.ndarray,
	kf_update_pos: np.ndarray,
	times: np.ndarray,
	fit_det_pts: np.ndarray,
	fit_det_coef: dict,
	fit_gt_pts: np.ndarray,
	fit_gt_coef: dict,
	fit_source: str,
	fit_method: str,
	show: bool,
):
	fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
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

	for i in range(3):
		valid_det = np.isfinite(det_pts[:, i])
		if np.any(valid_det):
			axes[i].plot(times[valid_det], det_pts[valid_det, i], color=det_colors[i], linewidth=1.8)
			axes[i].scatter(times[valid_det], det_pts[valid_det, i], color=det_colors[i], s=14, alpha=0.85, label="detection")

		valid_gt = np.isfinite(gt_pts[:, i])
		if np.any(valid_gt):
			axes[i].plot(times[valid_gt], gt_pts[valid_gt, i], color=gt_colors[i], linewidth=1.6, alpha=0.9)
			axes[i].scatter(times[valid_gt], gt_pts[valid_gt, i], color=gt_colors[i], s=14, alpha=0.85, label="gt")

		# 叠加 detection 对应的 KF update 点（时间严格对齐）
		valid_kf = np.isfinite(kf_update_pos[:, i])
		if np.any(valid_kf):
			axes[i].plot(
				times[valid_kf],
				kf_update_pos[valid_kf, i],
				"-.",
				color="tab:red",
				linewidth=1.6,
				alpha=0.9,
				label="kf update",
			)
			axes[i].scatter(
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
			axes[i].plot(t_fit, y_fit, fit_ls, color=fit_color, linewidth=2.0, label=fit_label)
			valid_fit = np.isfinite(fit_pts_axis)
			if np.any(valid_fit):
				axes[i].scatter(times[valid_fit], fit_pts_axis[valid_fit], marker=fit_marker, color=fit_color, s=24, alpha=0.9)

		if coef_for_text is not None:
			fit_formula_text[i] = f"fit({fit_source}, {fit_method}): {_poly_formula_str(labels[i], np.asarray(coef_for_text, dtype=float))}"

		axes[i].set_ylabel(labels[i])
		axes[i].grid(True, alpha=0.3)
		axes[i].legend(loc="best")

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
	fig.suptitle(f"{json_path.stem}  detection+gt: X/Y/Z vs t ({fit_method}, stat={fit_source})")
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
	out_path = output_dir / f"{json_path.stem}_xyz_vs_t_{fit_method}_{fit_source}.png"
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
	fig.suptitle(f"{json_path.stem}  fitted velocity: Vx/Vy/Vz vs t ({fit_method}, stat={fit_source})")
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
	out_path = output_dir / f"{json_path.stem}_vxyz_vs_t_{fit_method}_{fit_source}.png"
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
	kf_update_pos: np.ndarray,
	kf_update_vel: np.ndarray,
	fit_pts: np.ndarray,
	fit_vel: np.ndarray,
	fit_coef: dict,
	err_stats: dict,
):
	record = {
		"source_json": str(json_path),
		"fit_method": fit_method,
		"coefficients": fit_coef,
		"error_stats": err_stats,
		"samples": [],
	}

	for i in range(len(frame_ids)):
		record["samples"].append(
			{
				"frame_index": int(frame_ids[i]),
				"t": float(times[i]),
				"detection_xyz": [float(v) for v in pts[i]],
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


def _plot_one_trajectory(json_path: Path, output_dir: Path, fit_method: str, fit_source: str, show: bool = True):
	points, gt_points, frame_ids, times, kf_update_pos, kf_update_vel = _load_detection_series(json_path)

	if not points:
		print(f"[跳过] {json_path.name}: 没有 detection_pos")
		return

	det_pts = np.asarray(points, dtype=float)
	gt_pts = np.asarray(gt_points, dtype=float)
	ts = np.asarray(times, dtype=float)
	kf_pos = np.asarray(kf_update_pos, dtype=float)
	kf_vel = np.asarray(kf_update_vel, dtype=float)

	fit_det_pts, fit_det_vel, fit_det_coef = _compute_fit_points_and_velocity(ts, det_pts, fit_method)
	fit_gt_pts, fit_gt_vel, fit_gt_coef = _compute_fit_points_and_velocity(ts, gt_pts, fit_method)

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
		fit_det_vel=fit_det_vel,
		fit_gt_vel=fit_gt_vel,
		fit_source=selected_source,
		fit_method=fit_method,
		show=show,
	)
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
		kf_update_pos=kf_pos,
		kf_update_vel=kf_vel,
		fit_pts=fit_pts,
		fit_vel=fit_vel,
		fit_coef=fit_coef,
		err_stats=err_stats,
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
	return _build_err_bundle(kf_pos, kf_vel, fit_pts, fit_vel)


def _print_global_error_stats(global_err: dict[str, list[float]], title: str):
	print(f"\n===== {title}（err = kf_update - fit） =====")
	for key in ["x", "y", "z", "vx", "vy", "vz"]:
		arr = np.asarray(global_err.get(key, []), dtype=float)
		arr = arr[np.isfinite(arr)]
		stats = _compute_stats(arr)
		if stats is None:
			print(f"{key}: N/A")
			continue
		print(
			f"{key}: max={stats['max']:.6f}, min={stats['min']:.6f}, "
			f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
			f"abs_mean={stats['abs_mean']:.6f}, abs_std={stats['abs_std']:.6f}, n={stats['count']}"
		)


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

	print(
		f"共找到 {len(json_files)} 条轨迹，开始逐条绘制（fit={args.fit_method}, stat_source={args.fit_source}）..."
	)
	# 先计算并打印全局统计，再逐条出图
	global_err, global_err_down = _collect_global_error_stats(json_files, args.fit_method, args.fit_source)
	_print_global_error_stats(global_err, title="所有轨迹汇总误差统计（全时段）")
	_print_global_error_stats(global_err_down, title="所有轨迹汇总误差统计（仅下降段 vz<=0）")

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
