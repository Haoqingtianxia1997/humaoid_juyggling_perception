from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class FrameData:
	frame_index: int
	timestamp: float
	base_pos: np.ndarray
	base_rot: np.ndarray
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
	sim_ball_world_pos: np.ndarray
	sim_ball_world_vel: np.ndarray
	pos_error_xyz: np.ndarray
	vel_error_xyz: np.ndarray


@dataclass
class TrajectoryData:
	name: str
	frames: list[FrameData]
	base_pos_ref: np.ndarray
	init_update_index: int
	predict_samples: list[PredictSample]


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


def parse_trajectory_json(json_path: Path) -> list[FrameData]:
	with open(json_path, "r", encoding="utf-8") as f:
		obj = json.load(f)
	frames_raw = obj.get("frames", [])
	frames: list[FrameData] = []
	for i, fr in enumerate(frames_raw):
		base_pos = np.asarray(fr.get("body_pos", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
		base_rot = np.asarray(fr.get("body_rot", np.eye(3)), dtype=float).reshape(3, 3)
		kf_pos = fr.get("kf_pos", None)
		kf_vel = fr.get("kf_vel", None)
		frames.append(
			FrameData(
				frame_index=int(fr.get("frame_index", i)),
				timestamp=float(fr.get("timestamp", 0.0)),
				base_pos=base_pos,
				base_rot=base_rot,
				kf_pos_body=None if kf_pos is None else np.asarray(kf_pos, dtype=float).reshape(3),
				kf_vel_body=None if kf_vel is None else np.asarray(kf_vel, dtype=float).reshape(3),
				kf_state=str(fr.get("kf_state", "")).lower(),
				has_detection=bool(fr.get("has_detection", False)),
			)
		)
	frames.sort(key=lambda x: x.frame_index)
	return frames


def find_init_update_index(frames: list[FrameData], first_predict_idx: int) -> int | None:
	descending_updates = [
		i
		for i in range(first_predict_idx)
		if frames[i].kf_state == "update"
		and frames[i].kf_vel_body is not None
		and frames[i].kf_vel_body[2] < 0.0
	]
	if descending_updates:
		return descending_updates[-1]
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
) -> TrajectoryData | None:
	frames = parse_trajectory_json(json_path)
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
	init_fr = frames[init_idx]
	base_pos_ref = frames[0].base_pos.copy()
	if init_fr.kf_pos_body is None or init_fr.kf_vel_body is None:
		return None

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
	predict_samples.append(
		PredictSample(
			frame_index=init_fr.frame_index,
			timestamp=init_fr.timestamp,
			imu_world_pos=init_imu_world_pos.copy(),
			base_rot=init_fr.base_rot.copy(),
			kf_world_pos=init_kf_world_pos,
			kf_world_vel=init_kf_world_vel,
			sim_ball_world_pos=init_sim_ball_world_pos,
			sim_ball_world_vel=init_sim_ball_world_vel,
			pos_error_xyz=(init_kf_world_pos - init_sim_ball_world_pos).copy(),
			vel_error_xyz=(init_kf_world_vel - init_sim_ball_world_vel).copy(),
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
				sim_ball_world_pos=sim_ball_world_pos,
				sim_ball_world_vel=sim_ball_world_vel,
				pos_error_xyz=pos_error_xyz,
				vel_error_xyz=vel_error_xyz,
			)
		)

	return TrajectoryData(
		name=json_path.name,
		frames=frames,
		base_pos_ref=base_pos_ref,
		init_update_index=init_idx,
		predict_samples=predict_samples,
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
		)
		if td is not None and td.predict_samples:
			out.append(td)
	return out


def run(model_path: Path, trajectory_dir: Path) -> None:
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
	)
	if not trajectories:
		raise RuntimeError(f"未找到可用轨迹：{trajectory_dir}")

	print(f"已加载模型: {model_path}")
	print(f"轨迹目录: {trajectory_dir}")
	print(f"可用轨迹数: {len(trajectories)}")
	print("按键: ←/→=前后帧, ↑/↓=前后轨迹, Space=暂停/播放（默认暂停）")

	state = {
		"traj_i": 0,
		"sample_i": 0,
		"playing": False,
		"pending": [],
		"last_wall": time.monotonic(),
		"request_exit": False,
	}

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
		print(
			"  error (KF - Sim): "
			"pos_err_xyz=("
			f"{smp.pos_error_xyz[0]:+.4f}, {smp.pos_error_xyz[1]:+.4f}, {smp.pos_error_xyz[2]:+.4f}) m, "
			"vel_err_xyz=("
			f"{smp.vel_error_xyz[0]:+.4f}, {smp.vel_error_xyz[1]:+.4f}, {smp.vel_error_xyz[2]:+.4f}) m/s"
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
	args = parser.parse_args()

	model_path = resolve_model_path(args.model)
	trajectory_dir = resolve_trajectory_dir(args.trajectory_dir)
	run(model_path=model_path, trajectory_dir=trajectory_dir)


if __name__ == "__main__":
	main()
