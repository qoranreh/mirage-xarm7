"""XArm7 Lift 스모크: 모델 컴파일, reset/step, grip_site 자세, 손가락 개폐축, 그리퍼 열고 닫기, 관절 한계 여유."""
import numpy as np, robosuite, time
from robosuite import load_controller_config
t0 = time.time()
env = robosuite.make("Lift", robots="XArm7", controller_configs=load_controller_config(default_controller="OSC_POSE"),
                     has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, control_freq=20)
obs = env.reset(); r = env.robots[0]; m, d = env.sim.model, env.sim.data; g = r.gripper
print(f"make+reset {time.time()-t0:.1f}s | gripper={type(g).__name__} dof={g.dof} action_dim={env.action_dim} | actuators={r.robot_model.actuators}")
base = d.body_xpos[m.body_name2id(r.robot_model.root_body)]; sid = m.site_name2id(r.controller.eef_name)
R = d.site_xmat[sid].reshape(3, 3)
print(f"base={np.round(base,3)} grip_site={np.round(d.site_xpos[sid],3)} rel={np.round(d.site_xpos[sid]-base,3)}  (Panda rel: 0.466 0.004 0.100)")
print(f"grip axes x={np.round(R[:,0],2)} y={np.round(R[:,1],2)} z={np.round(R[:,2],2)}   (Panda: x~+Y y~+X z~-Z)")
lp, rp = g.important_geoms["left_fingerpad"][0], g.important_geoms["right_fingerpad"][0]
pl, pr = d.geom_xpos[m.geom_name2id(lp)], d.geom_xpos[m.geom_name2id(rp)]
ax = (pr - pl) / np.linalg.norm(pr - pl); print(f"finger axis in grip frame = {np.round(R.T@ax,3)} (expect +-x)  pad gap open={np.linalg.norm(pr-pl):.3f} m")
lo, hi = m.jnt_range[[m.joint_name2id(j) for j in r.robot_joints]].T
print("qpos", np.round(r._joint_positions, 3), "margins", np.round(np.minimum(r._joint_positions-lo, hi-r._joint_positions), 2))
print("obs keys:", sorted(k for k in obs if k.startswith("robot0")), "| eef_pos", np.round(obs["robot0_eef_pos"],3), "eef_quat", np.round(obs["robot0_eef_quat"],3), "gripper_qpos", np.round(obs["robot0_gripper_qpos"],3))
# 정지 액션 20스텝: 드리프트 확인
p0 = d.site_xpos[sid].copy()
for _ in range(20): obs, rew, done, _ = env.step(np.zeros(env.action_dim))
print(f"20 zero-steps drift = {np.linalg.norm(d.site_xpos[sid]-p0)*1000:.1f} mm")
# 그리퍼 닫기 60스텝 / 열기 60스텝
for _ in range(60): obs, *_ = env.step(np.r_[np.zeros(6), 1.0])
pl, pr = d.geom_xpos[m.geom_name2id(lp)], d.geom_xpos[m.geom_name2id(rp)]; print(f"after close: pad gap={np.linalg.norm(pr-pl):.3f} m gripper_qpos={np.round(obs['robot0_gripper_qpos'],3)}")
for _ in range(60): obs, *_ = env.step(np.r_[np.zeros(6), -1.0])
pl, pr = d.geom_xpos[m.geom_name2id(lp)], d.geom_xpos[m.geom_name2id(rp)]; print(f"after open : pad gap={np.linalg.norm(pr-pl):.3f} m")
# 아래로 5cm 이동 명령 (delta) 10스텝 → 추종 확인
p0 = d.site_xpos[sid].copy()
for _ in range(10): obs, *_ = env.step(np.array([0, 0, -1.0, 0, 0, 0, -1.0]))
print(f"down cmd 10 steps: dz = {(d.site_xpos[sid]-p0)[2]*1000:.1f} mm (expect ~ -50)")
env.close(); print("SMOKE OK")
