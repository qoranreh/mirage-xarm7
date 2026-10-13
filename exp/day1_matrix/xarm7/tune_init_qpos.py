#!/usr/bin/env python
"""
tune_init_qpos.py — xArm7 의 init_qpos 를 Panda 의 초기 grip_site 자세(Lift, table arena)에 맞춘다.

  python tune_init_qpos.py            # 결과만 출력
  python tune_init_qpos.py --write    # xarm7_robot.py 의 XARM7_INIT_QPOS 갱신

목표(베이스 기준): Panda 초기 grip_site 위치 (0.466, 0.004, 0.100) m, 자세 x→+Y, y→+X, z→-Z (손가락 개폐축 x 가 월드 +Y,
아래를 향함). 여러 초기값에서 scipy least_squares 로 관절각을 풀고, 관절 한계 안에서 오차가 가장 작은 해를 고른다.
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

TARGET_POS_REL = np.array([0.466, 0.004, 0.100])           # Panda grip_site - base, Lift 초기
TARGET_R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)  # columns = grip x, y, z (world)
ROBOT_PY = Path.home() / "hayeon/mirage/robosuite/robosuite/models/robots/manipulators/xarm7_robot.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    import robosuite
    from robosuite import load_controller_config
    env = robosuite.make("Lift", robots="XArm7", controller_configs=load_controller_config(default_controller="OSC_POSE"),
                         has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    env.reset()
    r = env.robots[0]; sim = env.sim; m, d = sim.model, sim.data
    qadr = [m.get_joint_qpos_addr(j) for j in r.robot_joints]
    sid = m.site_name2id(r.controller.eef_name)
    base = d.body_xpos[m.body_name2id(r.robot_model.root_body)].copy()
    lo, hi = m.jnt_range[[m.joint_name2id(j) for j in r.robot_joints]].T
    target_pos = base + TARGET_POS_REL
    print("base", np.round(base, 3), "target grip_site", np.round(target_pos, 3))

    def fk(q):
        d.qpos[qadr] = q; sim.forward()
        return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()

    def resid(q):
        p, R = fk(q)
        # j1/j3/j5 -> 0 약한 정규화: 팔꿈치가 옆으로 빠지지 않는 대칭 자세 선호
        return np.concatenate([(p - target_pos) * 10.0, (R - TARGET_R).flatten() * 1.0, q[[0, 2, 4]] * 0.3])

    seeds = [np.array([0, -0.247, 0, 0.909, 0, 1.156, 0]), np.array([0, 0.2, 0, 1.2, 0, 1.0, 0]),
             np.array([0, 0.5, 0, 1.5, 0, 1.0, 0]), np.array([0, -0.5, 0, 0.6, 0, 1.1, 0]),
             np.array([0, 0.3, 0, 1.8, 0, 1.5, 0]), np.array([0, 0.0, 0, 1.0, 0, 1.0, 0])]
    best = None
    for s in seeds:
        s = np.clip(s, lo + 1e-3, hi - 1e-3)
        sol = least_squares(resid, s, bounds=(lo, hi), max_nfev=2000)
        p, R = fk(sol.x)
        pos_err = np.linalg.norm(p - target_pos); ori_err = np.linalg.norm(R - TARGET_R)
        print(f"  seed {np.round(s,2)} -> pos_err={pos_err*1000:.1f}mm ori_err={ori_err:.3f} q={np.round(sol.x,3)}")
        if best is None or sol.cost < best[0]:
            best = (sol.cost, sol.x, pos_err, ori_err)
    _, q, pos_err, ori_err = best
    p, R = fk(q)
    print("\nBEST q =", np.round(q, 4).tolist())
    print(f"grip_site pos {np.round(p,3)} (target {np.round(target_pos,3)}), pos_err {pos_err*1000:.1f} mm, ori_err {ori_err:.3f}")
    print("grip axes: x", np.round(R[:, 0], 2), "y", np.round(R[:, 1], 2), "z", np.round(R[:, 2], 2))
    print("joint margins to limits (rad):", np.round(np.minimum(q - lo, hi - q), 2))
    env.close()

    if args.write:
        s = ROBOT_PY.read_text()
        new = "XARM7_INIT_QPOS = np.array([" + ", ".join(f"{v:.4f}" for v in q) + "])"
        s2, n = re.subn(r"XARM7_INIT_QPOS = np\.array\(\[[^\]]*\]\)", new, s)
        assert n == 1; ROBOT_PY.write_text(s2); print("wrote", ROBOT_PY.name, "->", new)


if __name__ == "__main__":
    main()
