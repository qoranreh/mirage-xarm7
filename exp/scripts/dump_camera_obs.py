#!/usr/bin/env python
"""
dump_camera_obs.py — robosuite Lift 환경에서 agentview RGB(84x84) + segmentation mask 한 프레임 덤프

목적 (p1):
  (1) 지정 로봇 env 가 이미지 obs 를 정상으로 내는지
  (2) seg mask 에서 로봇 픽셀이 물체·배경과 구분되는지
  (3) 이후 단계에서 쓸 obs 키 이름 확정

사용:
  source ~/venvs/mirage/bin/activate
  python dump_camera_obs.py --robots Panda            # 기본: --gl auto, --out ~/hayeon/dumps/day1
  python dump_camera_obs.py --robots Panda xArm7      # 미등록 로봇은 등록 목록을 출력하고 건너뜀
  python dump_camera_obs.py --robots Panda --gl osmesa

GL 백엔드:
  --gl auto  : egl 을 서브프로세스로 먼저 시험하고, 실패하면 osmesa 로 폴백. 둘 다 실패하면 진단 출력 후 종료(코드 2)
  robosuite 는 import 시점에 MUJOCO_GL 을 읽어 렌더 컨텍스트 클래스를 고정하므로(binding_utils.py:37-60),
  백엔드 결정과 환경변수 설정은 반드시 robosuite import 보다 먼저 끝나야 한다. 그래서 시험은 서브프로세스로 한다.
"""
import argparse
import ctypes.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_OUT = Path.home() / "hayeon" / "dumps" / "day1"
SEG_LEVELS_DEFAULT = ["instance", "class", "element", "robot_only"]  # robot_only 는 mirage-robosuite(xembody) 포크 전용


# --------------------------------------------------------------------------- GL 백엔드 선택
def probe_backend(name: str, timeout: int = 120):
    """mujoco.<name> 모듈을 별도 프로세스에서 import 해 본다. 성공 여부와 마지막 stderr 줄을 돌려준다."""
    env = dict(os.environ, MUJOCO_GL=name)
    if name == "osmesa":
        env["PYOPENGL_PLATFORM"] = "osmesa"
    # import 만으로는 부족하다: OSMesa 는 라이브러리가 없어도 import 가 통과할 수 있으므로 실제로 컨텍스트를 만들어 본다.
    code = (
        f"from mujoco.{name} import GLContext\n"
        "ctx = GLContext(max_width=16, max_height=16)\n"
        "ctx.make_current()\n"
        "ctx.free()\n"
        "print('ok')\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    last = (r.stderr.strip().splitlines() or [""])[-1]
    return r.returncode == 0, last


def choose_backend(requested: str) -> str:
    candidates = ["egl", "osmesa"] if requested == "auto" else [requested]
    failures = {}
    for name in candidates:
        ok, msg = probe_backend(name)
        print(f"[gl] probe {name:7s} -> {'OK' if ok else 'FAIL'}  {'' if ok else msg}", flush=True)
        if ok:
            return name
        failures[name] = msg
    print("\n[gl] 사용 가능한 오프스크린 GL 백엔드가 없습니다.", file=sys.stderr)
    for lib in ("EGL", "OSMesa", "GL", "GLX"):
        print(f"      find_library({lib!r:9s}) = {ctypes.util.find_library(lib)}", file=sys.stderr)
    print(f"      LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH', '')}", file=sys.stderr)
    for name, msg in failures.items():
        print(f"      {name}: {msg}", file=sys.stderr)
    print("      해결: libEGL(NVIDIA) 또는 libOSMesa 를 LD_LIBRARY_PATH 에 올린 뒤 재실행", file=sys.stderr)
    sys.exit(2)


# --------------------------------------------------------------------------- 유틸
def describe(v):
    try:
        import numpy as np
        a = np.asarray(v)
        extra = ""
        if a.size and np.issubdtype(a.dtype, np.number):
            extra = f"min={a.min():.4g} max={a.max():.4g}"
        return f"{str(tuple(a.shape)):16s} {str(a.dtype):9s} {extra}"
    except Exception:
        return f"{type(v).__name__}"


def colorize(seg2d, n_colors=20):
    """int mask -> RGB uint8. id 0 은 검정, 나머지는 tab20 순환."""
    import numpy as np
    from matplotlib import colormaps
    cmap = colormaps["tab20"]
    s = seg2d - min(int(seg2d.min()), 0)  # element 레벨의 배경(-1)을 0 으로 밀어 검정으로 (npy 에는 원본 id 저장)
    lut = (np.array([cmap(i % n_colors)[:3] for i in range(int(s.max()) + 1)]) * 255).astype(np.uint8)
    lut[0] = 0
    return lut[s]


# --------------------------------------------------------------------------- 로봇 하나 덤프
def dump_robot(robot: str, seg_levels, out: Path, gl: str, gpu_id: int, seed: int = 0) -> bool:
    import numpy as np
    import imageio.v2 as imageio
    import robosuite
    if getattr(robosuite, "__file__", None) is None:
        # cwd 가 ~/hayeon/mirage(리포 루트)면 그 안의 robosuite/ 디렉터리가 namespace 패키지로 잡혀 진짜 패키지를 가린다
        print(f"  !! robosuite 가 namespace 패키지로 잡혔습니다 (cwd={os.getcwd()}). 리포 루트가 아닌 곳에서 실행하세요.")
        sys.exit(3)
    from robosuite import load_controller_config
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    from robosuite.robots import ROBOT_CLASS_MAPPING

    print(f"\n{'=' * 70}\n[{robot}]")
    if robot not in REGISTERED_ROBOTS or robot not in ROBOT_CLASS_MAPPING:
        print(f"  !! '{robot}' 은 등록되어 있지 않습니다. 추측으로 진행하지 않고 건너뜁니다.")
        print(f"     REGISTERED_ROBOTS   : {sorted(REGISTERED_ROBOTS)}")
        print(f"     ROBOT_CLASS_MAPPING : {sorted(ROBOT_CLASS_MAPPING)}")
        return False

    cfg = load_controller_config(default_controller="OSC_POSE")
    t0 = time.time()
    env = robosuite.make(
        "Lift",
        robots=robot,
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        camera_segmentations=[list(seg_levels)],  # list-of-list = 카메라별 지정 (카메라 1개)
        render_gpu_device_id=gpu_id,
        control_freq=20,
        ignore_done=True,
    )
    r = env.robots[0]
    print(f"  make {time.time() - t0:.1f}s | gripper={type(r.gripper).__name__} controller={r.controller.name} dof={r.dof}")

    np.random.seed(seed)  # placement sampler 가 np.random 을 쓰므로 큐브 위치 재현용
    t0 = time.time()
    obs = env.reset()
    print(f"  reset {time.time() - t0:.1f}s")

    # B. obs 키 전부
    print("  --- obs keys ---")
    key_table = {}
    for k in sorted(obs):
        line = describe(obs[k])
        key_table[k] = line.split()
        print(f"    {k:36s} {line}")

    # seg id -> 이름 (robot_env._create_segementation_sensor 와 같은 순서로 enumerate)
    id_maps = {
        "instance": {i: n for i, n in enumerate(env.model.instances_to_ids.keys())},
        "class": {i: n for i, n in enumerate(env.model.classes_to_ids.keys())},
        "robot_only": {0: "other", 1: "robot+gripper"},
        # element 는 mujoco 원시 geom id (배경 = -1). 이름은 geom_id2name.
        "element": {i: (env.sim.model.geom_id2name(i) or f"geom{i}") for i in range(env.sim.model.ngeom)},
    }
    robot_class_names = {robot, type(r.gripper).__name__}

    out.mkdir(parents=True, exist_ok=True)
    stem = f"{robot}_agentview"
    summary = {
        "robot": robot, "gripper": type(r.gripper).__name__, "controller": r.controller.name,
        "gl_backend": gl, "robosuite": robosuite.__version__, "seed": seed,
        "obs_keys": {k: {"shape": describe(obs[k]).split()[0], "dtype": describe(obs[k]).split()[1]} for k in sorted(obs)},
        "segmentations": {},
    }

    # C. RGB — robosuite 기본 IMAGE_CONVENTION="opengl" 은 원점이 좌하단이라 그대로 저장하면 상하 반전. flipud 로 보정.
    rgb = obs["agentview_image"]
    assert rgb.shape == (84, 84, 3) and rgb.dtype == np.uint8, f"unexpected rgb {rgb.shape} {rgb.dtype}"
    imageio.imwrite(out / f"{stem}_rgb.png", np.flipud(rgb))
    print(f"  saved {stem}_rgb.png  (flipud 적용)")

    # C. seg — unique/픽셀수 출력, 컬러맵 PNG + 원본 id .npy
    for level in seg_levels:
        key = f"agentview_segmentation_{level}"
        if key not in obs:
            print(f"  !! obs 에 {key} 없음 — 이 버전이 '{level}' 을 지원하지 않는 것으로 보임")
            continue
        seg = np.asarray(obs[key])[..., 0]
        ids, counts = np.unique(seg, return_counts=True)
        names = id_maps.get(level, {})
        print(f"  --- {key}  shape={obs[key].shape} dtype={obs[key].dtype} ---")
        robot_px = 0
        rows = []
        for i, c in zip(ids.tolist(), counts.tolist()):
            n = names.get(i, "background" if i < 0 else "?")
            is_robot = (level == "robot_only" and i == 1) or (n in robot_class_names) or any(
                s in str(n).lower() for s in ("robot", "gripper", "panda", "xarm"))
            robot_px += c if is_robot else 0
            rows.append({"id": i, "name": n, "pixels": c, "robot": bool(is_robot)})
            print(f"    id={i:4d}  px={c:5d}  {'[ROBOT] ' if is_robot else '        '}{n}")
        print(f"    robot pixels: {robot_px} / {seg.size} = {100.0 * robot_px / seg.size:.1f}%")
        imageio.imwrite(out / f"{stem}_seg_{level}.png", np.flipud(colorize(seg)))
        np.save(out / f"{stem}_seg_{level}.npy", np.flipud(seg))
        summary["segmentations"][level] = {"key": key, "unique": rows, "robot_pixels": robot_px, "total": int(seg.size)}
        print(f"  saved {stem}_seg_{level}.png / .npy")

    with open(out / f"{robot}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out / f"{robot}_obs_keys.txt", "w") as f:
        for k in sorted(obs):
            f.write(f"{k:36s} {describe(obs[k])}\n")
    env.close()
    print(f"  saved {robot}_summary.json, {robot}_obs_keys.txt")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robots", nargs="+", default=["Panda"], help="예: --robots Panda xArm7")
    ap.add_argument("--seg", nargs="+", default=SEG_LEVELS_DEFAULT, help="instance class element robot_only 중 선택")
    ap.add_argument("--gl", choices=["auto", "egl", "osmesa"], default="auto")
    ap.add_argument("--gpu-id", type=int, default=-1, help="render_gpu_device_id (EGL 전용, -1=자동)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0, help="np.random seed (큐브 배치 재현)")
    args = ap.parse_args()

    gl = choose_backend(args.gl)
    os.environ["MUJOCO_GL"] = gl
    if gl == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    print(f"[gl] using MUJOCO_GL={gl}")

    results = {r: dump_robot(r, args.seg, args.out, gl, args.gpu_id, args.seed) for r in args.robots}
    print(f"\n{'=' * 70}\nout: {args.out}")
    for r, ok in results.items():
        print(f"  {r:10s} {'OK' if ok else 'SKIPPED (미등록)'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
