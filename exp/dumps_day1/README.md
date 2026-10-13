# day1 — Panda agentview RGB + segmentation 한 프레임 덤프 (p1)

생성: 2026-09-15, 잡 125798 (A100-SXM4-80GB, ece-agpu15), robosuite 1.4.1 (mirage-robosuite `xembody` 7ce0456), mujoco 2.3.2
렌더러: OSMesa (CPU 소프트웨어). 컨테이너에 libEGL/libOSMesa 가 없어 conda-forge `mesalib<24` 의 libOSMesa 사용.

## 재현

```bash
source ~/hayeon/scripts/env_mirage.sh          # venv + LD_LIBRARY_PATH(OSMesa) + MUJOCO_GL=osmesa
cd ~/hayeon                                     # 주의: ~/hayeon/mirage(리포 루트)에서 실행하면 robosuite 패키지가 가려짐
python scripts/dump_camera_obs.py --robots Panda --seed 0 --out ~/hayeon/dumps/day1
python scripts/dump_camera_obs.py --robots Panda xArm7   # xArm7 은 미등록 → 등록 목록 출력 후 skip, exit 1
```
`--gl auto` 가 기본: egl 을 먼저 시험하고 실패하면 osmesa. 둘 다 실패하면 진단 출력 후 exit 2.
`--seed 0` 으로 큐브 배치가 재현된다 (placement sampler 가 np.random 사용).

## 산출물

| 파일 | 내용 |
|---|---|
| `Panda_agentview_rgb.png` | 84×84 RGB, `np.flipud` 적용(robosuite IMAGE_CONVENTION="opengl" 은 상하 반전) |
| `Panda_agentview_seg_{instance,class,element,robot_only}.png` | 컬러맵 PNG (id 0/배경 = 검정, tab20 순환) |
| `Panda_agentview_seg_*.npy` | 원본 int32 id (84×84), flipud 적용 |
| `Panda_agentview_x4_composite.png` | rgb / instance / element / robot_only 나란히 ×4 |
| `Panda_summary.json`, `Panda_obs_keys.txt` | obs 키·shape·dtype, seg id→이름·픽셀수 |
| `GL_ENV.txt` | 사용한 mesa prefix / MUJOCO_GL / seed |

## 확정된 obs 키 (Lift, Panda, OSC_POSE, agentview 84×84, use_object_obs=True)

```
agentview_image                    (84,84,3)  uint8
agentview_segmentation_instance    (84,84,1)  int32   camera_segmentations="instance"
agentview_segmentation_class       (84,84,1)  int32   camera_segmentations="class"
agentview_segmentation_element     (84,84,1)  int32   camera_segmentations="element"  (원시 geom id, 배경=-1 가능)
agentview_segmentation_robot_only  (84,84,1)  int32   camera_segmentations="robot_only" (이 포크 전용, 0/1)
robot0_eef_pos(3) robot0_eef_quat(4) robot0_gripper_qpos(2) robot0_gripper_qvel(2)
robot0_joint_pos(7) robot0_joint_pos_cos(7) robot0_joint_pos_sin(7) robot0_joint_vel(7) robot0_proprio-state(39)
cube_pos(3) cube_quat(4) gripper_to_cube_pos(3) object-state(10)
```
seg 키 규칙: `f"{camera_name}_segmentation_{level}"` (robot_env.py `_create_segementation_sensor`). action_dim 7 (OSC_POSE 6 + gripper 1).

## seg id 표 (seed 0 프레임)

instance / class (id = `enumerate(model.instances_to_ids)` 순서; **미매핑 geom → 0**):
```
0  cube / BoxObject        6479 px   ← 큐브 64 px + 바닥 367 + 벽 578 + 테이블 5470 이 전부 0 으로 합쳐짐
1  Panda0 / Panda             4 px   (팔은 이 프레임에 거의 안 보임)
2  RethinkMount0 / RethinkMount 196 px (받침대, 그리퍼 뒤로 보임)
3  PandaGripper0 / PandaGripper 377 px
```
element (geom id → `sim.model.geom_id2name`):
```
0 floor 367 | 5 wall_rear_visual 578 | 8 table_visual 5470 | 85 cube_g0_vis 64
69 robot0_g49_vis 4 | 71 gripper0_hand_visual 350 | 73 gripper0_finger1_visual 11 | 76 gripper0_finger2_visual 16 | 82 mount0_pedestal_vis 196
```
robot_only: `1 = Panda + PandaGripper` 381 px (5.4 %), `0 = 그 외` (받침대·큐브·테이블 포함).

## 눈으로 확인할 체크리스트 (→ 확인 결과)

- [x] rgb 방향: 테이블이 아래, 벽이 위, 그리퍼가 화면 위에서 내려옴 (뒤집혀 있으면 flipud 가 두 번 적용된 것)
- [x] rgb 에 빨간 큐브가 테이블 위에 보임 (seed 0: 중앙 약간 왼쪽)
- [x] `robot_only` 흰/파란 영역이 rgb 의 그리퍼 실루엣과 일치, 손가락 사이 틈이 비어 있음
- [x] `robot_only` 에 받침대(pedestal) 와 큐브가 **포함되지 않음** — Mirage 로봇 마스크 용도로 적합
- [x] `element` 에서 큐브가 테이블과 다른 색 (분리됨)
- [ ] `instance`/`class` 에서 큐브 색이 배경과 **같음(검정)** — 이 두 레벨로는 물체/배경 구분 불가 (robosuite 의 id 0 기본값 충돌). 물체 마스크가 필요하면 `element` 를 쓰거나 mapping 기본값을 -1 로 바꿔야 함
- [ ] 팔(`Panda` 링크)이 agentview 84×84 에서 4 px — 팔 몸체는 이 카메라/자세에선 사실상 안 보임. 팔 마스크가 필요하면 `robot0_eye_in_hand` 외 다른 카메라 또는 다른 자세로 검토

## xArm7 (2026-09-16 등록 후 추가)

`python scripts/dump_camera_obs.py --robots Panda XArm7 --seed 0` → `XArm7_*` 파일 + `Panda_vs_XArm7_composite.png`.
- 같은 시점·같은 큐브 위치(seed 0)에서 xArm7 그리퍼(`XArmGripper`)가 Panda 와 같은 자리에 보임 (init_qpos 를 Panda 초기 grip_site 자세에 맞춰 튜닝했기 때문).
- seg: `robot_only` 280 px (Panda 381 px) — xArm 그리퍼 실루엣, 받침대·큐브 제외. `element` 에서 gripper0_hand/knuckle/finger 전부 분리. 팔 링크는 1 px (Panda 4 px) — 마찬가지로 agentview 에선 팔이 안 보임.
- obs 키는 Panda 와 동일 (`robot0_joint_pos` 7, `robot0_gripper_qpos` 6 — xArm 그리퍼 링크 조인트 6개).

## 발견 / 주의

1. **xArm7 은 원래 mirage-robosuite 어느 브랜치에도 없었음 → 직접 등록함 (exp/day1_matrix/README.md §7)**. 원문: (`master`, `xembody`, `xembody_followup` 모두 grep 0건). 등록 로봇: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e. 추가하려면 `models/robots/manipulators/xarm7_robot.py` + `models/assets/robots/xarm7/`(MJCF·메시) + 그리퍼 + `manipulators/__init__.py`, `robots/__init__.py:ROBOT_CLASS_MAPPING`, `controllers/config/default_xarm7.json` 이 필요.
2. robosuite `renderers/context/egl_context.py:52` 는 `CUDA_VISIBLE_DEVICES` 를 정수로 파싱하는데 이 클러스터는 UUID(`GPU-…`)를 넣어 준다 → EGL 사용 시 ValueError. 우회: `export MUJOCO_EGL_DEVICE_ID=0`. (오늘은 OSMesa 라 해당 없음)
3. `import robosuite` 는 `macros.MUJOCO_GPU_RENDERING=True` 면 import 시점에 EGL 을 요구한다. `robosuite/robosuite/macros_private.py` 에 `MUJOCO_GPU_RENDERING = False` 를 두었다 (.gitignore 에 포함).
4. pip 은 항상 `python -m pip` (venv 에 pip 바이너리 없음). numpy 는 `numpy<2` 제약 필수 (이미지 torch 2.1.0a0 / mujoco 2.3.2 가 numpy 1.x ABI). `open3d` 는 numpy 2 를 끌어와 보류. `opencv-python-headless` 사용(컨테이너에 libGL 없음). `pynput` 은 헤드리스에서 import 실패(정상, 미사용).
5. 실제 GPU 는 A6000 이 아니라 **A100** (a6gpu 노드 5/8 down, 나머지 GPU 전부 점유).
