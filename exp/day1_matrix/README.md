# day1_matrix — Mirage state-based transfer, xArm7 확장 실험 매트릭스 (p2)

작성 2026-09-16. 환경: 잡 125798(A100), `source ~/hayeon/scripts/env_mirage.sh`(venv + OSMesa; lowdim 은 렌더링 없음).
실행 러너: `mirage/mirage/benchmark/robosuite/run_robosuite_benchmark.py --config <yaml> -y` (**반드시 그 디렉터리에서** — 서버/클라이언트를 상대경로로 Popen 한다). `run_all.sh` 가 이를 처리한다.

## 파일

| 파일 | 역할 |
|---|---|
| `gen_configs.py` | 템플릿 + 셀별 override → `configs/*.yaml`, `cells.csv` (34셀). `--target-robot`, `--n-rollouts`, `--out` |
| `run_all.sh -n N` | N 셀 병렬 실행. 완료 셀 skip(재개), 로그 `~/hayeon/logs/day2/<cell>.log`, `--status`, `--dry`, `--filter` |
| `collect.py` | `results/<cell>/target.txt` → `results/summary.md` (Table I / VIII / VI + 논문 수치 열) |
| `_smoke/` | 최초 실측 (Lift Panda→UR5e 2 rollouts, `run2.log`) |
| `_dryrun/` | `gen_configs.py --target-robot UR5e --n-rollouts 2 --out _dryrun` 로 만든 검증용 매트릭스 + 결과 4셀 |

## 1. 인벤토리 — low_dim 체크포인트 (전부 존재, 각 ~8 MB, LFS 아님)

| task | env (ckpt) | ckpt | horizon(학습 rollout) |
|---|---|---|---|
| lift | Lift | `models/lift/low_dim/lift_ph_low_dim_epoch_1000_succ_100.pth` | 400 |
| can | PickPlaceCan | `models/can/low_dim/model_epoch_250_PickPlaceCan_success_1.0.pth` | 400 |
| stack | Stack_D0 (mimicgen) | `models/stack/low_dim/model_epoch_2000.pth` | 400 |
| square | Square_D0 (mimicgen) | `models/square/low_dim/model_epoch_950_Square_D0_success_0.96.pth` | 400 |
| two_piece | ThreePieceAssembly_D0 (mimicgen) | `models/two_piece/low_dim/model_epoch_1000_ThreePieceAssembly_D0_success_0.68.pth` | **500** |

모두 Panda, OSC_POSE(delta, kp 150), obs `robot0_eef_pos/quat, robot0_gripper_qpos, object`. 제외할 task 없음.
mimicgen env 는 `robomimic/envs/env_robosuite.py:14` 의 `import mimicgen_envs` 로 등록됨(확인).

## 2. config 필드 표 (`example_config.yaml` 전 필드 + 누락 필드 + 패치로 추가한 필드)

`ExperimentRobotsuiteConfig.from_yaml` (`robosuite_experiment_config.py`) 가 읽고, `RobosuiteExperiment.launch` (`robosuite_experiment.py`) 가 서버(`evaluate_policy_demo_source_robot_server.py`) / 클라이언트(`evaluate_policy_demo_target_robot_client.py`) CLI 인자로 바꾼다.

| 필드 | 필수 | → CLI | 역할 / 값 |
|---|---|---|---|
| `source_agent_path` | ✓ | server `--agent` | 소스(Panda 오라클) 체크포인트. **env 는 체크포인트의 env_metadata 로 결정** |
| `target_agent_path` | ✓ | client `--agent` | 타깃 쪽 env 생성용 (같은 체크포인트) |
| `n_rollouts` | ✓ | `--n_rollouts` | 롤아웃 수. 50 |
| `horizon` | ✓ | `--horizon` | 롤아웃 최대 스텝. task 별 ckpt 값(400/500) |
| `seed` | ✓ | `--seeds` | 하나만. 롤아웃 i 는 `seed*n_rollouts+i` 로 시드됨(server `run_experiments`) → **seed=0, n=50 ⇒ 0..49** |
| `passive` | ✓ | server `--passive` | 소스가 타깃의 "Request for Action" 을 기다렸다 행동. True 유지 |
| `connection` | ✓ | `--connection --port <random>` | localhost 소켓. True 유지. 포트 10000–65535 랜덤(병렬 실행 충돌 확률 낮음) |
| `naive` | ✓ (**example 에 없음**) | `--naive` | highdim naive 모드. lowdim 은 `false`. 빠지면 KeyError |
| `source_robot_name` | ✓ | server `--robot_name` | `Panda` |
| `target_robot_name` | ✓ | client `--robot_name` | robosuite `REGISTERED_ROBOTS` 이름. `XArm7` 은 §7 에서 등록함 |
| `source_tracking_error_threshold` / `source_num_iter_max` | ✓ | server | 소스에선 로깅 전용(help 문구) |
| `target_tracking_error_threshold` | ✓ | client `--tracking_error_threshold` | blocking 추종 허용 오차(pos+quat norm). 0.015 |
| `target_num_iter_max` | ✓ | client `--num_iter_max` | blocking 내부 반복 상한. **1 = 비블로킹**, 300 = blocking. delta 모드에선 미사용 |
| `delta_action` | ✓ | client `--delta_action` | true: 타깃이 소스와 같은 delta 액션 실행(비블로킹). false: 소스의 절대 EE pose 추종 |
| `enable_inpainting`, `use_ros`, `offline_eval`, `use_diffusion`, `diffusion_input_type` | ✓ | `--inpaint_enabled` 등 | lowdim 은 전부 false / "" |
| `results_folder` | ✓ | `--save_stats_path <folder>/{source,target}.txt` | 존재하면 러너가 override 질문 → `-y` 로 덮어씀 |
| `source_video_path` / `target_video_path` | 선택 | `--video_path` | 비디오(오프스크린 렌더 필요). 원본은 gripper_type 을 넘기던 버그 → 패치 |
| `source_gripper_type` / `target_gripper_type` | 선택 | `--gripper` | robosuite `ALL_GRIPPERS` 이름(PandaGripper, Robotiq85Gripper, Robotiq140Gripper, …). 생략 = 로봇 기본 그리퍼 |
| `device` | 선택(기본 cuda) | `--device` | 정책 디바이스 |
| `target_kp_scale` | 선택 (**패치로 추가**) | client `--kp_scale` | blocking 컨트롤러 kp(150) 배율 — Table VI gain |
| `target_proprio_z_offset` | 선택 (**패치로 추가**) | client `--proprio_z_offset` | 타깃 proprio EE z 바이어스 [m] — Table VI offset |

### config 에 없던 것을 어디서 바꾸는지 (코드 근거)

- **컨트롤러 gain**: server `Robot.step()` blocking 분기의 `kp = np.array([150]*6)` 하드코딩(클라이언트가 상속해 blocking 추종에 사용) → `* self.blocking_kp_scale` 로 패치, client `--kp_scale` 이 설정. dry-run 로그에서 `controller kp: [75. …]` 확인.
- **proprio z-offset**: 어디에도 없음. 논문 "z-offset in the proprioceptive values" = 실로봇 proprio 가 +b 만큼 틀리면 pose 서보가 물리적으로 goal−b 에 머물면서 goal 에 도달했다고 보고. sim 의 OSC 는 참 상태로 서보하므로 `Robot` 에 (**명령 z −= b, 측정 z += b**) 로 구현: `compute_eef_pose`(측정·소스로 보내는 `robot_pose`), `drive_robot_to_target_pose`/`step` blocking 분기(명령). 측정만 편향시키면 추종이 영원히 수렴 못 해 매 스텝 300회를 다 돌게 된다(dry-run 에서 확인, 수정함).
- **그리퍼**: `target_gripper_type` → `--gripper` → `FileUtils.env_from_checkpoint(..., gripper_types=)`. `GripperInterpolator` 는 import 만 되고 미사용 — 그리퍼 액션은 소스 값을 그대로 전달.

### 패치 목록 (`git -C ~/hayeon/mirage diff`, `git -C ~/hayeon/mirage/robomimic diff`)

1. `robomimic/robomimic/utils/file_utils.py:186` `map_location='cuda:4'` → `'cpu'` (포크 작성자 머신 하드코딩; GPU 1개 노드에서 로드 실패). upstream 동작.
2. server: `Robot.__init__` 에 `blocking_kp_scale=1.0`, `proprio_z_offset=0.0`; blocking kp 에 배율; `compute_eef_pose` 측정 편향; `drive_robot_to_target_pose`/`step` 명령 z 보정.
3. client: `--kp_scale`, `--proprio_z_offset` 인자 → 생성자 → 속성 설정; `wait_for_skip` 스레드의 `input()` EOFError 처리(stdin 없이 돌릴 때 트레이스백 방지).
4. `robosuite_experiment_config.py`: Optional `target_kp_scale`, `target_proprio_z_offset`. `robosuite_experiment.py`: 값이 있을 때만 client 인자 추가(기존 config 호환), video_path 버그 수정.

## 3. 매트릭스 — 34 셀 × 50 rollouts = 1,700 rollouts

| 표 | 셀 | 수 |
|---|---|---|
| Table I | 5 task × {pandagrip, default} × `block_iter300` | 10 |
| Table VIII | 5 task × 2 gripper × {`delta_iter1`, `block_iter1`} | 20 |
| Table VI | lift/default/`block_iter300` × {`_zoff2cm`, `_zoff5cm`, `_gain0.5`, `_gain2`} | 4 |

- Table VIII 의 `block_iter300` 은 Table I 셀 재사용. `delta_iter300` 은 생성하지 않음 — delta 모드는 `num_iter_max` 를 안 쓰므로 `delta_iter1` 과 동일.
- 그리퍼 `default` = 타깃 로봇 기본 그리퍼(필드 생략). XArm7 의 기본 그리퍼는 `XArmGripper`(UFACTORY xArm Gripper).
- Table VI 는 논문이 실로봇 Tiger 과제라 Lift 로 경향만 비교.
- 우선순위(`cells.csv` 순서): lift(8) → can(6) → stack(6) → square(6) → two_piece(6).
- seed: 전 셀 `seed: 0` → 롤아웃별 0..49 고정. 스크립트 레벨 루프 불필요.

## 4. 소요시간 (dry-run 실측, Lift Panda→UR5e, 2 rollouts, 셀 2개 동시)

| 모드 | 스텝 | 벽시계 | 스텝당 |
|---|---|---|---|
| `block_iter300` | 100 (2 롤아웃 모두 50스텝에 성공) | 1분 55초 | **≈1.1 s** (시작 오버헤드 ≈15 s 포함) |
| `delta_iter1` | 440 (성공 40 + 실패 400) | 4분 01초 | **≈0.55 s** |
| `block_iter300_gain0.5` | 153 | 4분 14초 | **≈1.6 s** (gain 낮아 blocking 반복 증가) |
| `block_iter300_zoff2cm` | 155 | 6분 52초 | **≈2.6 s** (일부 스텝이 300회 상한까지 감: 155스텝 중 4회 미도달) |

- 롤아웃 시간 ≈ 성공 시 50–60 s(blocking) / 25 s(delta), 실패 시 horizon 스텝 전부(400→ blocking ≈7 min, delta ≈4 min; two_piece 500).
- 셀(50회) ≈ 50·[p·55 + (1−p)·440] s (blocking). Lift(p≈1) ≈ 50 min, Can(p≈0.9) ≈ 1.2 h, Square(p≈0.5) ≈ 3.4 h, two_piece(p≈0.7, h=500) ≈ 2.6 h. delta 셀은 약 절반.
- 전체 직렬 ≈ 30–40 h. 셀당 프로세스 2개가 각 70–80% CPU(6코어 노드, 2셀 동시에 load ≈4.5) → **`-n 3`** 권장, ≈ 12–15 h. walltime 72 h 안.

## 5. 실행 순서

```bash
source ~/hayeon/scripts/env_mirage.sh
cd ~/hayeon/mirage/exp/day1_matrix
python gen_configs.py                                   # XArm7, 50 rollouts
python gen_configs.py --target-robot UR5e --n-rollouts 2 --out _dryrun   # 파이프라인 검증 (완료: 4셀)
./run_all.sh -n 2 -f _dryrun/cells.csv --filter '^lift_default_(block_iter300|delta_iter1|block_iter300_gain0.5|block_iter300_zoff2cm)$'
python collect.py --out _dryrun
nohup ./run_all.sh -n 3 > ~/hayeon/logs/day2/run_all.log 2>&1 &   # 본실험 (셸 닫아도 대기열 소비, 재개 가능)
./run_all.sh --status
python collect.py
```

## 6. 주의

- **컨테이너(NGC 이미지)엔 tmux 가 없다.** `run_all.sh` 는 tmux 가 없으면 자동으로 `bg` 백엔드(nohup+setsid, PID 파일 `~/hayeon/logs/day2/<cell>.pid`)를 쓴다. 로그인 노드처럼 tmux 가 있으면 세션 `day2` 에 창을 만든다.
- **xArm7 등록 완료(§7)**. `_dryrun_xarm7/` 로 파이프라인 검증.
- 러너는 결과 폴더가 있으면 override 를 묻고 `-y` 로 덮어쓴다 → 중단된 셀은 처음부터 다시(롤아웃 단위 재개 없음). `run_all.sh` 는 `target.txt` 가 있는 셀만 skip.
- 셀 로그는 `PYTHONUNBUFFERED=1` 로 실시간. 러너를 직접 돌릴 땐 stdout 이 파일이면 종료 전까지 아무것도 안 보인다(블록 버퍼링).
- 결과 파일은 JSON 두 객체 연속(`avg_rollout_stats` + `summary_stats`). `collect.py` 가 처리. 롤아웃마다 덮어써지므로 실행 중 셀도 부분 결과가 읽힌다(`(n=…)` 표기).
- `/home` 풀 용량 변동 큼(어제 186 MB 까지 감). 결과 파일은 작지만(셀당 수 KB) 실행 전 `df -h ~` 확인.

## 7. xArm7 등록 기록 (mirage-robosuite `xembody` 위에 추가)

원래 어느 브랜치에도 xArm7 이 없어 직접 등록했다. 소스: MuJoCo Menagerie `ufactory_xarm7` (BSD-3, UFACTORY; 메시 4.5 MB) 의
`xarm7_nohand.xml` + `hand.xml` 을 robosuite 규약(명시적 속성, `base`/`right_hand` 바디, `torq_j*` 모터, `eef`+`grip_site`, 이름 있는 collision geom, F/T 센서)으로 재작성.

| 파일 | 내용 |
|---|---|
| `robosuite/models/assets/robots/xarm7/robot.xml` + `meshes/` | 7-DoF 팔. 관절 한계·관성은 Menagerie 값, 토크 한계 50/50/30/30/20/20/20 Nm, 감쇠는 Panda 와 동일(0.1…0.01). `right_hand` 는 link7 원점, quat `0 0 0 1` |
| `robosuite/models/robots/manipulators/xarm7_robot.py` | `XArm7(ManipulatorModel)`: default_gripper `XArmGripper`, mount `RethinkMount`, controller `default_xarm7`, base offset은 Panda 와 동일, `XARM7_INIT_QPOS` |
| `robosuite/models/assets/grippers/xarm_gripper.xml` + `meshes/xarm_gripper/` | 1-액추에이터(텐던 `split`, ctrl 0=열림/255=닫힘) 6-조인트 링크. **`eef` 바디를 z −90° 회전**해 손가락 개폐축이 `grip_site` x 축이 되게 함 — Panda/Robotiq85 와 동일(실측). grip_site 는 패드 중심(z 0.145) |
| `robosuite/models/grippers/xarm_gripper.py` | `XArmGripper`: `format_action` 은 Robotiq85 와 같은 ±1 램프(speed 0.01), `_important_geoms` 에 finger/fingerpad collision |
| `robosuite/controllers/config/default_xarm7.json` | `default_panda.json` 복사 |
| 등록 | `manipulators/__init__.py`, `robots/__init__.py:ROBOT_CLASS_MAPPING`, `grippers/__init__.py:GRIPPER_MAPPING`, `environments/robot_env.py` robot_only 목록, `utils/input_utils.py`, mirage `GripperInterpolator` 매핑 |
| `xarm7/tune_init_qpos.py` | Panda 의 Lift 초기 grip_site 자세(베이스 기준 (0.466, 0.004, 0.100), x→+Y y→+X z→−Z)에 7관절을 최소자승으로 맞춤(+j1/j3/j5→0 정규화). 결과 `[0.0044, 0.18, 0.0041, 0.9088, -0.0009, 0.7288, 3.1508]`, 오차 0 mm |
| `xarm7/smoke_xarm7.py` | 컴파일·reset·개폐(0.093→0.008 m)·드리프트 0·delta 응답(−123 mm/10스텝, Panda −120 mm) 확인 |

주의: xArm7 도달거리 700 mm (Panda 855, UR5e 850). Lift 는 문제없음; Can(bins arena) 등 먼 목표는 dry-run 에서 도달 여부 확인 필요.

## 8. 본실험 실행 기록 (2026-09-28, 잡 127774 / ece-agpu9 A100)

- `configs/` 재생성: 9/16 파일은 `target_robot_name: xArm7` 이었는데 등록 이름은 `XArm7` (`ROBOT_CLASS_MAPPING` 키, 대소문자 정확) → `gen_configs.py` 기본값 수정 후 재생성.
- 실행 순서 변경: **Table I + VI 전부 → 그 뒤 Table VIII** (`build_cells()`; VIII 는 부차적이라 뒤로).
- `run_all.sh` 완료 판정: `target.txt` 존재 → `"Num Rollouts" ≥ n_rollouts`. 러너가 롤아웃마다 `target.txt` 를 덮어쓰므로 중간에 죽은 셀(잡 walltime 등)이 파일만 보고 영영 skip 되던 문제 수정. `--status` 가 `RUNNING(n=…)` / `partial(n=…/50 → 재실행 대상)` 표시.
- **z-offset 2셀(`_zoff2cm`, `_zoff5cm`) 이번 실행에서 제외** → 32셀 × 50 = 1,600 rollouts. `gen_configs.py --with-zoff` 로 복원.
  이유: xArm7 에서 명령 z 가 2 cm 낮아지면 그리퍼가 테이블에 막혀 추종 오차 0.025~0.04 (> 0.015) → 해당 스텝마다 blocking 300회 상한 (`Take 300 iterations` / `did not reach target pose`) → 실측 **45 s/step**, 롤아웃 1회 2~5 h, 50회 100 h+. 9/16 dry-run 도 113 스텝까지 성공 0. 구현(오프셋 부호·물리적 의미)과 롤아웃 수를 다시 정한 뒤 별도 실행. 중단한 로그: `~/hayeon/logs/day2/lift_default_block_iter300_zoff2cm.ABORTED-0928.log`.
- 실측 속도 (A100 노드, `-n 3`, 6코어): `lift_pandagrip_block_iter300` 0.82 s/step (거의 1~2 iterations/step) → 셀 ≈ 40 min; `lift_default_block_iter300` 4.1 s/step (일부 스텝 300회 상한) → 셀 ≈ 2.5 h.
- 9/16 xArm7 dry-run 셀 로그는 `~/hayeon/logs/day2/_dryrun_xarm7/` 로 이동.

### 8-1. 느린 셀 분할 (9/28 22:30, 실행 중 전환)

- 실측: `can_*_block_iter300` 은 스텝의 48% 가 300회 상한(추종 오차 대부분 >10 cm — xArm7 도달거리 700 mm 한계로 보임) → **롤아웃당 ~55 min, 셀당 45~55 h**. 잡 walltime(72 h, 10/1 15:10 종료) 안에 못 끝날 위험 + 중단 시 셀 통째로 재실행.
  `stack_default` 도 14% 가 상한(~15 min/rollout). `lift_default` 7%(오차 0.015~0.03, 소폭), `stack_pandagrip` 3.7%.
- 대책: `gen_configs.py` 의 `CHUNK_CELLS`(can×2, stack_default, square×2, two_piece×2) 를 **`<cell>__s<k>` 조각 10개 (seed=k, n_rollouts=5)** 로 분할. 서버 `run_experiments` 가 롤아웃 i 를 `set_seed(seed*n_rollouts+i)` 로 시드하므로 조각 k = 시드 5k..5k+4 → 원래 (seed 0, n 50) 과 **동일한 시드 0..49**. cells.csv 에 `base`, `seed` 열 추가(끝에 붙여 run_all.sh 파싱 영향 없음). 95 행 = 32 셀. `--no-chunk` 로 끔.
- `collect.py` 는 `base` 별로 `Num_Success`/`Num Rollouts` 를 합산해 성공률 재계산 (셀 상태 표에 조각 수).
- `run_all.sh --exclude REGEX` 추가. 병렬 `-n 5` (셀당 client ~1.1 코어, server ~0.05 코어; cpuset 6코어, RSS 셀당 ~2.5 GB).
- 전환 절차: 실행 중이던 unchunked 셀은 **n=5 에서 끊어 결과 폴더를 `<cell>__s0` 로 이름만 바꾸면 조각 0 과 같다** (target.txt 는 누적 통계). `can_pandagrip` 은 n=5 시점에 즉시 중단, `can_default`/`stack_default` 는 `~/hayeon/scripts/chunk_watch.sh` 가 n=5 가 되는 순간 중단·이동(`~/hayeon/logs/day2/chunk_watch.log`). 그동안 드라이버는 `--exclude '^(can_default_block_iter300|stack_default_block_iter300)__s0$'` 로 그 두 조각을 건너뜀.
  ```bash
  nohup ./run_all.sh -n 5 --exclude '^(can_default_block_iter300|stack_default_block_iter300)__s0$' >> ~/hayeon/logs/day2/run_all.log 2>&1 &
  ```
- 남은 작업량(대략): can 90 rollouts × 55 min ≈ 80 slot-h, stack_default ≈ 12, square/two_piece 미지(각 5~10?), Table VIII 20셀(비블로킹, 빠름) ≈ 15 → 5 슬롯 ≈ 28~35 h → **9/30 오전~오후 완료 예상**, walltime 마진 ≈ 1일.
- 300회 상한은 **롤아웃(초기 배치)마다 갈린다** — 드리프트 아님. Can: 대부분 롤아웃은 cap 0~10 회·~100 스텝에 성공, 일부(예: 시드 1, 4, 5; default 시드 1, 2, 4)는 cap 60~290 회로 실패 시 horizon 400 까지 3~4 h. stack_pandagrip 도 r49 한 번이 cap 222 (≈2.8 h). 조각 s3(시드 15~19)는 5/5 성공·12 min. 따라서 셀 시간은 "나쁜 롤아웃" 수에 좌우되고 분할이 그 직렬화를 풀어 준다.
- 무인 마무리: `~/hayeon/scripts/driver_handoff.sh` 가 `chunk_watch done` 을 기다렸다가 드라이버를 `--exclude` 없이 재시작 (`chunk_watch.log` 에 `handoff done`). 그 뒤로는 `run_all.sh -n 5` 가 모든 행(부분 s0 포함)을 처리한다.

### 8-2. Square D · Two Piece 는 num_iter_max 100 (9/29 20:50, 사용자 결정)

- 9/29 20:42 상태: 37/95 행 완료, Table I Lift 100/100 · Stack 96/90 · Can 80/72 확정, Square F 45 % (n=20). Square 조각 평균 **318 min**(Can F 156, Can D 227, Stack D 42). 남은 Square D 10 + Two Piece 20 + VIII 20 을 300 으로 돌리면 ≈41 h vs walltime 잔여 42 h → 초과 위험.
- 근거: 상한에 걸린 스텝의 추종 오차는 Square 에서 >0.10 이 2562 / 0.05~0.10 이 874 / <0.05 는 135 (도달 불가). **51~299회에서 뒤늦게 수렴한 스텝: Can 28/17701, Stack 126/13092, Square 8/6835.** 즉 상한을 100 으로 낮춰도 추종 결과는 ≤1 % 의 스텝에서만 달라진다. 원저자 코드의 함수 기본값도 `num_iter_max=100` (300 은 example_config 값).
- 적용: `gen_configs.py` `ITER_OVERRIDE` = `square_default`, `two_piece_pandagrip`, `two_piece_default` (아직 시작 안 한 셀). `source/target_num_iter_max: 100`. Square F 는 300 으로 마무리. 셀 이름·조회 키는 그대로, cells.csv `num_iter_max` 열과 `collect.py` 각주로 구분. `--no-iter-override` 로 끔.
- 절차: 드라이버를 sleep 중일 때 정지 → `python gen_configs.py` (완료·실행 중 행의 config 는 내용 동일) → `run_all.sh -n 5` 재시작. 실행 중이던 Square F 조각은 그대로.

### 8-3. Two Piece 를 큐 맨 뒤로 (9/30 16:30, 사용자 결정)

- 9/30 16:20 상태: 48/95 완료. Square F **26 %**(완료), Square D **0 % (n=24)** — 전부 실패라 상한 100 에서도 조각당 217~322 min. 잔여 ≈ Square D 19 + Two Piece 10~85 + VIII 20~30 slot-h = 10~27 h vs walltime 22.8 h.
- `gen_configs.py` `DEFER_CELLS`(two_piece blocking 조각 20개)를 Table VIII 뒤로 이동 → Square D + VIII 는 안에 끝나고, Two Piece 는 남는 시간에 돌다가 새 잡에서 재개. 순서만 바뀌고 설정은 그대로.
