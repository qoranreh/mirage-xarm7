# Cross-Painting 구현 대조 보고서 (Mirage 논문/공식 구현 vs. xarm_transfer)

작성일: 2026-08-27 (같은 날 E-4 정정, B-5 추가)
대상 구현: `mirage/mirage/benchmark/robosuite/xarm_transfer/` (미커밋 작업 트리 상태)
비교 기준: 이 레포에 함께 들어 있는 **공식 Mirage 구현**
(`mirage/mirage/benchmark/robosuite/evaluate_policy_demo_*.py`,
`mirage/mirage/ros_ws/src/gazebo_env/scripts/write_data_node_robosuite_better.py`,
`mirage/mirage/infra/ros_inpaint_publisher_sim.py`,
`mirage/mirage/gripper_interpolation/robosuite/gripper_interpolator.py`)

논문 본문(arXiv 2402.19249) 대신 **레포에 포함된 저자 구현**을 1차 기준으로 삼았다.
논문 서술은 추상적이라 픽셀 단위 검증이 불가능하지만, 저자 구현은 실행 가능한
사양이므로 모든 항목을 파일/라인으로 검증할 수 있다.

---

## 0. 요약

Cross-painting의 **핵심 알고리즘(마스킹 → Telea 인페인팅 → depth 기반 소스 로봇 합성)은
공식 구현과 실질적으로 동일**하다. 좌표/방향/스케일 규약도 검증 결과 모두 일치한다.

그러나 파이프라인 주변부에 **의도적 대체 5건, 논문에 없는 추가 4건, 미구현 4건,
수치 파라미터 불일치 3건**이 존재한다. 이 중 실패에 직접 기여할 가능성이 높은 것은
아래 3개다.

| 순위 | 항목 | 성격 |
|---|---|---|
| 1 | 타깃 추종 허용오차가 공식 대비 크게 느슨함 (25 mm / 7°, 접촉 시 60 mm / 18°) | 수치 불일치 |
| 2 | 그리퍼 상태 전달이 공식의 회귀 기반 `GripperInterpolator`가 아니라 자체 정규화 매핑 | 방식 대체 |
| 3 | 소스 로봇 레이어를 "빈 월드 로봇 전용 렌더"가 아니라 "소스 씬 전체 렌더 + 마스크 크롭"으로 생성 | 구조 대체 |

**검증되어 문제 없음이 확인된 항목**(과거에 의심할 만한 것들): 상하 플립 규약, RGB 채널
순서, `[0,1]` 정규화, CHW 레이아웃, metric depth 변환식, 카메라 내부/외부 파라미터,
마운트(RethinkMount) 처리, `policy.start_episode()` 호출.

---

## 1. 공식 Mirage cross-painting 파이프라인

한 policy step 기준 (online eval, `enable_inpainting: true`):

1. **타깃 로봇 env**(예: UR5e)가 자기 관측을 만든다.
   `agentview_image`(84×84 RGB), `agentview_depth`→`get_real_depth_map`,
   `agentview_segmentation_robot_only`.
   → `evaluate_policy_demo_target_robot_client.py:172-186`

2. 타깃이 **자신의 RGB / metric depth / 세그멘테이션 마스크 / EEF 4×4 pose /
   보간된 그리퍼 각도**를 ROS 노드로 발행한다.
   → `evaluate_policy_demo_target_robot_client.py:206-229`,
   `mirage/infra/ros_inpaint_publisher_sim.py:47-63`

3. **Gazebo가 소스 로봇(Panda)만** 빈 월드에서, 타깃이 보낸 EEF pose에 IK로 맞춰
   렌더한다. robot-only RGB / robot-only depth / robot 마스크를 얻는다.
   → `ros_ws/src/gazebo_env/scripts/write_data_node_robosuite_better.py:494-525`

4. **해석적 합성기**가 최종 이미지를 만든다
   (`write_data_node_robosuite_better.py:336-395`):

   ```python
   # (a) 타깃 로봇 픽셀을 RGB와 depth 양쪽에서 0으로 지운다
   robosuite_rgbd_masked = bitwise_and(rgbd, rgbd, mask=~target_robot_mask)

   # (b) depth 우선순위 결정: 0(소스 없음)=1000, 0(지워진 배경)=5
   joined_depth[0] = gazebo_source_depth ; joined_depth[0][==0] = 1000
   joined_depth[1] = masked_target_depth ; joined_depth[1][==0] = 5
   argmin = np.argmin(joined_depth, axis=0)          # 0=소스로봇, 1=배경

   # (c) 배경 복원: Telea, radius 3
   background = cv2.inpaint(masked_rgb, target_robot_mask,
                            inpaintRadius=3, flags=cv2.INPAINT_TELEA)

   # (d) 합성
   inpainted = background * argmin + gazebo_source_rgb * (1 - argmin)
   ```

5. 합성 이미지는 **소스 정책 프로세스**로 전달되어 `obs["agentview_image"]`만 교체한 뒤
   추론된다. **나머지 proprio 관측은 소스 Panda env 자신의 것**을 쓴다.
   → `evaluate_policy_demo_source_robot_server.py:682-693`

6. 소스 Panda env는 매 step **타깃의 object state를 복사받고, 타깃의 EEF pose로
   absolute-OSC 정렬**된 뒤 정책 액션을 자기 env에서 실행한다. 그 결과 achieved
   EEF pose가 타깃의 다음 목표가 된다.
   → `evaluate_policy_demo_source_robot_server.py:626-628`, `206-244`

7. 타깃은 그 pose를 absolute-OSC blocking control로 추종한다 (`kp=[150]*6`).
   → `evaluate_policy_demo_source_robot_server.py:300-340`

---

## 2. 현재 구현 파이프라인

`rollout.py:371-508` → `image_runtime.py:246-330` → `cross_painting.py:130-215`

1. 타깃 xArm7 env에서 `core.sim.render()`로 84×84 RGB / normalized depth /
   element 세그멘테이션을 직접 렌더한다. (`runtime.py:416-487`)
2. geom-ID로 **robot / gripper / manipulated-object 3종 마스크**를 분리한다.
3. 소스 Panda 워커(spawn 프로세스)에 타깃 프레임 + 타깃 pose + object qpos/qvel +
   그리퍼 aperture를 보낸다.
4. 워커는 object state를 소스로 복사 → 그리퍼 개방률 동기화 →
   `align_to_pose()`(absolute OSC, 3 mm / 2°, 최대 100회) → object state 재적용 →
   관측 갱신 → **소스 씬 전체를 렌더**하고 robot 마스크로 크롭.
5. 카메라가 다르면 RGB-D 재투영(z-buffer 4-이웃 splat)으로 타깃 카메라에 정렬.
   (`image_runtime.py:72-159`)
6. `CrossPainter.compose()`가 Telea 인페인팅 + depth 합성 수행.
7. 합성 이미지를 `image_override`로 넣어 정책 추론 → 오라클 forward dynamics →
   `AbsoluteBlockingController`로 xArm 추종.

---

## 3. 단계별 대조표

| # | 단계 | 공식 구현 | 현재 구현 | 판정 |
|---|---|---|---|---|
| 1 | 씬 이미지 출처 | 타깃 로봇 env | 타깃 xArm7 env | 일치 |
| 2 | 이미지 해상도 | 84×84 | 84×84 | 일치 |
| 3 | 상하 플립 | robosuite `opengl`(항등) → robomimic `[::-1]` | `sim.render()[::-1]` | **일치 (검증)** |
| 4 | 채널 순서 | RGB, `process_frame`은 HWC→CHW만 수행 | RGB, `transpose(2,0,1)` | **일치 (검증)** |
| 5 | 정규화 | `/255.0`, float32, `[0,1]` | `/255.0`, float32, `[0,1]` | 일치 |
| 6 | metric depth | `camera_utils.get_real_depth_map` | 동일 수식 자체 구현 (`runtime.py:387-399`) | **일치 (검증)** |
| 7 | 타깃 로봇 마스크 | robosuite `robot_only` (class 단위, robots+grippers) | geom-ID 기반 (robot_model+gripper bodies) | 대체 (§4.A-1) |
| 8 | 마운트/받침대 | 마스크에서 제외됨 | 마스크에서 제외됨 (양쪽 RethinkMount 동일) | 일치 |
| 9 | 마스크 팽창 | 없음 (sim 경로) | 그리퍼 마스크만 1 px | **추가 (§4.B-1)** |
| 10 | 물체 픽셀 보호 | 없음 | inpaint 마스크에서 제외 + 재합성 | **추가 (§4.B-2)** |
| 11 | 인페인팅 | Telea, radius 3 | Telea, radius 3 | 일치 |
| 12 | depth 합성 규칙 | argmin(소스, 배경) with 1000/5 상수 | `source_mask & (inpaint_mask \| src+ε<tgt)` | 동등 |
| 13 | 소스 로봇 렌더 | Gazebo, 빈 월드, robot-only | Robosuite 소스 씬 전체 렌더 후 마스크 크롭 | **대체 (§4.A-2)** |
| 14 | 소스 로봇 자세 결정 | Gazebo IK (EEF pose 직접) | Robosuite absolute-OSC 정렬 | 대체 (§4.A-3) |
| 15 | 조명/재질 정합 | Gazebo≠MuJoCo (LAB 휘도 보정 코드 주석 처리됨) | 동일 MuJoCo 씬 → 자동 일치 | **개선** |
| 16 | 카메라 정합 | Gazebo 카메라를 launch 파일로 수동 정합 | 동일 robosuite 카메라 → 항등 | **개선** |
| 17 | proprio 관측 출처 | 소스 Panda env 자신 | 소스 Panda env 자신 | 일치 |
| 18 | object state 동기화 | 타깃→소스, `qpos`만 | 타깃→소스, `qpos`+`qvel`, 정렬 후 재적용 | 확장 |
| 19 | 소스 정렬 허용오차 | 7D norm 0.015 (config), 300회 | 위치 3 mm **and** 자세 2°, 100회 | 불일치 (§4.D-1) |
| 20 | 정렬 실패 처리 | 경고 출력 후 진행 | 예외 → 롤아웃 중단 | 차이 (§4.E-3) |
| 21 | 그리퍼 소스↔타깃 매핑 | `GripperInterpolator` 선형 회귀 | 관절 range 정규화 개방률 + 핑거패드 실측 | **대체 (§4.A-4)** |
| 22 | 타깃 그리퍼 명령 | 정책 원시 액션을 blocking 전 구간에 인가 | 수렴 후 0.25 정규화 증분 1회만 | **대체 (§4.A-5)** |
| 23 | 타깃 추종 제어 | absolute OSC, `kp=[150]*6` | absolute OSC, `[150]*6` / 접촉 시 `[300,300,500,300,300,300]` | 추가 (§4.B-3) |
| 24 | 타깃 추종 허용오차 | 7D norm 0.015, 300회 | 25 mm / 7° + 4 settling / 접촉 시 60 mm / 18° | **불일치 (§4.D-2)** |
| 25 | forward dynamics | 시뮬: 소스 env 실행(oracle) / 실기: 학습 모델 | image 경로는 oracle만 | 미구현 (§4.C-3) |
| 26 | naive 베이스라인 | 있음 | 있음 (`--visual-input naive`) | 일치 |
| 27 | diffusion(ControlNet) 변형 | 있음 | 없음 | 미구현 (§4.C-1) |

---

## 4. 차이점 상세

### A. 구조적 대체 (의도적, 동등하거나 개선)

**A-1. 세그멘테이션 마스크 산출 방식**
공식은 robosuite의 `robot_only` 세그멘테이션을 쓴다
(`robosuite/environments/robot_env.py:478-497`). 그런데 이 경로는
`ALL_ROBOTS = {Sawyer, Panda, Jaco, Kinova3, IIWA, UR5e}` 하드코딩 집합에 클래스명이
있어야만 1을 부여한다. **xArm7은 이 집합에 없으므로 공식 경로를 그대로 쓰면 타깃
마스크가 전부 0이 되어 cross-painting이 무력화된다.** 현재 구현이 geom-ID 기반
자체 마스크로 바꾼 것은 필수 대체였고, 결과 범위(로봇 링크 + 그리퍼, 마운트 제외)는
Panda 기준으로 공식과 동일하다.

**A-2. 소스 로봇 레이어 렌더 방식**
공식: Gazebo 빈 월드에서 로봇만 렌더 → 소스 레이어는 물체/테이블에 가려지지 않는다.
현재: 소스 Panda 씬 전체를 렌더한 뒤 robot 마스크로 크롭 → **소스 씬의 물체나 테이블이
Panda를 가리면 그 픽셀이 소스 마스크에서 빠진다.**
object state를 동기화하므로 대부분 타깃 씬에서도 같은 물체가 같은 위치에 있어 결과가
같지만, 정렬 잔차(최대 3 mm)만큼 어긋난 경계 픽셀에서는 소스 로봇 실루엣이 미세하게
잘려나갈 수 있다. 위험도 낮음, 다만 **논문 구현과 동치임이 증명된 것은 아니다.**

**A-3. 소스 로봇 자세 결정**
공식은 Gazebo IK로 EEF pose만 맞춘다(물리 없음, 즉시 수렴).
현재는 absolute-OSC로 물리를 전진시키며 수렴시킨다. 이 때문에
(a) 수렴 잔차가 남고(위치 ≤3 mm, 자세 ≤2°),
(b) 정렬 자체가 소스 env 물리를 최대 100 step 전진시켜 물체를 건드리므로
    `set_manipulated_object_joint_states`를 정렬 후 다시 적용해야 했다
    (`image_runtime.py:276-279`).
공식 구현에도 동일한 `drive_robot_to_target_pose` 경로가 있으므로(§1-6) 전면적 이탈은
아니지만, **공식은 렌더용 로봇과 물리용 로봇을 분리했고 현재는 하나로 합쳤다**는 점이
근본 차이다.

**A-4. 그리퍼 상태 전달**
공식: `GripperInterpolator`가 소스/타깃 쌍별로 **미리 적합된 선형 회귀**로 관절값을 변환한다
(`gripper_interpolator.py:44-66`). 예: Panda→UR5e는 `-0.05702*x + 0.02671`.
Panda→Panda는 `-1 * x`.
현재: MJCF 관절 range로 개방률(0~1)을 정규화하고, 여기에 핑거패드 실측 거리를 함께 쓴다
(`runtime.py:651-728`).
**Panda↔xArm7 쌍에 대한 적합된 interpolator가 없으므로 자체 방식이 불가피했지만,
"두 그리퍼의 정규화 개방률이 같으면 시각적으로 같다"는 가정은 검증되지 않았다.**
xArm 그리퍼는 스트로크와 핑거패드 형상이 Panda와 다르므로, 학습 분포상 "반쯤 닫힌 Panda"가
실제로는 다른 물리 상태일 수 있다.

**A-5. 타깃 그리퍼 명령 인가 방식**
공식: blocking 루프 매 iteration마다 `action_target[-1] = self.prev_action[-1]`로 이전 그리퍼
액션을 계속 인가한다(`evaluate_policy_demo_source_robot_server.py:322`). robosuite의
`grip_action`은 물리 substep마다 호출되므로(`robosuite/robots/single_arm.py:258`),
blocking 반복 횟수만큼 그리퍼가 누적 적분되어 사실상 한 policy step 안에서 완전히 닫힌다.
현재: Cartesian/settling 구간은 그리퍼 입력 0으로 고정하고, 수렴 후 **0.25 정규화 증분을
정확히 1회**만 인가한다.
0.25는 Panda 그리퍼 `speed=0.01` × 25 substep으로부터 역산한 값이므로 **소스 Panda의
실제 폐합 속도와는 일치**한다. 즉 공식 구현의 동작(즉시 완전 폐합)보다 학습 분포에
가깝다. 다만 **공식 구현과 다른 동작**이며, 타깃 그리퍼가 8 policy step에 걸쳐 닫히므로
접촉 타이밍이 공식 대비 늦어진다.

### B. 논문/공식 구현에 없는 추가 로직

**B-1. 그리퍼 마스크 1 px 팽창** (`cross_painting.py:159-165`)
공식 sim 경로에는 팽창이 없다(실기/재투영 경로에만 `dilate/erode`가 있다).
팽창은 인페인팅 영역을 넓혀 xArm 그리퍼 경계의 안티에일리어싱 잔여를 지우지만,
동시에 배경/물체 경계를 더 뭉갠다.

**B-2. 물체 픽셀 보호** (`cross_painting.py:166-203`)
`inpaint_mask = (robot | dilated_gripper) & ~object_mask`,
그리고 소스가 앞에 없으면 물체 픽셀을 원본으로 되돌린다.
공식에는 이 개념 자체가 없다. **결과적으로 물체 주변 픽셀 구성이 학습 분포(Panda 씬)와
다른 방식으로 만들어진다.** 특히 그리퍼가 캔을 잡은 상태에서, 공식은 Panda 그리퍼가
캔을 가리도록 두지만 현재 구현은 캔을 더 적극적으로 보존한다.

**B-3. 접촉 검증 기반 제어 완화** (`profiles.py:237-248`, `control.py`)
`verified contact` 상태에서 허용오차를 60 mm / 18°로 완화하고 OSC 게인을
`[300,300,500,300,300,300]`으로 전환한다. 공식에는 전혀 없는 개념이다.
low_dim 경로를 성공시키기 위해 도입된 것으로 보이나, **image 경로에서는 정책이
피드백으로 보는 로봇 위치가 목표에서 최대 60 mm 벗어날 수 있다는 뜻**이다.

**B-4. 마스크 전용 168×168 슈퍼샘플링** (`--mask-render-scale 2`)
공식에 없음. RGB/depth/정책 입력은 84×84 유지.

**B-5. 정책 호출 횟수 (공식이 2회, 현재가 1회)**
공식은 inpaint 활성 시 한 step에서 정책을 **두 번** 호출한다: 먼저 소스 자신의 이미지로
`gt_action`(`evaluate_policy_demo_source_robot_server.py:675`), 이어서 합성 이미지로
`inpainted_img_action`(`:693`). 로컬 체크포인트 15개 중 image 8개가 `algo=bc, rnn=True,
seq_length=10`인 BC-RNN이므로, **공식 경로에서는 RNN 은닉 상태가 env step당 두 번
갱신된다.** 현재 구현은 1회만 호출한다. 논문 수치를 그대로 재현하려면 이 호출 패턴까지
맞춰야 할 수 있다.

### C. 논문/공식 구현에 있으나 미구현

**C-1. Diffusion(ControlNet) cross-painting 변형**
공식은 `analytic / masked / target_robot` 세 가지 diffusion 입력 모드를 지원한다
(`evaluate_policy_demo_target_robot_client.py:44-56, 233-255`). 현재는 해석적 경로만 있다.
논문에서도 해석적 방법이 주 방법이므로 필수는 아니다.

**C-2. 다중 타깃 로봇**
공식은 UR5e/Sawyer/Jaco/Kinova3/IIWA를 지원한다. 현재는 xArm7 단일.
재현 목적상 **Panda→UR5e를 한 번도 재현해 본 적이 없다**는 점이 문제다.
공식 경로가 우리 환경에서 성공하는지 확인되지 않았으므로, 실패 원인이 xArm7 고유인지
파이프라인 전반인지 분리할 근거가 없다.

**C-3. image 경로의 학습 forward dynamics**
현재 image 체크포인트는 `online_oracle`만 지원하고 학습 Ridge replay는 low_dim 전용이다.
시뮬 재현 범위에서는 oracle이 정답이므로 재현성 자체에는 영향 없음.

**C-4. `eef_pose_matrix[1] += 0.1` 류의 태스크별 베이스 오프셋 보정**
공식은 PickPlace에서 Gazebo 베이스 y 오프셋을 하드코딩 보정한다
(`evaluate_policy_demo_target_robot_client.py:216-222`). 현재 구현은 동일 시뮬레이터를
쓰므로 해당 없음. 기록 목적으로만 남긴다.

### D. 수치 파라미터 불일치

**D-1. 소스 정렬 허용오차**
공식 벤치마크 설정: `source_tracking_error_threshold: 0.015`, `source_num_iter_max: 300`
(`config/example_config.yaml:32-33`). 이 오차는 **위치 3성분 + 쿼터니언 4성분을 합친
7차원 벡터의 L2 norm**이다.
현재: 위치 3 mm **및** 자세 2°를 각각, 최대 100회 (`runtime.py:834-841`).
`xarm_transfer/README.md`는 이를 "Mirage 원본과 동일한 3 mm / 2°"라고 기술하는데,
3 mm/100회는 `drive_robot_to_target_pose`의 **함수 기본값**이지 벤치마크가 실제로 쓰는
값이 아니다. **문서 표현을 수정해야 한다.**
(현재 값이 더 엄격하므로 성능상 불리하지는 않다.)

**D-2. 타깃 추종 허용오차 — 가장 중요한 불일치**
공식: 7D norm 0.015, 300회, `kp=[150]*6`.
현재: 위치 **25 mm**, 자세 **7°**, 연속 5 step 유지, 접촉 시 **60 mm / 18°**
(`profiles.py:237-248`).
공식의 0.015 norm은 순수 위치 오차로 환산하면 최대 15 mm이고 자세 성분이 섞이면 더
작아진다. **현재 구현은 위치 기준으로 최소 1.7배, 접촉 상태에서는 4배 이상 느슨하다.**
low_dim 정책은 물체 상태를 직접 관측하므로 이 오차를 견디지만, image 정책은
자기가 보는 로봇 위치가 곧 피드백이므로 **매 step 최대 25~60 mm의 시각적 정렬 오차가
누적되는 폐루프**가 된다.

**D-3. 안전 필터 예측 오차 한계**
`max_prediction_position_error_m=0.03`, `max_prediction_orientation_error_rad=10°`
(`runtime.py:1270-1276`). 공식에는 대응 개념이 없다. oracle 모드에서는 예측 오차가 0이라
발동하지 않으므로 image 경로에는 영향 없음.

### E. 검증되지 않은 가정 / 의심 지점

**E-1. Tool-frame -90° Z 보정**
공식 구현은 소스/타깃 EEF pose를 **아무 보정 없이 직접 주고받는다**
(`action = source_env_robot_state.robot_pose`). 현재 구현은 `SimulationTransform` v2에서
`T_sourceTool_targetTool`에 -90° Z 회전을 넣는다.
자기 정합성은 유지된다(정렬된 Panda의 proprio를 그대로 읽으므로). 다만
**이 보정값이 xArm7 그리퍼 규약에 대해 옳다는 근거는 코드 주석뿐이며 실측/시각 검증
기록이 없다.** 90°가 틀리면 합성 이미지의 Panda 손목 방향이 학습 분포와 어긋난다.
`results/.../cross_paint_breakdown_*.png`에서 육안 확인이 필요하다.

**E-2. 그림자와 마운트 잔여**
로봇 세그멘테이션은 그림자를 포함하지 않으므로, **xArm7이 테이블에 드리운 그림자는
지워지지 않고 최종 정책 이미지에 남는다.** 이는 공식 구현도 동일한 한계이나,
Panda↔UR5e는 실루엣이 비슷해 영향이 작았다. 실측 결과(PickPlaceCan image_proprio seed 0, 400 step) 로봇이 차지하는 픽셀은
타깃 중앙값 533 / 최대 558, 소스 중앙값 776 / 최대 794다 (84×84 = 7056 px).
**소스 Panda가 타깃 xArm보다 중앙값 기준 +46 %, 최대값 기준 +42 % 더 큰 면적을
차지한다.** 그림자·잔여 실루엣의 상대적 비중이 그만큼 크다.

**E-3. 정렬 실패 시 동작**
공식은 정렬 실패 시 경고만 출력하고 롤아웃을 계속한다
(`evaluate_policy_demo_source_robot_server.py:234-239`).
현재는 `RuntimeContractError`를 던져 워커가 죽고 롤아웃이 중단된다
(`runtime.py:860-863`). 재현 실험에서 **공식이라면 이어졌을 에피소드가 조기 종료**될 수 있다.

**E-4. 소스 성공 플래그는 독립적인 증거가 아니다 (2026-08-27 정정)**
image 롤아웃 리포트 25건 전부에서 `source_task_succeeded = target_task_succeeded = false`다.
초판에서는 이를 "소스 env에서조차 실패하므로 인식 단계가 원인"이라고 해석했으나,
**이 추론은 성립하지 않는다.**
cross-paint 경로에서 소스 Panda env는 매 step 타깃의 object qpos/qvel을 복사받고
(`image_runtime.py:269, 279`) 타깃 EEF pose로 정렬된다. 그런데 PickPlace의
`_check_success`는 물체 위치와 그리퍼 site 위치만 본다
(`robosuite/environments/manipulation/pick_place.py` `_check_success`, `not_in_bin`).
두 입력이 모두 타깃에서 복사된 값이므로 **소스 성공 플래그는 타깃 성공을 거의 그대로
반영할 뿐 독립 신호가 아니다.**
(low_dim 경로는 소스가 리더이므로 상황이 다르다. 실제로 Lift에서 소스 55 step /
타깃 57 step으로 값이 갈렸다.)
따라서 "인식이 원인"이라는 결론은 **현재 데이터로 뒷받침되지 않으며**, §5-1의
sanity check를 실제로 돌려야만 인식 경로와 제어 경로를 분리할 수 있다.

---

## 5. 재현 실패와의 연결 — 우선 검증 순서

1. **소스 정책 sanity check (최우선).**
   합성 이미지 대신 **소스 Panda env의 원본 `agentview_image`를 그대로 넣어**
   동일 파이프라인을 돌린다. 여기서도 실패하면 cross-painting이 아니라
   정렬/물체 동기화/관측 조립 경로의 문제다. 성공하면 문제는 합성 픽셀에 있다.
   §4.E-4에 따라 **이 실험 전에는 인식/제어 중 무엇이 원인인지 단정할 수 없다.**

2. **공식 경로 재현 (Panda→UR5e).**
   `run_robosuite_benchmark.py` + Gazebo 인페인팅으로 논문 수치를 한 번 재현한다.
   기준선이 없으면 우리 구현의 실패를 진단할 수 없다.

3. **B-1/B-2 제거 실험.**
   그리퍼 팽창과 물체 보호를 끄고 공식과 동일한 `inpaint_mask = robot_mask`로
   되돌려 비교한다.

4. **D-2 강화 실험.**
   타깃 허용오차를 공식 수준(위치 15 mm 이하, 접촉 완화 비활성)으로 조이고
   image 롤아웃을 재실행한다.

5. **E-1 육안 검증.**
   `--cross-paint-video`로 만든 5-패널 영상에서 합성된 Panda의 손목 방향이
   실제 Panda 롤아웃과 같은지 프레임 단위로 비교한다.

---

## 6. 근거 파일 목록

공식 구현
- `mirage/mirage/benchmark/robosuite/evaluate_policy_demo_target_robot_client.py:66-95, 165-260, 262-336`
- `mirage/mirage/benchmark/robosuite/evaluate_policy_demo_source_robot_server.py:206-275, 505-628, 660-760`
- `mirage/mirage/ros_ws/src/gazebo_env/scripts/write_data_node_robosuite_better.py:336-395, 453-491, 494-525`
- `mirage/mirage/infra/ros_inpaint_publisher_sim.py:6-63`
- `mirage/mirage/gripper_interpolation/robosuite/gripper_interpolator.py:11-66`
- `mirage/mirage/benchmark/robosuite/config/example_config.yaml:28-48`

현재 구현
- `mirage/mirage/benchmark/robosuite/xarm_transfer/cross_painting.py:20-215`
- `mirage/mirage/benchmark/robosuite/xarm_transfer/image_runtime.py:66-159, 203-400`
- `mirage/mirage/benchmark/robosuite/xarm_transfer/runtime.py:120-148, 277-487, 606-728, 765-890, 1167-1205, 1257-1302`
- `mirage/mirage/benchmark/robosuite/xarm_transfer/rollout.py:371-508`
- `mirage/mirage/benchmark/robosuite/xarm_transfer/profiles.py:230-260`
- `mirage/mirage/benchmark/robosuite/xarm_transfer/robosuite_xarm7.py:81-127`

업스트림 규약 확인
- `robosuite/robosuite/environments/robot_env.py:386-412, 445-500`
- `robosuite/robosuite/utils/mjcf_utils.py:93-96` (`opengl` → 1, 플립 없음)
- `robosuite/robosuite/utils/camera_utils.py` `get_real_depth_map`
- `robosuite/robosuite/robots/single_arm.py:235-265` (그리퍼 substep 적분)
- `robosuite/robosuite/models/grippers/panda_gripper.py:61-62` (`speed = 0.01`)
- `robomimic/robomimic/envs/env_robosuite.py:203-240` (`[::-1]` + `process_obs`)
- `robomimic/robomimic/utils/obs_utils.py:365-388, 871-891` (HWC→CHW, `/255`)

실험 증거
- `mirage/mirage/benchmark/robosuite/results/xarm_transfer/pick-place-can/image_*/`
  (모든 image 롤아웃 리포트에서 `source_task_succeeded = target_task_succeeded = false`)
