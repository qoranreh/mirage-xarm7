# 서버 작업 (2026-09, A100 클러스터) — state 기반 transfer, xArm7 확장

`main`(3c8570b) 기준. `agent/xarm7-development` 브랜치(7~8월 작업)와 별개로 진행했고, 수정한 파일은 서로 겹치지 않는다.

| 경로 | 내용 |
|---|---|
| `mirage/mirage/benchmark/robosuite/*`, `gripper_interpolation/robosuite/gripper_interpolator.py` | 벤치마크 server/client·config·gripper interpolator 수정 |
| `patches/server/mirage-robosuite-7ce0456-xarm7.patch` | robosuite 서브모듈(7ce0456)에 xArm7 로봇·그리퍼 추가 (`git apply` 로 적용, STL 포함 binary patch) |
| `patches/server/mirage-robomimic-a540f08-diffusion-policy.patch` | robomimic 서브모듈(a540f08)에 diffusion policy 추가 |
| `exp/day1_matrix/` | 실험 매트릭스 (configs, run_all.sh, collect.py, results) — 자세한 내용은 그 안 README |
| `exp/dumps_day1/` | Panda vs xArm7 agentview RGB·segmentation 덤프 (원래 위치 `~/hayeon/dumps/day1`) |
| `exp/scripts/` | env_mirage.sh 등 (원래 위치 `~/hayeon/scripts`, 스크립트 안의 경로는 원래 위치 기준) |

서브모듈 패치 적용:
```bash
git submodule update --init robosuite robomimic
git -C robosuite apply ../patches/server/mirage-robosuite-7ce0456-xarm7.patch
git -C robomimic apply ../patches/server/mirage-robomimic-a540f08-diffusion-policy.patch
```
실행 로그(`~/hayeon/logs/day2`)는 포함하지 않았다.
