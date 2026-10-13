# 사용:  source ~/hayeon/scripts/env_mirage.sh
# Mirage venv + 헤드리스 OSMesa 렌더링.
# 컨테이너(NGC pytorch:23.10)에 libEGL/libOSMesa 가 없어서 conda-forge mesalib<24 의 libOSMesa 를 LD_LIBRARY_PATH 로 물린다.
source ~/venvs/mirage/bin/activate
export MESA_PREFIX=~/.conda/envs/mesa-osmesa
export LD_LIBRARY_PATH="$MESA_PREFIX/lib:$LD_LIBRARY_PATH"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
# 주의 1: ~/hayeon/mirage (리포 루트) 에서 python 을 실행하지 말 것 — 그 안의 robosuite/ 디렉터리가 진짜 패키지를 가린다.
# 주의 2: pip 은 항상 `python -m pip` (venv 에 pip 바이너리가 없어 그냥 pip 은 다른 파이썬의 pip 을 잡는다).
