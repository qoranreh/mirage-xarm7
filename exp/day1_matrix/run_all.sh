#!/usr/bin/env bash
# run_all.sh — cells.csv 의 셀들을 N 개씩 병렬 실행 (p2)
#
#   ./run_all.sh -n 3                        # 동시 3셀, 기본 cells.csv, 우선순위(csv 순서)대로
#   ./run_all.sh -n 2 -f _dryrun/cells.csv   # 다른 매트릭스
#   ./run_all.sh -n 3 --filter '^lift_'      # 셀 이름 정규식 필터
#   ./run_all.sh -n 5 --exclude '__s0$'      # 정규식에 맞는 셀은 실행하지 않음 (--status 에는 표시)
#   ./run_all.sh --dry                       # 실행 없이 무엇을 돌릴지만 출력
#   ./run_all.sh --status                    # 실행 중 / 완료 / 대기 요약
#
# 백엔드 (자동 선택, --backend 로 강제):
#   tmux : tmux 세션 `day2` 에 창 하나 = 셀 하나 (창 이름 = 셀 이름). 로그인 노드 등 tmux 가 있을 때.
#   bg   : nohup+setsid 백그라운드 프로세스 + PID 파일 (~/hayeon/logs/day2/<cell>.pid). 컨테이너(NGC 이미지엔 tmux 없음).
#          스크립트 자체를 nohup 으로 띄우면 셸을 닫아도 대기열이 계속 소비된다:  nohup ./run_all.sh -n 3 > ~/hayeon/logs/day2/run_all.log 2>&1 &
# - results/<cell>/target.txt 의 "Num Rollouts" 가 csv 의 n_rollouts 이상이면 완료로 보고 skip (재개 가능).
#   러너는 롤아웃마다 target.txt 를 덮어쓰므로 중간에 죽은 셀은 부분 결과가 남는다 → 파일 존재만으로 판단하면 영영 건너뛰게 됨 (9/28 수정)
# - 로그: ~/hayeon/logs/day2/<cell>.log   (첫 줄 start, 마지막 줄 exit=<code>)
# - 각 셀은 run_robosuite_benchmark.py -y 가 서버/클라이언트 프로세스 2개를 띄운다 (mujoco CPU-bound → 6코어 노드에서 -n 3 권장)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION=day2
N=3
CSV="$HERE/cells.csv"
FILTER=""
EXCLUDE=""
DRY=0
STATUS=0
BACKEND=""
LOGDIR="$HOME/hayeon/logs/day2"
ENVSH="$HOME/hayeon/scripts/env_mirage.sh"
BENCH="$HOME/hayeon/mirage/mirage/mirage/benchmark/robosuite"
POLL=20

while [ $# -gt 0 ]; do
  case "$1" in
    -n) N="$2"; shift 2 ;;
    -f) CSV="$2"; shift 2 ;;
    --filter) FILTER="$2"; shift 2 ;;
    --exclude) EXCLUDE="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --dry) DRY=1; shift ;;
    --status) STATUS=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
[[ "$CSV" = /* ]] || CSV="$HERE/$CSV"
[ -f "$CSV" ] || { echo "cells.csv 없음: $CSV  (먼저 gen_configs.py)"; exit 2; }
mkdir -p "$LOGDIR"
if [ -z "$BACKEND" ]; then command -v tmux >/dev/null 2>&1 && BACKEND=tmux || BACKEND=bg; fi

# ---- 백엔드별: 실행 중 셀 목록 / 실행 -------------------------------------------------------------
running_cells() {
  if [ "$BACKEND" = tmux ]; then
    tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -v '^__keep$' || true
  else
    local p
    for p in "$LOGDIR"/*.pid; do
      [ -f "$p" ] || continue
      if kill -0 "$(cat "$p")" 2>/dev/null; then basename "$p" .pid; else rm -f "$p"; fi
    done
  fi
}
running_count() { running_cells | grep -c . || true; }
n_done() { grep -m1 '"Num Rollouts"' "$1/target.txt" 2>/dev/null | grep -oE '[0-9]+' | head -1; }   # 첫 JSON 객체(avg_rollout_stats)의 롤아웃 수
is_done() {  # $1=results_dir $2=n_rollouts
  local n; n=$(n_done "$1")
  [ -n "$n" ] && [ "$n" -ge "${2:-1}" ]
}
launch() {  # $1=cell $2=config
  local cell=$1 config=$2 log="$LOGDIR/$1.log"
  local cmd="source $ENVSH; cd $BENCH; echo \"start \$(date '+%F %T') cell=$cell\" > $log; PYTHONUNBUFFERED=1 python run_robosuite_benchmark.py --config $config -y >> $log 2>&1; echo \"exit=\$? end \$(date '+%F %T')\" >> $log"
  if [ "$BACKEND" = tmux ]; then
    tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n __keep 'sleep infinity'
    tmux new-window -d -t "$SESSION" -n "$cell" "bash -c '$cmd'"
  else
    nohup setsid bash -c "$cmd" </dev/null >/dev/null 2>&1 &
    echo $! > "$LOGDIR/$cell.pid"
  fi
}

# ---- csv: cell,priority,table,task,gripper,mode,num_iter_max,variant,config,results_dir,... ------------
if [ "$STATUS" = 1 ]; then
  echo "backend=$BACKEND  running($(running_count)): $(running_cells | tr '\n' ' ')"
  while IFS=, read -r cell priority table task gripper mode n_iter variant config results_dir rest; do
    [ "$cell" = "cell" ] && continue
    n_roll=${rest%%,*}
    if is_done "$results_dir" "$n_roll"; then s=done
    elif running_cells | grep -qx "$cell"; then s="RUNNING(n=$(n_done "$results_dir"))"
    elif [ -f "$results_dir/target.txt" ]; then s="partial(n=$(n_done "$results_dir")/$n_roll → 재실행 대상)"
    elif [ -f "$LOGDIR/$cell.log" ]; then s="stopped($(tail -1 "$LOGDIR/$cell.log" | cut -c1-40))"
    else s=pending; fi
    printf "  %-45s %s\n" "$cell" "$s"
  done < "$CSV"
  exit 0
fi

total=0; skipped=0; launched=0
while IFS=, read -r cell priority table task gripper mode n_iter variant config results_dir rest; do
  [ "$cell" = "cell" ] && continue
  [ -n "$FILTER" ] && ! [[ "$cell" =~ $FILTER ]] && continue
  [ -n "$EXCLUDE" ] && [[ "$cell" =~ $EXCLUDE ]] && continue
  total=$((total+1))
  n_roll=${rest%%,*}
  if is_done "$results_dir" "$n_roll"; then echo "skip (done)    $cell"; skipped=$((skipped+1)); continue; fi
  if running_cells | grep -qx "$cell"; then echo "skip (running) $cell"; continue; fi
  if [ "$DRY" = 1 ]; then echo "would run      $cell  ($config)"; continue; fi
  while [ "$(running_count)" -ge "$N" ]; do sleep "$POLL"; done
  launch "$cell" "$config"
  launched=$((launched+1))
  echo "launched [$BACKEND] $cell  → $LOGDIR/$cell.log   (running: $(running_count)/$N)"
  sleep 2
done < "$CSV"
echo "total=$total skipped=$skipped launched=$launched backend=$BACKEND | 상태: $0 --status ; 로그: tail -f $LOGDIR/<cell>.log"
