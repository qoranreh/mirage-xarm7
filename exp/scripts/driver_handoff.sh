#!/usr/bin/env bash
# 9/28: chunk_watch.sh 가 끝나면(can_default/stack_default 가 __s0 로 이동됨) --exclude 로 띄운 드라이버를 exclude 없이 재시작.
# 그러면 __s0 조각이 partial 로 남았더라도 드라이버가 알아서 재실행한다 → 이후 사람 개입 불필요.
LOG=$HOME/hayeon/logs/day2; cd $HOME/hayeon/mirage/exp/day1_matrix || exit 1
until grep -q 'chunk_watch done' $LOG/chunk_watch.log 2>/dev/null; do sleep 60; done
for d in $(pgrep -f '^bash \./run_all\.sh'); do
  # 드라이버가 launch() 도중(pid 파일 쓰기 전)에 죽으면 새 드라이버가 같은 셀을 또 띄울 수 있으므로, sleep 중일 때만 죽인다
  for i in $(seq 1 60); do pgrep -P "$d" -x sleep >/dev/null && break; sleep 1; done
  pg=$(ps -o pgid= -p "$d" | tr -d ' '); if [ "$pg" = "$d" ]; then kill -TERM -- -"$pg"; else kill -TERM "$d"; fi
done
sleep 3
echo "=== handoff $(date '+%F %T'): chunk_watch done → restart without --exclude ===" >> $LOG/run_all.log
setsid nohup ./run_all.sh -n 5 >> $LOG/run_all.log 2>&1 < /dev/null &
sleep 5; echo "$(date '+%F %T') handoff done: driver=$(pgrep -f '^bash \./run_all\.sh' | tr '\n' ' ')" >> $LOG/chunk_watch.log
