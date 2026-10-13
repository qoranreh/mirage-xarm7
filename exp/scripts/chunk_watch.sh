#!/usr/bin/env bash
# 9/28 분할 전환용: 실행 중이던 unchunked 셀이 n=5 (= 조각 s0, 시드 0..4) 에 도달하면 즉시 중단하고 결과를 <cell>__s0 로 옮긴다.
# 셀이 스스로 죽은 경우에도(부분 결과) 같은 이름으로 옮긴다 → run_all.sh 가 partial 로 보고 s0 를 재실행.
LOG=$HOME/hayeon/logs/day2; RES=$HOME/hayeon/mirage/exp/day1_matrix/results
cells="can_default_block_iter300 stack_default_block_iter300"
while [ -n "$(echo $cells)" ]; do
  for c in $cells; do
    n=$(grep -m1 '"Num Rollouts"' $RES/$c/target.txt 2>/dev/null | grep -oE '[0-9]+' | head -1)
    alive=0; [ -f $LOG/$c.pid ] && kill -0 "$(cat $LOG/$c.pid)" 2>/dev/null && alive=1
    if [ "${n:-0}" -ge 5 ] || [ $alive = 0 ]; then
      [ $alive = 1 ] && { kill -KILL -- -"$(cat $LOG/$c.pid)" 2>/dev/null; sleep 2; }
      rm -f $LOG/$c.pid
      n2=$(grep -m1 '"Num Rollouts"' $RES/$c/target.txt 2>/dev/null | grep -oE '[0-9]+' | head -1)
      mv $RES/$c $RES/${c}__s0 && mv $LOG/$c.log $LOG/${c}__s0.log
      echo "$(date '+%F %T') $c: stopped at n=${n2:-0} (alive_before=$alive) → results moved to ${c}__s0"
      cells=$(echo " $cells " | sed "s/ $c / /")
    fi
  done
  sleep 30
done
echo "$(date '+%F %T') chunk_watch done"
