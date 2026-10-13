#!/usr/bin/env python
"""
gen_configs.py — Mirage state-based transfer 실험 매트릭스의 셀별 config yaml 생성 (p2)

  python gen_configs.py                                  # 32 셀, target=XArm7, 50 rollouts → configs/, cells.csv
  python gen_configs.py --target-robot UR5e --n-rollouts 2 --out _dryrun   # 파이프라인 검증용 대역

파일명 규칙: {task}_{gripper}_{delta|block}_iter{N}{_variant}.yaml
  gripper : pandagrip (target_gripper_type=PandaGripper) | default (필드 생략 = 로봇 기본 그리퍼)
  variant : _zoff2cm _zoff5cm (target_proprio_z_offset) | _gain0.5 _gain2 (target_kp_scale)   ← Table VI, Lift/default 만

셀 구성:
  Table I    : task × gripper × block_iter300                         (10)
  Table VIII : task × gripper × {delta_iter1, block_iter1}             (20)  block_iter300 은 Table I 재사용,
                                                                              delta 모드는 num_iter_max 를 안 쓰므로 delta_iter300 ≡ delta_iter1 (생성 안 함)
  Table VI   : lift × default × block_iter300 × {gain0.5, gain2}                    (2)   zoff2cm/zoff5cm 은 --with-zoff (기본 제외, 아래 ZOFF_VARIANTS 주석)

seed: config 의 seed 하나가 서버/클라이언트에 --seeds 로 전달되고, run_experiments 가 롤아웃 i 에
      set_seed(seed * n_rollouts + i) 를 하므로 seed=0, n_rollouts=50 → 롤아웃별 seed 0..49 (고정).
"""
import argparse
import csv
import re
from pathlib import Path

import yaml

REPO = Path.home() / "hayeon" / "mirage"
MODELS = REPO / "mirage" / "mirage" / "models"
DEFAULT_OUT = REPO / "exp" / "day1_matrix"

# task → (low_dim 체크포인트, 학습 config 의 rollout.horizon)   ※ 체크포인트에서 읽은 값
TASKS = {
    "lift":      ("lift/low_dim/lift_ph_low_dim_epoch_1000_succ_100.pth", 400),
    "can":       ("can/low_dim/model_epoch_250_PickPlaceCan_success_1.0.pth", 400),
    "stack":     ("stack/low_dim/model_epoch_2000.pth", 400),
    "square":    ("square/low_dim/model_epoch_950_Square_D0_success_0.96.pth", 400),
    "two_piece": ("two_piece/low_dim/model_epoch_1000_ThreePieceAssembly_D0_success_0.68.pth", 500),
}
PRIORITY = ["lift", "can", "stack", "square", "two_piece"]
GRIPPERS = {"pandagrip": "PandaGripper", "default": None}
VARIANTS = {  # Table VI (Lift, default gripper, blocking 300)
    "_gain0.5": {"target_kp_scale": 0.5},
    "_gain2":   {"target_kp_scale": 2.0},
}
# z-offset 변형은 기본 매트릭스에서 제외 (9/28 결정, --with-zoff 로 포함).
# 현 구현(명령 z −= b)에서는 xArm7 그리퍼가 테이블에 막혀 추종 오차 0.025~0.04 > 0.015 가 되어 매 스텝 blocking 300회를 다 돌고
# (스텝당 ~45 s, 롤아웃 1회 2~5 h, 50회 100 h+), 성공도 0 이었다. 구현(오프셋 부호·의미)과 롤아웃 수를 다시 정한 뒤 별도 실행.
ZOFF_VARIANTS = {
    "_zoff2cm": {"target_proprio_z_offset": 0.02},
    "_zoff5cm": {"target_proprio_z_offset": 0.05},
}
# 느린 blocking 셀 분할 (9/28). Can 은 xArm7 도달거리 문제로 스텝의 절반이 추종 오차 >10 cm 로 300회 상한을 돌아
# 롤아웃당 ~55 min → 셀당 45~55 h. 이런 셀은 (seed=k, n_rollouts=CHUNK_SIZE) 조각 여러 개로 나눠 병렬 실행하고
# 중단 시 손실 단위를 줄인다. 서버 run_experiments 가 롤아웃 i 를 set_seed(seed*n_rollouts+i) 로 시드하므로
# 조각 k 는 시드 5k..5k+4 → 조각 0..9 가 원래 (seed=0, n=50) 의 시드 0..49 와 정확히 같다 (같은 실험).
# stack_pandagrip_block_iter300 은 분할 전에 완료되어 제외. collect.py 가 base 열로 조각을 합산한다. --no-chunk 로 끔.
CHUNK_CELLS = re.compile(r"^(can|square|two_piece)_(pandagrip|default)_block_iter300$|^stack_default_block_iter300$")
CHUNK_SIZE = 5
# blocking 상한 완화 (9/29). Square/Two Piece 는 실패 롤아웃에서 xArm7 이 도달 못 하는 자세(추종 오차 대부분 >5 cm)에 매 스텝
# 300회를 다 돌아(스텝당 ~45 s) 조각당 5 h+ → 72 h walltime 초과 위험. 51~299회에서 뒤늦게 수렴하는 스텝은 Can 0.16 %, Stack 1 %,
# Square 0.1 % 뿐이라 상한 100(원저자 코드 함수 기본값; 300 은 example_config 값)으로 낮춰도 결과는 사실상 같다.
# 아직 시작 안 한 셀에만 적용, Square F 는 300 으로 마무리(셀 내 일관). 셀 이름은 *_block_iter300 그대로(collect.py 조회 키),
# 실제 값은 cells.csv 의 num_iter_max 열과 collect.py 각주에 나온다. --no-iter-override 로 끔.
ITER_OVERRIDE = re.compile(r"^(square_default|two_piece_(pandagrip|default))_block_iter300$")
ITER_OVERRIDE_VALUE = 100
# 큐 뒤로 미루는 셀 (9/30 사용자 결정). Two Piece blocking 셀은 소요가 가장 불확실(D 가 Square D 처럼 전부 실패하면 조각당 ~7.5 h)해서
# Table VIII 뒤로 보낸다 → Square D + VIII 가 walltime 안에 끝나고, Two Piece 는 남는 시간에 돌다가 새 잡에서 이어감. --no-defer 로 끔.
DEFER_CELLS = re.compile(r"^two_piece_(pandagrip|default)_block_iter300$")


def template(task, target_robot, n_rollouts, seed, source_robot):
    ckpt, horizon = TASKS[task]
    ckpt = str(MODELS / ckpt)
    assert Path(ckpt).is_file(), f"checkpoint 없음: {ckpt}"
    # example_config.yaml 의 전 필드 + from_yaml 이 필수로 읽는 naive
    return {
        "source_agent_path": ckpt,
        "target_agent_path": ckpt,
        "n_rollouts": n_rollouts,
        "horizon": horizon,
        "seed": seed,
        "passive": True,
        "connection": True,
        "naive": False,
        "source_robot_name": source_robot,
        "target_robot_name": target_robot,
        "source_tracking_error_threshold": 0.015,
        "source_num_iter_max": 300,
        "target_tracking_error_threshold": 0.015,
        "target_num_iter_max": 300,
        "delta_action": False,
        "enable_inpainting": False,
        "use_ros": False,
        "offline_eval": False,
        "use_diffusion": False,
        "diffusion_input_type": "",
        "results_folder": None,  # 셀별로 채움
    }


def build_cells(variants=VARIANTS):
    """(cell_name, task, gripper_key, mode, num_iter_max, variant, table, overrides) 목록.
    실행 순서(= cells.csv 순서) : Table I + VI 전부(task 우선순위 순) → 그 뒤에 Table VIII(task 우선순위 순).
    9/28 결정: VIII 는 부차적이라 본실험 결과(I, VI)를 먼저 확보."""
    main, ablation = [], []
    for task in PRIORITY:
        for gkey, gname in GRIPPERS.items():
            base = {"target_gripper_type": gname} if gname else {}
            main.append((f"{task}_{gkey}_block_iter300", task, gkey, "block", 300, "", "I", dict(base)))
            if task == "lift" and gkey == "default":
                for vkey, vover in variants.items():
                    main.append((f"{task}_{gkey}_block_iter300{vkey}", task, gkey, "block", 300, vkey, "VI", dict(base, **vover)))
            ablation.append((f"{task}_{gkey}_delta_iter1", task, gkey, "delta", 1, "", "VIII",
                             dict(base, delta_action=True, source_num_iter_max=1, target_num_iter_max=1)))
            ablation.append((f"{task}_{gkey}_block_iter1", task, gkey, "block", 1, "", "VIII",
                             dict(base, source_num_iter_max=1, target_num_iter_max=1)))
    return main + ablation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-robot", default="XArm7", help="robosuite ROBOT_CLASS_MAPPING 키 (대소문자 정확히; xArm7 은 KeyError). 파이프라인 검증엔 UR5e")
    ap.add_argument("--source-robot", default="Panda")
    ap.add_argument("--n-rollouts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="configs/, results/, cells.csv 가 생기는 디렉터리")
    ap.add_argument("--with-zoff", action="store_true", help="Table VI z-offset 변형(zoff2cm/zoff5cm)도 포함 (기본 제외, 위 주석 참고)")
    ap.add_argument("--no-chunk", action="store_true", help="CHUNK_CELLS 셀을 5롤아웃 조각으로 나누지 않음")
    ap.add_argument("--no-iter-override", action="store_true", help="ITER_OVERRIDE 셀의 num_iter_max 를 100 으로 낮추지 않음")
    ap.add_argument("--no-defer", action="store_true", help="DEFER_CELLS 를 큐 뒤로 미루지 않음")
    args = ap.parse_args()
    variants = dict(VARIANTS, **ZOFF_VARIANTS) if args.with_zoff else VARIANTS

    out = args.out if args.out.is_absolute() else DEFAULT_OUT / args.out
    cfg_dir, res_dir = out / "configs", out / "results"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_chunked = 0
    for cell, task, gkey, mode, n_iter, variant, table, over in build_cells(variants):
        # (name, seed, n_rollouts) 조각 목록. 분할 셀: 조각 k 의 seed = args.seed*조각수 + k → 시드가 원래 셀과 동일
        if not args.no_chunk and CHUNK_CELLS.match(cell) and args.n_rollouts > CHUNK_SIZE and args.n_rollouts % CHUNK_SIZE == 0:
            n_pieces = args.n_rollouts // CHUNK_SIZE
            pieces = [(f"{cell}__s{k}", args.seed * n_pieces + k, CHUNK_SIZE) for k in range(n_pieces)]
            n_chunked += 1
        else:
            pieces = [(cell, args.seed, args.n_rollouts)]
        for name, seed, n_roll in pieces:
            cfg = template(task, args.target_robot, n_roll, seed, args.source_robot)
            cfg.update(over)
            if not args.no_iter_override and ITER_OVERRIDE.match(cell):
                cfg["source_num_iter_max"] = cfg["target_num_iter_max"] = ITER_OVERRIDE_VALUE
            n_iter = cfg["target_num_iter_max"]
            cfg["results_folder"] = str(res_dir / name)
            path = cfg_dir / f"{name}.yaml"
            with open(path, "w") as f:
                f.write(f"# cell={name} base={cell} table={table} task={task} gripper={gkey} mode={mode} num_iter_max={n_iter} variant={variant or '-'}\n")
                f.write(f"# generated by gen_configs.py (target={args.target_robot}, n_rollouts={n_roll}, seed={seed}; rollout i seed = seed*n_rollouts+i)\n")
                yaml.safe_dump(cfg, f, sort_keys=False)
            rows.append(dict(cell=name, priority=len(rows), table=table, task=task, gripper=gkey, mode=mode, num_iter_max=n_iter,
                             variant=variant or "-", config=str(path), results_dir=str(res_dir / name),
                             n_rollouts=n_roll, target_robot=args.target_robot, base=cell, seed=seed))

    if not args.no_defer:  # DEFER_CELLS 조각을 맨 뒤로 (안정 정렬), priority 다시 매김
        rows = [r for r in rows if not DEFER_CELLS.match(r["base"])] + [r for r in rows if DEFER_CELLS.match(r["base"])]
        for i, r in enumerate(rows):
            r["priority"] = i
    with open(out / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    by_table = {}
    for r in rows:
        by_table[r["table"]] = by_table.get(r["table"], 0) + 1
    n_base = len({r["base"] for r in rows})
    print(f"{len(rows)} rows ({n_base} cells, {n_chunked} chunked into {args.n_rollouts // CHUNK_SIZE}×{CHUNK_SIZE}) → {cfg_dir}  "
          f"(rows: Table I={by_table.get('I',0)}, VIII={by_table.get('VIII',0)}, VI={by_table.get('VI',0)})")
    print(f"cells.csv → {out / 'cells.csv'}   target={args.target_robot} n_rollouts={args.n_rollouts} seed={args.seed}")


if __name__ == "__main__":
    main()
