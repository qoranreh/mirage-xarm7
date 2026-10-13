#!/usr/bin/env python
"""
collect.py — 셀별 결과(results/<cell>/target.txt)를 모아 Table I / VIII / VI 형식 markdown 으로 (p2)

  python collect.py                          # cells.csv + results/ → results/summary.md (stdout 에도 출력)
  python collect.py --out _dryrun            # 다른 매트릭스 디렉터리

target.txt 형식: json.dump 가 두 번 연속 (avg_rollout_stats, summary_stats) → raw_decode 로 두 객체를 읽는다.
전이 성공률 = 첫 객체의 Success_Rate[0] (롤아웃 평균), 롤아웃 수 = "Num Rollouts".
논문 수치: arXiv 2402.19249 Table I / VIII (Franka gripper / target default gripper), Table VI 는 실로봇 Tiger 과제.
"""
import argparse
import csv
import json
from pathlib import Path

DEFAULT_OUT = Path.home() / "hayeon" / "mirage" / "exp" / "day1_matrix"
TARGET = "xArm7"

# ---- 논문 수치 (Chen et al., RSS 2024, HTML 판에서 전사) ------------------------------------------------
PAPER_ROBOTS = ["UR5e", "IIWA", "Kinova Gen3", "Sawyer", "Jaco"]
# task → (source, [(franka%, default%) per robot])
PAPER_TABLE_I = {
    "lift":      (100, [(99, 100), (100, 100), (100, 100), (100, 100), (100, 99)]),
    "stack":     (95,  [(96, 86),  (94, 87),   (96, 81),   (94, 85),   (96, 66)]),
    "can":       (98,  [(98, 88),  (97, 90),   (97, 83),   (96, 83),   (96, 55)]),
    "two_piece": (96,  [(91, 89),  (89, 70),   (88, 92),   (92, 90),   (90, 34)]),
    "square":    (81,  [(74, 69),  (72, 48),   (34, 49),   (83, 75),   (21, 1)]),
}
PAPER_TABLE_VIII = {  # delta actions
    "lift":      (100, [(91, 99), (94, 53), (96, 98), (98, 72), (96, 63)]),
    "stack":     (95,  [(36, 41), (39, 28), (74, 43), (60, 40), (36, 38)]),
    "can":       (98,  [(87, 90), (93, 25), (89, 73), (78, 41), (86, 60)]),
    "two_piece": (96,  [(44, 44), (65, 45), (54, 48), (43, 29), (41, 15)]),
    "square":    (81,  [(10, 6),  (15, 1),  (3, 2),   (5, 5),   (9, 0)]),
}
PAPER_TABLE_VI = [  # (component, factor, variation, paper %, our cell)
    ("Control", "Gain",   "x0.5", 90, "lift_default_block_iter300_gain0.5"),
    ("Control", "Gain",   "x2",   60, "lift_default_block_iter300_gain2"),
    ("Control", "Offset", "2 cm", 70, "lift_default_block_iter300_zoff2cm"),
    ("Control", "Offset", "5 cm", 20, "lift_default_block_iter300_zoff5cm"),
]
TASK_LABEL = {"lift": "Lift", "stack": "Stack", "can": "Can", "two_piece": "Two Piece Assembly", "square": "Square"}
TASK_ORDER = ["lift", "stack", "can", "two_piece", "square"]


def read_stats(path: Path):
    """두 개의 연속 JSON 객체 → (avg_rollout_stats, summary_stats). 파일 없거나 깨지면 None."""
    if not path.is_file():
        return None
    text = path.read_text()
    dec, objs, i = json.JSONDecoder(), [], 0
    try:
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            obj, j = dec.raw_decode(text, i)
            objs.append(obj); i = j
    except json.JSONDecodeError:
        return None
    return objs[0] if objs else None


def load_results(out: Path):
    """cells.csv 행을 base 셀 단위로 합산. 9/28 부터 느린 셀은 `<cell>__s<k>` 조각(seed k, 5 rollouts)으로 나뉘어
    base 열에 원래 셀 이름이 있다. 조각들의 Num_Success / Num Rollouts 를 더해 성공률을 다시 계산한다 (미분할 셀은 동일)."""
    cells = {}
    with open(out / "cells.csv") as f:
        for r in csv.DictReader(f):
            st = read_stats(Path(r["results_dir"]) / "target.txt")
            src = read_stats(Path(r["results_dir"]) / "source.txt")
            base = r.get("base") or r["cell"]
            c = cells.setdefault(base, dict(r, cell=base, n_rollouts=0, pieces=0, n_done=0, n_succ=0.0, src_n=0, src_succ=0.0))
            c["n_rollouts"] += int(r["n_rollouts"]); c["pieces"] += 1
            if st:
                c["n_done"] += int(st["Num Rollouts"]); c["n_succ"] += float(st["Num_Success"][0])
            if src:
                c["src_n"] += int(src["Num Rollouts"]); c["src_succ"] += float(src["Num_Success"][0])
    for c in cells.values():
        c["sr"] = c["n_succ"] / c["n_done"] if c["n_done"] else None
        c["src_sr"] = c["src_succ"] / c["src_n"] if c["src_n"] else None
    return cells


def fmt(cell, expect):
    if cell is None or cell["sr"] is None:
        return "—"
    s = f"{100 * cell['sr']:.0f}%"
    if cell["n_done"] < expect:
        s += f" (n={cell['n_done']})"
    return s


def table_I_like(cells, paper, mode_suffix, title, caption):
    hdr = ["Task", "Source (Franka, paper)", "Source (Panda, ours)*"] + [f"{r} (paper) F/D" for r in PAPER_ROBOTS] + [f"**{TARGET} F/D (ours)**"]
    lines = [f"### {title}", "", caption, "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for t in TASK_ORDER:
        src, per = paper[t]
        cp = cells.get(f"{t}_pandagrip_{mode_suffix}"); cd = cells.get(f"{t}_default_{mode_suffix}")
        expect = int((cp or cd or {"n_rollouts": 50})["n_rollouts"])
        ours_src = next((c["src_sr"] for c in (cp, cd) if c and c["src_sr"] is not None), None)
        row = [TASK_LABEL[t], f"{src}%", ("—" if ours_src is None else f"{100 * ours_src:.0f}%")]
        row += [f"{f}/{d}" for f, d in per]
        row.append(f"{fmt(cp, expect)} / {fmt(cd, expect)}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else DEFAULT_OUT / args.out
    cells = load_results(out)
    global TARGET
    TARGET = next(iter(cells.values()))["target_robot"]  # 열 이름은 cells.csv 의 실제 타깃 로봇
    done = sum(1 for c in cells.values() if c["sr"] is not None)

    md = [f"# day1_matrix 결과 요약  ({done}/{len(cells)} cells with results; target={next(iter(cells.values()))['target_robot']})", ""]
    md.append(table_I_like(cells, PAPER_TABLE_I, "block_iter300", "Table I — state-based transfer (blocking, num_iter_max=300)",
                           "F/D = Franka(Panda) gripper / target default gripper. 논문 열은 Table I 그대로. "
                           "*ours source = source.txt 의 Success_Rate — passive 모드에선 타깃이 성공하면 소스 루프가 먼저 끊겨 과소집계됨(참고용). 논문 source 열은 별도 source-only 실행값."))
    odd = sorted(c["cell"] for c in cells.values() if c["cell"].endswith("block_iter300") and str(c["num_iter_max"]) != "300")
    if odd:
        md.append("※ blocking 상한 `num_iter_max=100` 으로 실행한 셀: " + ", ".join(f"`{c}`" for c in odd)
                  + " (9/29 walltime 대책. 도달 불가 스텝의 300회 반복이 시간의 대부분이었고, 51~299회에서 수렴하는 스텝은 ≤1 % 라 영향 미미. 나머지 block_iter300 셀은 300.)\n")
    md.append(table_I_like(cells, PAPER_TABLE_VIII, "delta_iter1", "Table VIII — delta actions (target executes the same delta action)",
                           "논문 Table VIII 과 같은 구조. 우리 셀 = `*_delta_iter1`."))
    # block_iter1 (비블로킹 절대제어) 은 논문에 없음 → 별도 표
    hdr = ["Task", f"{TARGET} block_iter1 F/D", f"{TARGET} block_iter300 F/D (=Table I)", f"{TARGET} delta F/D (=Table VIII)"]
    lines = ["### Ablation — num_iter_max 1 vs 300 (absolute pose, non-blocking vs blocking)", "",
             "논문 대응 표 없음. delta 모드는 num_iter_max 를 쓰지 않으므로 delta_iter300 셀은 생성하지 않았다.", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for t in TASK_ORDER:
        row = [TASK_LABEL[t]]
        for suf in ("block_iter1", "block_iter300", "delta_iter1"):
            cp, cd = cells.get(f"{t}_pandagrip_{suf}"), cells.get(f"{t}_default_{suf}")
            expect = int((cp or cd or {"n_rollouts": 50})["n_rollouts"])
            row.append(f"{fmt(cp, expect)} / {fmt(cd, expect)}")
        lines.append("| " + " | ".join(row) + " |")
    md.append("\n".join(lines) + "\n")

    hdr = ["Component", "Factor", "Variation", "Paper (Tiger pick-place, real robot)", f"**{TARGET} Lift, default gripper (ours)**"]
    lines = ["### Table VI — sensitivity (control gain / proprio z-offset)", "",
             "논문 Table VI 는 실로봇 Tiger 과제 수치라 직접 비교가 아니라 경향 비교용. 우리 셀은 Lift/default/blocking300 기준.", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    base = cells.get("lift_default_block_iter300")
    lines.append(f"| Control | (baseline) | — | — | {fmt(base, int(base['n_rollouts']) if base else 50)} |")
    for comp, fac, var, paper, cell in PAPER_TABLE_VI:
        c = cells.get(cell)
        lines.append(f"| {comp} | {fac} | {var} | {paper}% | {fmt(c, int(c['n_rollouts']) if c else 50)} |")
    md.append("\n".join(lines) + "\n")

    # 셀 전체 상태
    lines = ["### 셀 상태 (조각 = `<cell>__s<k>` 5-rollout 조각 수)", "", "| cell | table | 조각 | done/n | target SR | source SR |", "|---|---|---|---|---|---|"]
    for c in cells.values():
        sr = "—" if c["sr"] is None else f"{100 * c['sr']:.0f}%"
        ssr = "—" if c["src_sr"] is None else f"{100 * c['src_sr']:.0f}%"
        lines.append(f"| {c['cell']} | {c['table']} | {c['pieces']} | {c['n_done']}/{c['n_rollouts']} | {sr} | {ssr} |")
    md.append("\n".join(lines) + "\n")

    text = "\n".join(md)
    (out / "results").mkdir(exist_ok=True)
    (out / "results" / "summary.md").write_text(text)
    print(text)
    print(f"→ {out / 'results' / 'summary.md'}")


if __name__ == "__main__":
    main()
