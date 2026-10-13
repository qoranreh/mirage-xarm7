# day1_matrix 결과 요약  (32/32 cells with results; target=XArm7)

### Table I — state-based transfer (blocking, num_iter_max=300)

F/D = Franka(Panda) gripper / target default gripper. 논문 열은 Table I 그대로. *ours source = source.txt 의 Success_Rate — passive 모드에선 타깃이 성공하면 소스 루프가 먼저 끊겨 과소집계됨(참고용). 논문 source 열은 별도 source-only 실행값.

| Task | Source (Franka, paper) | Source (Panda, ours)* | UR5e (paper) F/D | IIWA (paper) F/D | Kinova Gen3 (paper) F/D | Sawyer (paper) F/D | Jaco (paper) F/D | **XArm7 F/D (ours)** |
|---|---|---|---|---|---|---|---|---|
| Lift | 100% | 90% | 99/100 | 100/100 | 100/100 | 100/100 | 100/99 | 100% / 100% |
| Stack | 95% | 0% | 96/86 | 94/87 | 96/81 | 94/85 | 96/66 | 96% / 90% |
| Can | 98% | 20% | 98/88 | 97/90 | 97/83 | 96/83 | 96/55 | 80% / 72% |
| Two Piece Assembly | 96% | 20% | 91/89 | 89/70 | 88/92 | 92/90 | 90/34 | 96% / 98% |
| Square | 81% | 4% | 74/69 | 72/48 | 34/49 | 83/75 | 21/1 | 26% / 2% |

※ blocking 상한 `num_iter_max=100` 으로 실행한 셀: `square_default_block_iter300`, `two_piece_default_block_iter300`, `two_piece_pandagrip_block_iter300` (9/29 walltime 대책. 도달 불가 스텝의 300회 반복이 시간의 대부분이었고, 51~299회에서 수렴하는 스텝은 ≤1 % 라 영향 미미. 나머지 block_iter300 셀은 300.)

### Table VIII — delta actions (target executes the same delta action)

논문 Table VIII 과 같은 구조. 우리 셀 = `*_delta_iter1`.

| Task | Source (Franka, paper) | Source (Panda, ours)* | UR5e (paper) F/D | IIWA (paper) F/D | Kinova Gen3 (paper) F/D | Sawyer (paper) F/D | Jaco (paper) F/D | **XArm7 F/D (ours)** |
|---|---|---|---|---|---|---|---|---|
| Lift | 100% | 94% | 91/99 | 94/53 | 96/98 | 98/72 | 96/63 | 96% / 82% |
| Stack | 95% | 44% | 36/41 | 39/28 | 74/43 | 60/40 | 36/38 | 50% / 42% |
| Can | 98% | 60% | 87/90 | 93/25 | 89/73 | 78/41 | 86/60 | 74% / 28% |
| Two Piece Assembly | 96% | 42% | 44/44 | 65/45 | 54/48 | 43/29 | 41/15 | 50% / 30% |
| Square | 81% | 4% | 10/6 | 15/1 | 3/2 | 5/5 | 9/0 | 6% / 0% |

### Ablation — num_iter_max 1 vs 300 (absolute pose, non-blocking vs blocking)

논문 대응 표 없음. delta 모드는 num_iter_max 를 쓰지 않으므로 delta_iter300 셀은 생성하지 않았다.

| Task | XArm7 block_iter1 F/D | XArm7 block_iter300 F/D (=Table I) | XArm7 delta F/D (=Table VIII) |
|---|---|---|---|
| Lift | 100% / 100% | 100% / 100% | 96% / 82% |
| Stack | 92% / 88% | 96% / 90% | 50% / 42% |
| Can | 94% / 86% | 80% / 72% | 74% / 28% |
| Two Piece Assembly | 90% / 98% | 96% / 98% | 50% / 30% |
| Square | 18% / 24% | 26% / 2% | 6% / 0% |

### Table VI — sensitivity (control gain / proprio z-offset)

논문 Table VI 는 실로봇 Tiger 과제 수치라 직접 비교가 아니라 경향 비교용. 우리 셀은 Lift/default/blocking300 기준.

| Component | Factor | Variation | Paper (Tiger pick-place, real robot) | **XArm7 Lift, default gripper (ours)** |
|---|---|---|---|---|
| Control | (baseline) | — | — | 100% |
| Control | Gain | x0.5 | 90% | 100% |
| Control | Gain | x2 | 60% | 100% |
| Control | Offset | 2 cm | 70% | — |
| Control | Offset | 5 cm | 20% | — |

### 셀 상태 (조각 = `<cell>__s<k>` 5-rollout 조각 수)

| cell | table | 조각 | done/n | target SR | source SR |
|---|---|---|---|---|---|
| lift_pandagrip_block_iter300 | I | 1 | 50/50 | 100% | 90% |
| lift_default_block_iter300 | I | 1 | 50/50 | 100% | 92% |
| lift_default_block_iter300_gain0.5 | VI | 1 | 50/50 | 100% | 86% |
| lift_default_block_iter300_gain2 | VI | 1 | 50/50 | 100% | 100% |
| can_pandagrip_block_iter300 | I | 10 | 50/50 | 80% | 20% |
| can_default_block_iter300 | I | 10 | 50/50 | 72% | 8% |
| stack_pandagrip_block_iter300 | I | 1 | 50/50 | 96% | 0% |
| stack_default_block_iter300 | I | 10 | 50/50 | 90% | 30% |
| square_pandagrip_block_iter300 | I | 10 | 50/50 | 26% | 4% |
| square_default_block_iter300 | I | 10 | 50/50 | 2% | 0% |
| lift_pandagrip_delta_iter1 | VIII | 1 | 50/50 | 96% | 94% |
| lift_pandagrip_block_iter1 | VIII | 1 | 50/50 | 100% | 92% |
| lift_default_delta_iter1 | VIII | 1 | 50/50 | 82% | 48% |
| lift_default_block_iter1 | VIII | 1 | 50/50 | 100% | 86% |
| can_pandagrip_delta_iter1 | VIII | 1 | 50/50 | 74% | 60% |
| can_pandagrip_block_iter1 | VIII | 1 | 50/50 | 94% | 56% |
| can_default_delta_iter1 | VIII | 1 | 50/50 | 28% | 24% |
| can_default_block_iter1 | VIII | 1 | 50/50 | 86% | 50% |
| stack_pandagrip_delta_iter1 | VIII | 1 | 50/50 | 50% | 44% |
| stack_pandagrip_block_iter1 | VIII | 1 | 50/50 | 92% | 12% |
| stack_default_delta_iter1 | VIII | 1 | 50/50 | 42% | 46% |
| stack_default_block_iter1 | VIII | 1 | 50/50 | 88% | 80% |
| square_pandagrip_delta_iter1 | VIII | 1 | 50/50 | 6% | 4% |
| square_pandagrip_block_iter1 | VIII | 1 | 50/50 | 18% | 6% |
| square_default_delta_iter1 | VIII | 1 | 50/50 | 0% | 0% |
| square_default_block_iter1 | VIII | 1 | 50/50 | 24% | 14% |
| two_piece_pandagrip_delta_iter1 | VIII | 1 | 50/50 | 50% | 42% |
| two_piece_pandagrip_block_iter1 | VIII | 1 | 50/50 | 90% | 18% |
| two_piece_default_delta_iter1 | VIII | 1 | 50/50 | 30% | 36% |
| two_piece_default_block_iter1 | VIII | 1 | 50/50 | 98% | 34% |
| two_piece_pandagrip_block_iter300 | I | 10 | 50/50 | 96% | 20% |
| two_piece_default_block_iter300 | I | 10 | 50/50 | 98% | 24% |
