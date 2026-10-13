# day1_matrix 결과 요약  (4/34 cells with results; target=UR5e)

### Table I — state-based transfer (blocking, num_iter_max=300)

F/D = Franka(Panda) gripper / target default gripper. 논문 열은 Table I 그대로. *ours source = source.txt 의 Success_Rate — passive 모드에선 타깃이 성공하면 소스 루프가 먼저 끊겨 과소집계됨(참고용). 논문 source 열은 별도 source-only 실행값.

| Task | Source (Franka, paper) | Source (Panda, ours)* | UR5e (paper) F/D | IIWA (paper) F/D | Kinova Gen3 (paper) F/D | Sawyer (paper) F/D | Jaco (paper) F/D | **UR5e F/D (ours)** |
|---|---|---|---|---|---|---|---|---|
| Lift | 100% | 50% | 99/100 | 100/100 | 100/100 | 100/100 | 100/99 | — / 100% |
| Stack | 95% | — | 96/86 | 94/87 | 96/81 | 94/85 | 96/66 | — / — |
| Can | 98% | — | 98/88 | 97/90 | 97/83 | 96/83 | 96/55 | — / — |
| Two Piece Assembly | 96% | — | 91/89 | 89/70 | 88/92 | 92/90 | 90/34 | — / — |
| Square | 81% | — | 74/69 | 72/48 | 34/49 | 83/75 | 21/1 | — / — |

### Table VIII — delta actions (target executes the same delta action)

논문 Table VIII 과 같은 구조. 우리 셀 = `*_delta_iter1`.

| Task | Source (Franka, paper) | Source (Panda, ours)* | UR5e (paper) F/D | IIWA (paper) F/D | Kinova Gen3 (paper) F/D | Sawyer (paper) F/D | Jaco (paper) F/D | **UR5e F/D (ours)** |
|---|---|---|---|---|---|---|---|---|
| Lift | 100% | 50% | 91/99 | 94/53 | 96/98 | 98/72 | 96/63 | — / 50% |
| Stack | 95% | — | 36/41 | 39/28 | 74/43 | 60/40 | 36/38 | — / — |
| Can | 98% | — | 87/90 | 93/25 | 89/73 | 78/41 | 86/60 | — / — |
| Two Piece Assembly | 96% | — | 44/44 | 65/45 | 54/48 | 43/29 | 41/15 | — / — |
| Square | 81% | — | 10/6 | 15/1 | 3/2 | 5/5 | 9/0 | — / — |

### Ablation — num_iter_max 1 vs 300 (absolute pose, non-blocking vs blocking)

논문 대응 표 없음. delta 모드는 num_iter_max 를 쓰지 않으므로 delta_iter300 셀은 생성하지 않았다.

| Task | UR5e block_iter1 F/D | UR5e block_iter300 F/D (=Table I) | UR5e delta F/D (=Table VIII) |
|---|---|---|---|
| Lift | — / — | — / 100% | — / 50% |
| Stack | — / — | — / — | — / — |
| Can | — / — | — / — | — / — |
| Two Piece Assembly | — / — | — / — | — / — |
| Square | — / — | — / — | — / — |

### Table VI — sensitivity (control gain / proprio z-offset)

논문 Table VI 는 실로봇 Tiger 과제 수치라 직접 비교가 아니라 경향 비교용. 우리 셀은 Lift/default/blocking300 기준.

| Component | Factor | Variation | Paper (Tiger pick-place, real robot) | **UR5e Lift, default gripper (ours)** |
|---|---|---|---|---|
| Control | (baseline) | — | — | 100% |
| Control | Gain | x0.5 | 90% | 100% |
| Control | Gain | x2 | 60% | — |
| Control | Offset | 2 cm | 70% | 100% |
| Control | Offset | 5 cm | 20% | — |

### 셀 상태

| cell | table | done/n | target SR | source SR |
|---|---|---|---|---|
| lift_pandagrip_block_iter300 | I | 0/2 | — | — |
| lift_pandagrip_delta_iter1 | VIII | 0/2 | — | — |
| lift_pandagrip_block_iter1 | VIII | 0/2 | — | — |
| lift_default_block_iter300 | I | 2/2 | 100% | 50% |
| lift_default_delta_iter1 | VIII | 2/2 | 50% | 50% |
| lift_default_block_iter1 | VIII | 0/2 | — | — |
| lift_default_block_iter300_zoff2cm | VI | 2/2 | 100% | 0% |
| lift_default_block_iter300_zoff5cm | VI | 0/2 | — | — |
| lift_default_block_iter300_gain0.5 | VI | 2/2 | 100% | 50% |
| lift_default_block_iter300_gain2 | VI | 0/2 | — | — |
| can_pandagrip_block_iter300 | I | 0/2 | — | — |
| can_pandagrip_delta_iter1 | VIII | 0/2 | — | — |
| can_pandagrip_block_iter1 | VIII | 0/2 | — | — |
| can_default_block_iter300 | I | 0/2 | — | — |
| can_default_delta_iter1 | VIII | 0/2 | — | — |
| can_default_block_iter1 | VIII | 0/2 | — | — |
| stack_pandagrip_block_iter300 | I | 0/2 | — | — |
| stack_pandagrip_delta_iter1 | VIII | 0/2 | — | — |
| stack_pandagrip_block_iter1 | VIII | 0/2 | — | — |
| stack_default_block_iter300 | I | 0/2 | — | — |
| stack_default_delta_iter1 | VIII | 0/2 | — | — |
| stack_default_block_iter1 | VIII | 0/2 | — | — |
| square_pandagrip_block_iter300 | I | 0/2 | — | — |
| square_pandagrip_delta_iter1 | VIII | 0/2 | — | — |
| square_pandagrip_block_iter1 | VIII | 0/2 | — | — |
| square_default_block_iter300 | I | 0/2 | — | — |
| square_default_delta_iter1 | VIII | 0/2 | — | — |
| square_default_block_iter1 | VIII | 0/2 | — | — |
| two_piece_pandagrip_block_iter300 | I | 0/2 | — | — |
| two_piece_pandagrip_delta_iter1 | VIII | 0/2 | — | — |
| two_piece_pandagrip_block_iter1 | VIII | 0/2 | — | — |
| two_piece_default_block_iter300 | I | 0/2 | — | — |
| two_piece_default_delta_iter1 | VIII | 0/2 | — | — |
| two_piece_default_block_iter1 | VIII | 0/2 | — | — |
