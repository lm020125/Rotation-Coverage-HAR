# FMC-PC / RCBD-HAR 机制对照实验结果包（2026-09-22 拉取）

本目录是 AutoDL 服务器上**真实训练产物**的本地归档，取代此前所有手工演示数字。所有指标均由训练脚本直接产出，逐 run 绑定配置/profile/checkpoint 的 SHA-256，未做任何手工修改。

## 一、来源与完整性
- 服务器实例：`connect.cqa1.seetacloud.com:18134`，项目根 `/root/RCBD-HAR`
- REALDISP 结果目录：`/root/autodl-tmp/experiments/realdisp_mechanism_controls_20260920/`（24 个训练 run，含修复后 C2；完成标记 `COMPLETED`）
- WEAR 结果目录：`/root/autodl-tmp/experiments/wear_angle_only_20260920/`（fmc_pc × 2 种子 × 3 折 = 6 折）
- 总包文件：`../results_package_20260922.tar.gz`
- 总包 SHA-256：`e299da1ae14997f07e46860ecdbe4ff786bc5e25a6e29d6472d987cd0e733b94`（本地下载后已比对一致）
- 逐文件清单（相对路径/字节数/SHA-256，共 374 个文件）：`RESULTS_MANIFEST.csv`（UTF-8 BOM，Excel 可直接打开）

## 二、目录结构
```
results_package_20260922/
├─ realdisp_mechanism_controls_20260920/     # REALDISP 8 单元 × 3 种子
│   ├─ c1_uniform180_no_pc / c1_uniform180_pc
│   ├─ c2_beta_no_pc / c2_beta_pc            # 修复后逐肢体矩匹配（新）
│   ├─ c3_clip_no_pc / c3_clip_pc
│   ├─ c4_gt_no_pc / c4_gt_pc
│   ├─ archive_buggy_c2_20260922_100624/     # 旧 buggy C2 原始产物（留痕，勿用于结论）
│   ├─ summary.json / queue_status.json / runs.partial.json / COMPLETED
│   └─ <unit>/<seed>/{valid_signal_self, valid_signal_ideal, as_released_self}/metrics.json
│                                            # 每套含 confusion_matrix / per_class / per_participant
├─ wear_angle_only_20260920/
│   ├─ fmc_pc/seed_{3407,256}/fold_{1,2,3}/{metrics,history,split}.json
│   ├─ summary.json                          # 含逐折、OOF、逐参与者 macro-F1
│   └─ protocol_lock.json                    # config/folds/profile SHA + gain/noise 修正说明
├─ profiles_mechanism_controls/              # 冻结 profile 及 .audit.json（含 archive）
├─ ops/                                       # 链式守护脚本、日志、PID/LAUNCHED 标记
└─ logs_tail/                                 # 三个大训练日志的尾部 400 行（去 tqdm 回车）留证
```

## 三、REALDISP 机制对照结果（Macro-F1 %，均值 ± 总体标准差 ddof=0，种子 256/3407/114514）

有效信号 Self（主判读口径）
| 单元 | 无 PC | +PC |
|---|---|---|
| C1 U(0,180°) 均匀 | **89.96 ± 1.37** | 87.15 ± 1.25 |
| C2 Beta 逐肢体矩匹配（修复后） | **86.57 ± 3.10** | 86.54 ± 2.54 |
| C3 经验池截断 >90° | **79.23 ± 1.99** | 83.18 ± 2.88 |
| C4 重力匹配生成器 | 87.16 ± 2.31 | **88.17 ± 1.68** |

完整发布 Self（保留 subject 13 全零 Self 记录）
| 单元 | 无 PC | +PC |
|---|---|---|
| C1 | 76.83 ± 1.15 | 74.29 ± 0.94 |
| C2 | 73.70 ± 2.60 | 73.64 ± 2.37 |
| C3 | 66.84 ± 1.47 | 70.83 ± 2.48 |
| C4 | 74.45 ± 1.78 | 74.91 ± 1.53 |

有效信号 Ideal
| 单元 | 无 PC | +PC |
|---|---|---|
| C1 | 93.19 ± 1.24 | 91.04 ± 1.08 |
| C2 | 92.40 ± 0.05 | 92.33 ± 0.17 |
| C3 | 92.39 ± 1.22 | 90.44 ± 3.31 |
| C4 | 91.89 ± 1.48 | 92.10 ± 0.59 |

REALDISP 同台锚点（同管线）：无增强 RCBD 72.75 ± 0.88；U(0,90°) 无 PC 78.89、+PC 75.84；经验幅度无 PC 87.68；角度版 FMC-PC（主结果）88.89 ± 2.25。

判读：C1 普通 U(180°) 即 89.96，追平/略超经验主结果 → 经验频率校准非必要；C2 修复后 86.57，与经验无 PC 87.68 差 1.11 pp（±2 pp 内）→ 频率具体形状不重要；C3 截掉 >90° 尾部后跌到 79.23，落回 U(90°) 的 78.89 → 有效成分是 >90° 大角度尾部覆盖；C4 重力匹配 88.17 ≈ 主结果 → 与生成器/随机轴选择无关。

## 四、WEAR angle-only 结果（fmc_pc，关 gain/noise；三折 OOF 为混淆矩阵合并的 samplewise macro-F1）
| 种子 | fold1 | fold2 | fold3 | 三折 OOF |
|---|---|---|---|---|
| 3407 | 83.46 | 85.12 | 89.28 | **85.83%** |
| 256 | 85.79 | 86.40 | 88.15 | **86.65%** |
| **跨种子** | | | | **86.24 ± 0.58%** |

同台锚点（seed3407，archived，同 config/folds/profile）：Clean 78.62；U(0,180°)+PC 84.17；FMC-NoPC 85.75；FMC-PC-GT 85.98（三种子主结果 86.29 ± 0.34）。
判读：angle-only 86.24 ≈ 冻结 86.29（差 0.05 pp），gain/noise 几乎零贡献；高于 U(180°)+PC 门槛 84.17，跨数据集迁移成立。逐参与者结果（76.1–96.6）见各折 `metrics.json` 与 `summary.json`。

## 五、关键 SHA-256
- 经验冻结 profile `placement_profiles_subjects1_10.json`：`b05eedb46aa01cff0c2331b09bc3ca4911821a0ef7c8291c0807dd19c555b2b7`
- 修复后逐肢体 Beta profile `placement_profiles_beta_moment_matched.json`：`89de4e69d5750ad1fdf37c044644248c55b31a65c265819af5619224bb1c14d0`
- WEAR config `wear_rcbd.yaml`：`2f7c95e5bed8d3ff4553dd49a07ca0c83a6c7e148aea46236eeeb0205aed1985`
- WEAR folds `wear_paper_folds.json`：`adf1243ab5353b8cff6d81bba8e0bcfe3671a9729035ef987a59c912388ee611`

## 六、未包含内容（需要时可另行拉取）
- 模型权重 `best.pt`（43 个，约 0.3 GB）：结论可由代码 + 公开数据 + 本包配置复现，故未纳入；如需审计三重 SHA 绑定可再从服务器拉取。
- 原始数据集（REALDISP / WEAR）与预处理中间产物。
- 完整 tqdm 训练日志（`wear_queue.log` 约 179 MB、`mechanism_controls.log` 约 23 MB）；本包仅存尾部 400 行于 `logs_tail/`，每 epoch 指标以各 run `history.json` 为准。

## 七、自行核对
1. 比对总包 SHA 与第一节是否一致。
2. REALDISP：读取 `realdisp_.../summary.json` 与各 `<unit>/<seed>/<场景>/metrics.json`（含混淆矩阵、逐 33 类、逐参与者）。
3. WEAR：读取 `wear_angle_only_20260920/summary.json`（逐折/OOF/逐参与者）与 `protocol_lock.json`（哈希与 gain/noise 修正记录）。
4. 任何文件的 SHA 见 `RESULTS_MANIFEST.csv`。
