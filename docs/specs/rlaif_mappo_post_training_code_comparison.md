# Original versus isolated post-training code

| 对比项 | 原始实现 | 新 post-training 实现 |
| --- | --- | --- |
| 训练入口 | 原 `train_mappo_async.py` | 新 `train_mappo_post_training.py` |
| RLAIF 初始化 | 随机初始化 | MAPPO-Env base checkpoint |
| actor | 随机 | 加载父 actor |
| critic | 随机 | 加载父 critic |
| optimizer | 新建 | 加载模型后重新新建 |
| RM | assignment-only | assignment-only |
| 偏好使用方式 | learned reward | learned reward |
| 策略优化 | MAPPO | MAPPO |
| DPO | 无 | 无 |
| 原流水线 | Phase 0–9 | 保持不变 |
| 新流水线 | 不适用 | Phase 0–11 |
| 输出目录 | `results/formal` | `results/formal_post_training` |
| benchmark | 原始规模 | 900 rows |
| 主要比较 | 原始方法比较 | RLAIF-post vs env-continuation |

## Reuse boundary

The new trainer reuses the established actor/centralized-critic networks, asynchronous transition buffer, event and observation schemas, entity encoders, environment construction, scenario sampling, device resolution, and Reward Model wrapper. These are common infrastructure rather than changes to the original trainer.

The isolated implementation deliberately mirrors the existing MAPPO math: event-time discounting, TD residual and GAE/returns, probability ratio, clipped surrogate, entropy regularization, MSE critic loss, and actor/critic gradient clipping. Parent validation, optimizer reset, frozen-RM validation, reward composition, and lineage checkpoint schema are new post-training responsibilities. The original `training/mappo_trainer.py` is not modified or dispatched to the new trainer.

The only intentional semantic differences are initialization from a validated parent, fresh optimizer state after weight loading, assignment-only RM augmentation in the aligned child, and independent lineage/output schemas. There is no DPO policy update.
