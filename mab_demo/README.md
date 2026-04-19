# 金融场景 MAB 多臂老虎机 Demo

复现《金融场景MAB技术应用设计文档.md》中"王女士 case"的全链路 MAB 算法栈，纯 Python 实现，无第三方依赖，一键运行。

## 一、运行

```bash
cd mab_demo
python3 run_demo.py
```

预期输出包含 4 个阶段的逐步推演 + 1 份"原始画像 vs MAB 增强画像"对比报告。

## 二、Demo 解决了什么问题

王女士在 APP 中输入"推荐一款教育理财产品"。但她的：
- **历史画像**：R3 中等风险，6 个月持有周期 → 与"教育金=15年R1"矛盾
- **当日行为**：点过 5 万的教育年金险又秒退、点过 1 年期理财又秒退、划过股票基金不看 → 产生大量噪音与漏掉的负样本

传统 SQL 过滤推荐系统在这种"画像-意图冲突"下，命中真实需求的概率几乎为 0。

本 demo 用 6 种 MAB 算法 4 阶段串联，让画像 5 秒内从 R3 短期 → R1·15年·2万/年 自动收敛。

## 三、6 种 MAB 算法对应的源码

| 阶段 | 算法 | 源码 |
|---|---|---|
| ① 入场洗数 | 因果去偏老虎机 | [`bandits/causal_debias.py`](bandits/causal_debias.py) |
| ① 入场洗数 | 隐式负反馈老虎机 | [`bandits/implicit_feedback.py`](bandits/implicit_feedback.py) |
| ② 零样本定调 | LLM 生成先验分布 | [`bandits/llm_prior.py`](bandits/llm_prior.py) |
| ② 零样本定调 | 非平稳衰减老虎机 | [`bandits/nonstationary.py`](bandits/nonstationary.py) |
| ③ 组合生臂 | 对话式组合老虎机 | [`bandits/slate_ccb.py`](bandits/slate_ccb.py) |
| ③ 组合生臂 | 大模型动态生臂 | [`bandits/generative_arm.py`](bandits/generative_arm.py) |
| ④ 闭环进化 | 贝叶斯后验坍缩 | [`orchestrator.stage_4_evolution`](orchestrator.py) |

## 四、画像对比效果（核心 KPI）

| 维度 | 原始画像 | MAB 增强后 |
|---|---|---|
| 画像置信度 | 0.35 | **0.97** |
| 意图标签 | 短期理财 | **子女教育金·15年定投·R1保本** |
| R1 风险概率 | 0.05 | **0.93** |
| R3 风险概率 | 0.60 | **0.00** |
| 年预算 σ | 20,000 | **1,455 (缩 13.7x)** |
| **目标产品命中率** | **0.00%** | **91.33%** |

## 五、源码导航

- 数据：[`data.py`](data.py) — 王女士原始画像、行为流、产品库、最终回复
- 编排：[`orchestrator.py`](orchestrator.py) — 四阶段流水线主线
- 对比：[`profile_compare.py`](profile_compare.py) — 原始 vs 增强画像差异展示
- 入口：[`run_demo.py`](run_demo.py) — 主程序

详细架构说明见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。
