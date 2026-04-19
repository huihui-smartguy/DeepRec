# 金融场景 MAB Demo · 代码架构说明文档

## 一、设计目标

把《金融场景MAB技术应用设计文档.md》中的"四阶段融合架构"完整落地为一个**可独立运行、可观测、可对比**的 Python 后端 demo。

具体目标：
1. 复现王女士 case 中"画像-意图冲突 + 兴趣漂移 + 隐式负反馈"三大典型噪声。
2. 串联 6 种 MAB 技术（因果去偏 / 隐式负反馈 / LLM 先验 / 非平稳衰减 / 对话式组合 / 大模型生臂），按业务流自然解决冲突。
3. 提供"原始画像 vs 增强画像"的量化对比，证明 MAB 的有效性（命中率从 0% 提升至 91%）。

## 二、目录结构

```
mab_demo/
├── data.py                       # 王女士原始画像、行为流、产品库、最终回复
├── bandits/                      # 六种 MAB 算法
│   ├── __init__.py
│   ├── causal_debias.py          # 阶段一·算法1: 因果去偏
│   ├── implicit_feedback.py      # 阶段一·算法2: 隐式负反馈
│   ├── llm_prior.py              # 阶段二·算法1: LLM 生成先验分布
│   ├── nonstationary.py          # 阶段二·算法2: 非平稳衰减
│   ├── slate_ccb.py              # 阶段三·算法1: 对话式组合 MAB
│   └── generative_arm.py         # 阶段三·算法2: 大模型动态生臂
├── orchestrator.py               # 四阶段流水线编排 + 阶段四闭环进化
├── profile_compare.py            # 原始画像 vs 增强画像 对比报告
├── run_demo.py                   # 主入口
├── ARCHITECTURE.md               # 本文档
└── README.md                     # 使用说明
```

## 三、四阶段流水线（与设计文档对应）

```
┌─────────┐  阶段一: 入场洗数  ┌────────────┐  阶段二: 零样本定调  ┌──────────┐
│ 原始    │ ───────────────▶  │ 干净行为流 │ ──────────────▶     │ 注入先验 │
│ 行为流  │  ① 因果去偏      │ + 惩罚矩阵 │  ③ LLM Prior      │ 的 MAB   │
│ + 静态  │  ② 隐式负反馈    │            │  ④ 非平稳衰减     │ 状态空间 │
│ 画像    │                   │            │                     │          │
└─────────┘                    └────────────┘                     └──────────┘
                                                                       │
       阶段四: 闭环进化         阶段三: 组合生臂                       │
   ┌──────────┐  ⑦ 贝叶斯  ┌────────────┐                              │
   │ 增强画像 │ ◀────── ── │ 用户回复   │ ◀── ⑥ 大模型生臂 ── ⑤ Slate-CCB
   │ (固化)   │   后验    │ + 点击     │     高情商追问       2 产品 + 1 追问
   └──────────┘           └────────────┘
```

## 四、模块职责与算法实现要点

### 阶段一·算法1：`bandits/causal_debias.py` 因果去偏 MAB

- **输入**：6 条原始行为事件
- **算法**：基于因果规则集（`source_module`、`action`、`note` 等元数据）识别"看似有意图、实则是路径噪音"的混淆变量（Confounder）
- **关键规则**：
  - `source_module == "我的持仓"` → 持仓回看，非购买意图
  - `底部导航 + click_tab` → 路径噪音
  - `note 中含"随便看看"` → 浏览惯性
- **输出**：`(clean_events, confounders)`，将 6 条 → 3 条真因果事件

### 阶段一·算法2：`bandits/implicit_feedback.py` 隐式负反馈 MAB

- **输入**：清洗后的真因果事件
- **算法**：
  - 检测 `dwell_seconds < 5` 或 `quick_exit=True` → 秒退（reward = -1.0）
  - 检测 `silent_ignore=True` → 主动划过（reward = -0.8，扣到 risk 维度）
  - **因果归因**：秒退的根因往往集中在某一维度。用"教育金常识基线"反查偏离最大的维度（min_invest 或 term）作为主责，主责扣 -1.0，次责扣 -0.2。
- **输出**：`penalty[dim][bucket] = 累计 reward`
- **关键设计**：不直接扣 risk 维度（risk 由阶段二的 LLM 先验和非平稳处理）

### 阶段二·算法1：`bandits/llm_prior.py` LLM 生成先验分布

- **输入**：静态画像 + 主诉文本
- **算法（mock 的 LLM 推理）**：
  - 从家庭结构推理出周期：5 岁子女 → 17 年到大学（取 15 年作为定投基金主流周期）
  - 从年收入推理预算：60 万 × 3-5% = 1.8-3 万/年
  - 从教育金常识推理风险偏好：本金安全 > 高收益 → R1 主导
- **输出**：高斯/分类先验
  ```python
  prior["term_years"]    = N(μ=15, σ=2)
  prior["annual_budget"] = N(μ=24000, σ=6000)
  prior["risk"]          = {R1:0.55, R2:0.35, R3:0.08, R4:0.02}
  ```
- **价值**：MAB 跳过冷启动盲盒试错，第一轮就锁定优质区域

### 阶段二·算法2：`bandits/nonstationary.py` 非平稳衰减 MAB

- **输入**：静态画像 + LLM 先验 + 隐式负反馈惩罚
- **算法**：
  1. 计算 `KL(LLM_prior || historical)` 作为概念漂移强度
  2. 若 KL > 阈值 0.4，触发衰减
  3. 用 `α = sigmoid(1.5 * (KL - 0.4))` 生成历史/先验融合权重
  4. `fused[k] = (1-α) * historical[k] + α * llm_prior[k]`
  5. 叠加显式负反馈：若该 risk 等级被主动划过（penalty ≤ -0.5），额外乘 0.05 二次压制
- **输出**：风险维度后验分布
- **效果**：王女士的 KL = 1.408 → α = 0.819 → R3 从 0.60 → 0.18，R1 从 0.05 → 0.48

### 阶段三·算法1：`bandits/slate_ccb.py` 对话式组合 MAB (Slate-CCB)

- **输入**：完整产品库 + 当前 MAB 状态（先验 + 衰减后风险）+ 隐式负反馈惩罚
- **算法**：
  1. **Step 1 隐式负反馈硬过滤**：剔除被打到 -0.8 以下的金额段、期限段
  2. **Step 2 质量底线过滤**：`base_quality(p) = term_match × risk_match ≥ 0.18`，确保两款产品都满足"教育金 must-have"
  3. **Step 3 ask 维度选择**：在 budget/term/risk 中选不确定性最大的维度作为追问目标
  4. **Step 4 组合臂枚举**：对每个 (a, b) 配对计算
     ```
     score = 0.55 · avg_quality + 0.30 · budget_coverage + 0.15 · straddle
     ```
     其中 `straddle` 奖励"一上一下跨越 budget 均值"的组合（最大化预算维度上的二分查找信息增益）
- **输出**：`{exploit, explore, ask_dim, eig, score}`
- **效果**：选出 P006 (1.5万) [利用臂] + P007 (3万) [探索臂] + ask_dim=budget，与设计文档完全一致

### 阶段三·算法2：`bandits/generative_arm.py` 大模型动态生臂 (Generative Arms)

- **输入**：Slate-CCB 输出的硬核机器指令
- **算法**：
  1. LLM 实时生成 4 个候选追问臂（mock 为固定 4 句话，对应不同语气策略）
  2. 每个臂用三个子分打分：
     - `empathy`：是否包含痛点关键词（王女士、宝宝、品质生活、节奏…）
     - `precision`：是否同时锚定 A、B 两个具体金额
     - `conciseness`：长度奖励（100-180 字最佳）
  3. `total = 0.5 × empathy + 0.3 × precision + 0.2 × conciseness`
- **输出**：得分最高的追问话术
- **效果**：选中 ARM_3 «高情商 + 锚点 + 痛点回应» (total=0.917)

### 阶段四：`orchestrator.stage_4_evolution` 闭环进化

- **输入**：客户的真实回复（点击 P006 / 忽略 P007 / 文本"差不多每年 2 万吧, 别亏钱"）
- **算法**：
  1. **annual_budget 维度**：高斯-高斯共轭后验
     ```
     posterior_var = 1 / (1/prior_var + 1/obs_var)
     posterior_mu  = posterior_var · (prior_mu/prior_var + obs_mu/obs_var)
     ```
  2. **term 维度**：客户点击 P006(15年) → 强确认 → σ 直接坍缩到 0.5
  3. **risk 维度**：点 R1 / 忽略 R2 → R1 概率压倒性提升
- **输出**：固化的增强画像 `UserProfile`，置信度 0.97

## 五、原始画像 vs 增强画像 对比器

`profile_compare.py` 提供量化对比：

| 维度 | 原始画像 | 增强后 | 增益 |
|---|---|---|---|
| 画像置信度 | 0.35 | 0.97 | +62% |
| 意图标签 | 短期理财 | 子女教育金·15年定投·R1保本 | ✓ 已识别 |
| 风险 R1 | 0.05 | 0.93 | 主导反转 |
| 风险 R3 | 0.60 | 0.00 | 旧主导坍缩 |
| 期限 μ (年) | 0.5 | 15.0 | 跨越 30 倍 |
| 年预算 σ (元) | 20,000 | 1,455 | 缩 13.7x |
| **目标产品命中率** | **0.00%** | **91.33%** | **从不可能到必然** |

## 六、技术栈与运行环境

- **语言**：Python 3.7+
- **依赖**：纯标准库（`math`、`itertools`、`collections`、`dataclasses`），无第三方依赖
- **运行**：`cd mab_demo && python3 run_demo.py`
- **可观测性**：所有算法默认 `verbose=True`，每个阶段都打印中间状态、矩阵、决策依据，方便业务方做白盒验证

## 七、与生产环境的差异

为了 demo 简洁，以下环节做了简化：

| 模块 | Demo 简化 | 生产实现建议 |
|---|---|---|
| LLM 推理 | 硬编码确定性 mock | DeepSeek-V3 / Qwen-Max + vLLM, 输出 JSON 先验 |
| 因果归因 | 规则集 | DML/双重机器学习 / Causal Forest |
| 隐式负反馈检测 | dwell + flag | Flink 实时窗口 + Bayesian Surprise |
| Slate 枚举 | O(n²) 暴力 | LinUCB / NeuralUCB + 分层树 Bandit |
| 生臂打分 | 关键词匹配 | Reward Model (RLHF 训练的小模型) |
| 后验更新 | 解析高斯共轭 | NUTS 采样 / Variational Inference |

## 八、可扩展点

1. **新增 case**：在 `data.py` 增加新的 `UserProfile` 与 `BEHAVIOR_LOG`，主流程无需修改
2. **新增 MAB 算法**：在 `bandits/` 下新增模块，在 `orchestrator.run_pipeline` 中插入调用
3. **接入真实 LLM**：替换 `bandits/llm_prior.py` 与 `bandits/generative_arm.py` 中的 mock 函数
4. **多用户对比**：批量跑 `run_pipeline` 输出聚合命中率指标
