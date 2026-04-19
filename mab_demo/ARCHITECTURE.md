# 金融场景 MAB Demo · 代码架构说明文档（工业化版本）

## 一、设计目标

把《金融场景MAB技术应用设计文档.md》中的"四阶段融合架构"完整落地为一个**可独立运行、可观测、可对比、支持真实 LLM 接入**的 Python 后端 demo。

具体目标：
1. 复现王女士 case 中"画像-意图冲突 + 兴趣漂移 + 隐式负反馈"三大典型噪声。
2. 串联 6 种 MAB 技术（因果去偏 / 隐式负反馈 / LLM 先验 / 非平稳衰减 / 对话式组合 / 大模型生臂），按业务流自然解决冲突。
3. 提供"原始画像 vs 增强画像"的量化对比，证明 MAB 的有效性（命中率从 0% 提升至 91%）。
4. **支持真实 LLM（DeepSeek/Qwen/vLLM/OpenAI）接入，全链路带 Schema 校验与优雅降级**。

## 二、目录结构

```
mab_demo/
├── config.py                     # 全局配置: 所有环境变量集中管理
├── schemas.py                    # LLM JSON 输出严格校验(防幻觉炸管道)
├── nlu.py                        # NLU 槽位抽取(LLM + 正则兜底)
├── data.py                       # 王女士画像/行为流/产品库/最终回复
├── clients/                      # 基础设施层
│   ├── __init__.py
│   ├── llm_client.py             # OpenAI 兼容客户端(retry/cache/timeout/降级)
│   └── reward_model.py           # 可插拔奖励模型(rule/llm_judge/external)
├── prompts/                      # Prompt 模板(版本化)
│   ├── llm_prior.py
│   ├── generative_arm.py
│   └── nlu.py
├── bandits/                      # 6 种 MAB 算法
│   ├── causal_debias.py          # 阶段一·算法1(规则 + 可选 LLM 兜底)
│   ├── implicit_feedback.py      # 阶段一·算法2
│   ├── llm_prior.py              # 阶段二·算法1(真 LLM + Mock 降级)
│   ├── nonstationary.py          # 阶段二·算法2
│   ├── slate_ccb.py              # 阶段三·算法1
│   └── generative_arm.py         # 阶段三·算法2(真 LLM + Mock 降级)
├── orchestrator.py               # 四阶段流水线 + 阶段四闭环进化(含 NLU)
├── profile_compare.py            # 原始 vs 增强画像对比
├── run_demo.py                   # 主入口
├── tests/
│   └── test_llm_path.py          # 伪造 LLM 响应, 验证真实路径可通
├── requirements.txt              # 可选依赖(openai)
├── ARCHITECTURE.md               # 本文档
└── README.md                     # 使用说明
```

## 三、工业化架构分层

```
┌──────────────── 业务编排层 ─────────────────┐
│  orchestrator.py  ·  run_pipeline()          │
└──────────┬──────────────────────────────────┘
           │ 调用
┌──────────▼──── 算法层 (6 种 MAB) ────────────┐
│  bandits/causal_debias       (规则为主)       │
│  bandits/implicit_feedback   (纯算法)         │
│  bandits/llm_prior           (LLM)            │
│  bandits/nonstationary       (纯算法)         │
│  bandits/slate_ccb           (纯算法)         │
│  bandits/generative_arm      (LLM + RM)       │
│  nlu.extract_slots           (LLM + 正则)     │
└──────────┬──────────────────────────────────┘
           │ 调用
┌──────────▼──── 基础设施层 ────────────────────┐
│  clients/llm_client     OpenAI 兼容客户端     │
│  clients/reward_model   可插拔 RM              │
│  schemas                JSON Schema 校验器     │
│  prompts                Prompt 模板(版本化)    │
│  config                 环境变量聚合           │
└─────────────────────────────────────────────┘
```

## 四、LLM 接入点与降级路径

| # | 模块 | Mock → LLM 改造位置 | 降级策略 |
|---|---|---|---|
| 1 | 阶段二·先验分布 | `bandits/llm_prior.py:_llm_reasoning_real` | 失败 → 常识推理 Mock |
| 2 | 阶段三·生臂 | `bandits/generative_arm.py:_generate_candidates_real` | 失败 → 固定 4 条模板 |
| 3 | 阶段三·RM 打分 | `clients/reward_model.py:get_reward_model` | 默认规则 RM；可切 LLM Judge / External RM |
| 4 | 阶段四·NLU 抽槽 | `nlu.extract_slots` | 失败 → 正则抽取 |
| 5 | 阶段一·因果分类 | `bandits/causal_debias._llm_classify_confounder` | 默认关闭；启用后失败 → 规则兜底 |

**降级原则**：任何一层 LLM 失败都不会让主流程断链。`CONFIG.fallback_on_fail=True`（默认）时自动兜底。

## 五、关键设计决策

### 5.1 Schema 严格校验

LLM 返回后必须经 [`schemas.py`](schemas.py) 验证：
- 字段完整性（缺必要字段直接拒绝）
- 类型正确性
- 数值在业务合理区间（如 `term_years ∈ [0.1, 40]`，`risk` 概率之和 ≈ 1）
- 非法输出 → `SchemaError` → 上层走降级路径

**业务价值**：一次 LLM 幻觉（比如 `term_years=999`）可能让后续所有维度方差炸掉。Schema 层是必不可少的护城河。

### 5.2 进程内缓存

`LLMClient._cache` 按 `sha256(prompt + model + temperature + system)` 做幂等缓存：
- 命中率估计：相同客户主诉下的先验调用约 40-60% 命中
- 生产环境建议替换为 Redis，TTL 设 1-6 小时

### 5.3 Prompt 版本化

每个 prompt 模板都有 `VERSION` 字段（`prompts/llm_prior.py:4` 等），运行时会打印：
```
Prompt 版本: v1.0.2026-04-19
```
生产必备：改 prompt 必改版本号，方便监控 A/B 测试命中率变化。

### 5.4 可插拔奖励模型

通过环境变量 `MAB_RM_TYPE` 切换：
```
rule_based    (默认)     规则匹配 + 长度奖励, <1ms
llm_judge     (离线评估)  LLM 当 Judge, ~500ms, 仅测试用
external_rm   (生产推荐)  专用 RM 推理服务, <10ms
```

### 5.5 重试与超时

`LLMClient.chat_json` 的重试逻辑：
- 指数退避：`0.5 * 2^attempt`，最多 2s
- 默认 2 次重试（可配 `MAB_LLM_RETRIES`）
- 区分 JSON 解析失败 与 网络失败，分别打日志

## 六、四阶段流水线（与设计文档对应）

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
   │ (固化)   │   后验    │ ⑧ NLU 抽槽 │     高情商追问       2 产品 + 1 追问
   └──────────┘           └────────────┘
```

## 七、运行方式

### 7.1 不接 LLM（零依赖）
```bash
cd mab_demo && python3 run_demo.py
```
所有 LLM 节点自动降级到 Mock，输出与前版完全一致。

### 7.2 接入 DeepSeek / Qwen / OpenAI
```bash
pip install -r requirements.txt
export MAB_LLM_API_KEY=sk-xxx
export MAB_LLM_BASE_URL=https://api.deepseek.com/v1   # 或其他
export MAB_LLM_MODEL=deepseek-chat
python3 run_demo.py
```
输出中会看到：
```
推理来源: LLM
生成来源: LLM
[NLU·LLM] 抽取槽位: {...}
```

### 7.3 接入本地 vLLM
```bash
# 先起 vLLM: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B
export MAB_LLM_API_KEY=dummy
export MAB_LLM_BASE_URL=http://localhost:8000/v1
export MAB_LLM_MODEL=Qwen/Qwen2.5-7B
```

### 7.4 测试 LLM 路径（无需真实 API key）
```bash
python3 tests/test_llm_path.py
```
用 FakeLLMClient monkey-patch，验证所有 LLM 调用点的代码路径 + Schema 校验。

## 八、可观测性

所有算法默认 `verbose=True`，每个阶段都打印：
- Prompt 版本号
- LLM 调用来源（LLM / Mock + 降级原因）
- 中间矩阵（惩罚表、先验分布、EIG 分数等）
- 决策依据（为什么选这个组合）

方便业务方做白盒验证，也便于生产出错时快速定位是哪一层降级了。

## 九、生产落地的剩余改造项

| 项目 | Demo 实现 | 生产建议 |
|---|---|---|
| LLM 客户端 | 单例 OpenAI SDK | + 连接池 + gRPC 双通道 + 负载均衡 |
| 缓存 | 进程内 dict | Redis Cluster with TTL |
| Schema 校验 | 手写 validator | Pydantic v2 + 自动代码生成 |
| 因果归因 | 规则 + LLM 兜底 | DML/双重机器学习, Causal Forest |
| RM 打分 | 规则 | 离线训练 DeBERTa-RM, 量化 INT8 部署 |
| 监控 | print + logging | OpenTelemetry + Prometheus |
| Prompt 管理 | 代码内常量 | PromptOps 平台(版本/灰度/回滚) |
| 降级开关 | 全局 flag | 多级熔断(Hystrix 模式) |

## 十、与原版 Demo 的差异

| 维度 | 原 Demo (v1) | 工业化版 (v2) |
|---|---|---|
| LLM 调用 | 纯 Mock | LLM + Mock 双路径 |
| JSON 校验 | 无 | `schemas.py` 严格校验 |
| Prompt | 硬编码 | 版本化模板 |
| RM | 规则内联 | 可插拔接口 |
| NLU | 硬编码结果 | LLM + 正则兜底 |
| 因果归因 | 3 条规则 | 规则 + 可选 LLM 兜底 |
| 可观测性 | 基础打印 | + Prompt 版本、调用来源、降级原因 |
| 测试 | 无 | `tests/test_llm_path.py` |
| 依赖 | 零 | 可选 openai (不装也能跑) |
