# MAB 算法实现详解

本文档逐一解析 `mab_demo` 中 7 个 MAB / 强化学习相关模块的代码实现，配合源码行号便于跳读。

四阶段流水线与算法的对应关系：

| 阶段 | 算法 | 源码 |
|---|---|---|
| ① 入场洗数 | 因果去偏老虎机 | [bandits/causal_debias.py](mab_demo/bandits/causal_debias.py) |
| ① 入场洗数 | 隐式负反馈老虎机 | [bandits/implicit_feedback.py](mab_demo/bandits/implicit_feedback.py) |
| ② 零样本定调 | LLM 生成先验分布 | [bandits/llm_prior.py](mab_demo/bandits/llm_prior.py) |
| ② 零样本定调 | 非平稳衰减老虎机 | [bandits/nonstationary.py](mab_demo/bandits/nonstationary.py) |
| ③ 组合生臂 | 对话式组合老虎机 (Slate-CCB) | [bandits/slate_ccb.py](mab_demo/bandits/slate_ccb.py) |
| ③ 组合生臂 | 大模型动态生臂 | [bandits/generative_arm.py](mab_demo/bandits/generative_arm.py) |
| ④ 闭环进化 | 贝叶斯后验坍缩 | [orchestrator.py](mab_demo/orchestrator.py) `stage_4_evolution` |

---

## 1. 因果去偏老虎机 (Causal Debiasing MAB)

**目的**：把"Position Bias（路径噪音）"和"History Bias（浏览惯性）"从行为流中剥离，防止 MAB 把这些**非意图**事件当作奖励信号去更新臂的估计值。

**源码**：[bandits/causal_debias.py](mab_demo/bandits/causal_debias.py)

### 1.1 双通道设计（规则 + 可选 LLM）

工业化版本走 **规则引擎主路径 + LLM 兜底副路径** 的双通道设计：

```
事件进入 ──► 规则引擎 ──► 命中规则? ──是──► 打 Confounder 标签
                              │
                             否
                              │
                              ▼
                    CONFIG.causal_use_llm? ──否──► 默认视为 Causal
                              │
                             是
                              │
                              ▼
                  LLM 二分类 ──► Schema 校验 ──► 标签
```

环境变量开关：`MAB_CAUSAL_USE_LLM`（默认 `0`），**生产慎开**——因为行为流每秒可能上千条，LLM 调用成本与延迟会失控。

### 1.2 Mock 版本：规则引擎（主路径）

源码 [causal_debias.py:18-28](mab_demo/bandits/causal_debias.py) 定义 3 条硬规则：

```python
CAUSAL_RULES = [
    (lambda e: e.get("source_module") == "我的持仓",
     "Confounder",
     "来自'我的持仓'页面 → 查看自有资产, 与新购买无因果链路"),
    (lambda e: e.get("source_module") == "底部导航" and e.get("action") == "click_tab",
     "Confounder",
     "底部 tab 进入 → 路径噪音, 不能据此推断品类偏好"),
    (lambda e: e.get("note", "").startswith("随便看看"),
     "Confounder",
     "停留时长虽长但无后续动作 → 浏览惯性, 切断与意图的关联"),
]
```

每条规则是一个 `(条件函数, 标签, 归因说明)` 三元组。主循环 [causal_debias.py:60-72](mab_demo/bandits/causal_debias.py) 按顺序匹配，首个命中即出结果：

```python
for ev in behavior_log:
    label, reason = "Causal", "通过因果安检"
    for cond, lab, why in CAUSAL_RULES:
        if cond(ev):
            label, reason = lab, why
            break
```

**特点**：

- **延迟 < 5 ms**，纯内存匹配
- **可解释**：每条事件都带 `_causal_reason` 字段
- **硬编码知识上限**：需要运营同学持续更新规则库

在王女士 case 中，这 3 条规则精准识别了：
- 查看"我的持仓股票基金"事件 → 非购买意图
- 点击底部"理财"tab → 路径噪音
- 带"随便看看"note 的浏览事件 → 浏览惯性

### 1.3 LLM 真实版本：调用链

源码 [causal_debias.py:31-50](mab_demo/bandits/causal_debias.py)：

```python
def _llm_classify_confounder(event: Dict[str, Any]) -> Tuple[bool, str]:
    from clients.llm_client import llm_client
    from schemas import validate_causal_label
    if not llm_client.enabled:
        return False, "LLM 不可用, 视为正常"
    prompt = (
        f"判断以下用户行为事件是否为'混淆变量'(即非真实购买意图的路径噪音):\n"
        f"事件: {event}\n\n"
        f"仅返回 JSON: {{\"is_confounder\": bool, \"reason\": \"<简短说明>\"}}"
    )
    raw = llm_client.chat_json(prompt=prompt, temperature=0.0)
    if raw is None:
        return False, "LLM 调用失败, 默认视为因果"
    try:
        out = validate_causal_label(raw)
        return out["is_confounder"], out["reason"]
    except Exception as e:
        log.warning(f"Causal LLM output invalid: {e}")
        return False, "Schema 错误, 默认视为因果"
```

**关键实现要点**：

| 要点 | 代码表现 | 为什么 |
|---|---|---|
| `temperature=0.0` | LLM 调用参数 | 二分类任务应完全确定性，避免同一事件不同结果 |
| 保守降级 | 失败全部当作 `Causal` | 宁可保留少量噪音，也不要把真意图误杀 |
| Schema 校验 | `validate_causal_label(raw)` | 防止 LLM 返回 `{"is_confounder": "maybe"}` 之类幻觉 |
| 触发条件 | `CONFIG.causal_use_llm and ev.get("note")` | 只对规则未命中且有歧义信号的事件调 LLM，控制成本 |

主流程中 LLM 兜底的挂载位置 [causal_debias.py:69-72](mab_demo/bandits/causal_debias.py)：

```python
# 规则未命中, 且启用了 LLM 兜底, 且事件有 ambiguity 信号 → 调 LLM
if label == "Causal" and CONFIG.causal_use_llm and ev.get("note"):
    is_conf, llm_reason = _llm_classify_confounder(ev)
    if is_conf:
        label, reason = "Confounder", f"LLM 兜底: {llm_reason}"
```

### 1.4 生产路线图

| 实现层级 | Demo | 生产推荐 |
|---|---|---|
| 规则引擎 | 3 条硬规则 | 数百条规则 + 决策树 |
| 因果推断 | LLM 二分类 | DML / Causal Forest / T-Learner |
| 特征输入 | 事件 JSON | + 用户历史 + 上下文 Embedding |

---

## 2. 隐式负反馈老虎机 (Implicit Negative Feedback MAB)

**目的**：把"秒退"和"主动划过"这种隐式负信号翻译成**显式负奖励**，写入特征级**惩罚矩阵**，供下游硬过滤与后验推理使用。

**源码**：[bandits/implicit_feedback.py](mab_demo/bandits/implicit_feedback.py)

### 2.1 两类负信号，两种处理策略

| 信号 | 含义 | 处理维度 | 惩罚强度 |
|---|---|---|---|
| `quick_exit` / `dwell_seconds < 5` | 点进去秒退 | `min_invest` 或 `term`（因果归因到主责） | -1.0（主责）/ -0.2（次责） |
| `silent_ignore` | 卡片划过不点 | `risk`（显式拒绝风险等级） | -0.8 |

常量定义 [implicit_feedback.py:15-17](mab_demo/bandits/implicit_feedback.py)：

```python
QUICK_EXIT_THRESHOLD = 5
QUICK_EXIT_PENALTY = -1.0
SWIPE_PENALTY = -0.8
```

### 2.2 因果归因：解决"维度污染"

**这是本模块的核心创新点**。传统做法：秒退了就把商品所有维度都 -1。

问题：比如一款「5 万起投 + R1 + 15 年」的产品，用户嫌 5 万起投太高秒退了，但传统做法把 R1（其实是好风险等级）也打了 -1，结果下游 Slate-CCB 把候选池清空。

解决方案 [implicit_feedback.py:41-45](mab_demo/bandits/implicit_feedback.py)：**用业务基线反查偏离最大的维度作为主责**。

```python
EDU_BASELINE = {"min_invest": 20_000, "term_years": 15}

def _attribute_root_cause(attrs: Dict[str, Any]) -> str:
    """归因到偏离基线最大的那个维度。"""
    invest_dev = abs(attrs["min_invest"] - EDU_BASELINE["min_invest"]) / EDU_BASELINE["min_invest"]
    term_dev = abs(attrs["term_years"] - EDU_BASELINE["term_years"]) / EDU_BASELINE["term_years"]
    return "min_invest" if invest_dev >= term_dev else "term"
```

对"5 万起投 / 15 年 / R1"这款秒退的教育年金险：
- `invest_dev = |50000-20000|/20000 = 1.5`
- `term_dev = |15-15|/15 = 0`
- **主责 = min_invest**，只扣 min_invest 分桶，term 维度轻微连坐，risk 维度完全不碰

### 2.3 分桶离散化

为了让惩罚能泛化到"同类"产品而不仅仅是单个 SKU，对金额与期限做分桶 [implicit_feedback.py:23-38](mab_demo/bandits/implicit_feedback.py)：

```python
def _bucket_invest(amount: int) -> str:
    if amount >= 50_000:  return "≥5万"
    if amount >= 10_000:  return "1-5万"
    return "<1万"

def _bucket_term(years: float) -> str:
    if years <= 1:   return "≤1年"
    if years <= 5:   return "1-5年"
    if years <= 10:  return "5-10年"
    return ">10年"
```

### 2.4 主循环

[implicit_feedback.py:52-82](mab_demo/bandits/implicit_feedback.py)：

```python
for ev in clean_events:
    attrs = ev.get("product_attrs")
    if not attrs:
        continue

    # 主动划过: 显式拒绝该风险等级
    if ev.get("silent_ignore"):
        penalty["risk"][attrs["risk"]] += SWIPE_PENALTY
        continue

    # 秒退: 因果归因到主责维度
    if ev.get("quick_exit") or (ev.get("dwell_seconds", 999) < QUICK_EXIT_THRESHOLD):
        root = _attribute_root_cause(attrs)
        invest_b = _bucket_invest(attrs["min_invest"])
        term_b = _bucket_term(attrs["term_years"])

        if root == "min_invest":
            penalty["min_invest"][invest_b] += QUICK_EXIT_PENALTY      # -1.0
            penalty["term"][term_b]         += QUICK_EXIT_PENALTY * 0.2 # -0.2
        else:
            penalty["term"][term_b]         += QUICK_EXIT_PENALTY       # -1.0
            penalty["min_invest"][invest_b] += QUICK_EXIT_PENALTY * 0.2 # -0.2
```

### 2.5 输出结构

返回一个嵌套 dict，结构：

```python
penalty = {
    "min_invest": {"≥5万": -1.0, "<1万": -0.2},
    "term":       {"1-5年": -1.0, ">10年": -0.2},
    "risk":       {"R4": -0.8}
}
```

这个惩罚矩阵会被两个下游消费：
- **阶段二·非平稳衰减**：`risk` 维度的强扣会进入二次衰减（详见 §4）
- **阶段三·Slate-CCB**：`≤-0.8` 的分桶直接进入**硬黑名单**，从候选池剔除

### 2.6 为何不用 LLM

隐式负反馈是**纯数值操作**，LLM 帮不上忙，反而会引入延迟和不确定性。这也是设计上刻意保留为"纯算法"的原因。

---

## 3. LLM 生成先验分布 (LLM-Generated Priors)

**目的**：打破 MAB 冷启动盲盒。传统 Thompson Sampling / UCB 在零样本情况下对所有臂等概率采样，要几十上百轮才能收敛；LLM Prior 用**常识推理**直接给出一个高置信的先验分布，让 MAB 第一轮就锁定优质区域。

**源码**：[bandits/llm_prior.py](mab_demo/bandits/llm_prior.py)，Prompt 模板 [prompts/llm_prior.py](mab_demo/prompts/llm_prior.py)

### 3.1 输出数据结构

LLM 必须输出如下 schema [schemas.py](mab_demo/schemas.py):

```json
{
  "cot_trace": ["推理步骤1", "推理步骤2", ...],
  "priors": {
    "term_years":    {"mu": 15.0, "sigma": 2.0},
    "annual_budget": {"mu": 24000, "sigma": 6000},
    "risk": {"R1": 0.55, "R2": 0.35, "R3": 0.08, "R4": 0.02}
  }
}
```

- **`term_years` / `annual_budget`**：连续变量，高斯先验 `N(μ, σ)`
- **`risk`**：离散变量，Dirichlet 先验（用归一化概率表示）
- **`cot_trace`**：思维链，用于可观测性与离线 A/B

### 3.2 真实 LLM 路径

[llm_prior.py:22-42](mab_demo/bandits/llm_prior.py)：

```python
def _llm_reasoning_real(profile: UserProfile, query: str) -> Dict[str, Any]:
    prompt = PRIOR_PROMPT.format(
        name=profile.name, age=profile.age, gender=profile.gender,
        annual_income=profile.annual_income, family=profile.family,
        risk_level=profile.risk_level, holdings=profile.holdings,
        avg_holding_months=profile.avg_holding_months, query=query,
    )
    raw = llm_client.chat_json(
        prompt=prompt,
        system=PRIOR_SYSTEM,
        temperature=0.2,          # 先验生成偏确定性
    )
    if raw is None:
        raise RuntimeError("LLM client returned None")
    return validate_llm_prior(raw)   # Schema 严格校验
```

**关键参数**：
- `temperature=0.2`：低温度 → 同一用户同一主诉应得到**基本一致**的先验，不允许"今天推 R1 明天推 R3"的随机性
- `validate_llm_prior(raw)`：校验 `mu>0`、`sigma>0`、`risk` 概率之和 ≈ 1、`term_years ∈ [0.1, 40]` 等**业务合理性**，拒绝幻觉（如 `term_years=999`）

### 3.3 Mock 降级路径

[llm_prior.py:46-65](mab_demo/bandits/llm_prior.py)：

```python
def _llm_reasoning_mock(profile: UserProfile, query: str) -> Dict[str, Any]:
    trace = []
    trace.append(f"主诉解析: '{query}' → 场景=子女教育金规划")
    child_age = 5
    horizon = 22 - child_age
    trace.append(f"家庭结构: 5岁子女 → 至大学(22岁)需 {horizon} 年; 主流教育定投周期 15 年")
    income = profile.annual_income
    edu_lo, edu_hi = int(income * 0.03), int(income * 0.05)
    trace.append(f"收入水平: 年入{income/10000:.0f}万 → 教育定投通常占 3%-5%...")
    trace.append("教育金底线: 本金安全 > 高收益 → R1 主导, R2 次位")
    return {
        "cot_trace": trace,
        "priors": {
            "term_years":    {"mu": 15.0, "sigma": 2.0},
            "annual_budget": {"mu": 24_000, "sigma": 6_000},
            "risk": {"R1": 0.55, "R2": 0.35, "R3": 0.08, "R4": 0.02},
        },
    }
```

Mock 版本用人工编码的常识推理（"5 岁孩子 → 15 年"、"年收入 60w × 3-5% = 2-3w"）模拟 LLM 的输出。**两个版本输出 schema 完全一致**，下游无感知。

### 3.4 三态降级编排

[llm_prior.py:68-84](mab_demo/bandits/llm_prior.py)：

```python
def _llm_reasoning(profile, query, verbose=False):
    source = "LLM"
    if llm_client.enabled:
        try:
            out = _llm_reasoning_real(profile, query)
            return out, source
        except (SchemaError, Exception) as e:
            if not CONFIG.fallback_on_fail:
                raise
            source = f"Mock (LLM 降级: {type(e).__name__})"
    else:
        source = "Mock (LLM 未启用)"

    out = _llm_reasoning_mock(profile, query)
    return out, source
```

三种运行状态对用户侧完全透明：

| 状态 | source 标识 |
|---|---|
| 有 API key + LLM 调用成功 | `LLM` |
| 有 API key + LLM 调用失败 | `Mock (LLM 降级: RuntimeError)` |
| 无 API key | `Mock (LLM 未启用)` |

### 3.5 在王女士 case 中的效果

输入：38 岁、R3、0.5y 持有、60w 年入、5 岁子女 + 主诉"推荐一款教育理财产品"。
输出：

```
term_years    ~ N(μ=15.0,  σ=2.0)
annual_budget ~ N(μ=24000, σ=6000)
risk          ~ {R1:0.55, R2:0.35, R3:0.08, R4:0.02}
```

原画像 R3 概率 0.60，先验直接把 R1 拉到 0.55，为下游的非平稳衰减铺好路。

---

## 4. 非平稳衰减老虎机 (Non-stationary / Discounted MAB)

**目的**：检测"概念漂移"（本轮意图与历史画像严重冲突），一旦触发则**让 LLM 先验主导风险维度**，实现"秒级画像转身"。

**源码**：[bandits/nonstationary.py](mab_demo/bandits/nonstationary.py)

### 4.1 算法架构

```
┌──────────────────────────────────────────────────────────┐
│  历史 risk 分布 p = profile.risk_dist = {R3:0.60, ...}   │
│  LLM 先验    q = llm_prior["risk"]   = {R1:0.55, ...}   │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
          KL(q || p) 衡量"先验相对历史"的意外程度
                            │
                            ▼
        ┌──────── KL > DRIFT_THRESHOLD? ────────┐
        │                                       │
        否                                      是
        │                                       │
        ▼                                       ▼
   返回原画像                         α = sigmoid(1.5 * (KL - 0.4))
                                              │
                                              ▼
                    fused[k] = (1-α) * historical[k] + α * prior[k]
                                              │
                                              ▼
                              额外惩罚: penalty[risk][k] ≤ -0.5 → ×0.05
```

### 4.2 漂移强度：KL 散度

[nonstationary.py:19-26](mab_demo/bandits/nonstationary.py)：

```python
def _kl(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-6
    s = 0.0
    for k in set(p.keys()) | set(q.keys()):
        pk = p.get(k, eps)
        qk = q.get(k, eps)
        s += pk * math.log((pk + eps) / (qk + eps))
    return max(0.0, s)
```

**计算方向**：`KL(LLM_prior || historical)` 而非 `KL(historical || LLM_prior)`，语义：
> 如果我们按历史画像来理解世界，LLM 先验会让我们多付多少比特的"意外信息"？

对王女士 case：
- `historical = {R3: 0.60, R2: 0.25, R4: 0.10, R1: 0.05}`
- `prior      = {R1: 0.55, R2: 0.35, R3: 0.08, R4: 0.02}`
- `KL ≈ 1.41` ≫ `DRIFT_THRESHOLD = 0.4` → 触发衰减

### 4.3 融合权重：Sigmoid 映射

[nonstationary.py:41-42](mab_demo/bandits/nonstationary.py)：

```python
alpha = 1 / (1 + math.exp(-KL_TO_ALPHA_GAIN * (drift - DRIFT_THRESHOLD)))
```

`KL_TO_ALPHA_GAIN = 1.5`：控制 sigmoid 陡峭度。漂移越大，α 越接近 1（历史被抛弃）。

王女士 case 中：`α = sigmoid(1.5 × (1.41 - 0.40)) ≈ 0.819`

### 4.4 融合公式 + 显式负反馈叠加

[nonstationary.py:43-52](mab_demo/bandits/nonstationary.py)：

```python
for k in ["R1", "R2", "R3", "R4"]:
    old = profile.risk_dist.get(k, 0)
    new = (1 - alpha) * old + alpha * llm_prior["risk"].get(k, 0)
    # 叠加显式负反馈(主动划过 R4 → 0)
    if penalty.get("risk", {}).get(k, 0) <= -0.5:
        new *= 0.05
    fused[k] = new
s = sum(fused.values()) or 1.0
fused = {k: v / s for k, v in fused.items()}
```

**两层叠加**：
1. `(1-α) * historical + α * prior`：主路径
2. `if penalty <= -0.5: new *= 0.05`：阶段一的隐式负反馈（如划过 R4 股票基金）再做二次压制

### 4.5 王女士 case 输出

| 风险等级 | 原画像 | LLM 先验 | 融合后（α=0.819）| 含显式惩罚后 | 归一化 |
|---|---|---|---|---|---|
| R1 | 0.05 | 0.55 | 0.464 | 0.464 | **0.48** |
| R2 | 0.25 | 0.35 | 0.332 | 0.332 | 0.34 |
| R3 | 0.60 | 0.08 | 0.174 | 0.174 | **0.18** |
| R4 | 0.10 | 0.02 | 0.034 | 0.0017（×0.05）| 0.002 |

R3 从 0.60 → 0.18，R1 从 0.05 → 0.48，画像完成"转身"。

### 4.6 为何不用 LLM

本阶段是**纯数值融合**，KL + sigmoid 完全可解释，加 LLM 只会引入不确定性与延迟。这也是设计上的刻意分工：LLM 只负责**生成先验**，融合与决策交给算法。

---

## 5. 对话式组合老虎机 (Slate-CCB)

**目的**：业务约束"每次必须输出 [产品 A + 产品 B + 1 个追问]"。在所有候选组合中，用 **Expected Information Gain (EIG)** 找到"既能利用、又能探索、还能最大化信息收敛"的组合臂。

**源码**：[bandits/slate_ccb.py](mab_demo/bandits/slate_ccb.py)

### 5.1 四步决策管线

```
┌──────────────────────────────────────────────────┐
│  Step 1  应用阶段一隐式负反馈 → 硬黑名单过滤     │
└───────────────────────┬──────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────┐
│  Step 2  质量底线过滤 (term × risk ≥ 0.18)       │
└───────────────────────┬──────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────┐
│  Step 3  选 ask 维度 (不确定性最大的维度)         │
└───────────────────────┬──────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────┐
│  Step 4  枚举 C(N,2) 组合, 打 EIG 分 → 选 argmax │
└──────────────────────────────────────────────────┘
```

### 5.2 评分权重

[slate_ccb.py:20-23](mab_demo/bandits/slate_ccb.py)：

```python
ALPHA_QUALITY  = 0.55   # 利用: 平均匹配度
BETA_COVERAGE  = 0.30   # 探索: 预算覆盖宽度
GAMMA_STRADDLE = 0.15   # 探索: 是否跨越均值（一上一下）
QUALITY_THRESHOLD = 0.18  # 候选质量底线
```

55% 利用 + 45% 探索，是典型的 **exploitation-heavy** 配比——因为金融推荐"乱推一通"的代价很大。

### 5.3 匹配度：高斯核 + 风险查表

[slate_ccb.py:26-35](mab_demo/bandits/slate_ccb.py)：

```python
def _match_term(p, mu, sigma):
    return math.exp(-((p["term_years"] - mu) ** 2) / (2 * sigma ** 2 + 1e-6))

def _match_risk(p, risk_dist):
    return risk_dist.get(p["risk"], 0.0)

def _match_budget(p, mu, sigma):
    return math.exp(-((p["min_invest"] - mu) ** 2) / (2 * sigma ** 2 + 1e-6))
```

- 连续变量（`term`、`budget`）：用**高斯核**衡量产品与后验均值的匹配度
- 离散变量（`risk`）：直接查 `risk_dist` 表

### 5.4 硬约束：term × risk 的乘积底线

[slate_ccb.py:38-41](mab_demo/bandits/slate_ccb.py)：

```python
def _base_quality(p, state):
    """硬约束维度（term × risk）的乘积匹配度。"""
    return _match_term(p, state["term_mu"], state["term_sigma"]) * \
           _match_risk(p, state["risk_dist"])
```

**用乘积而非加权和**：任意一维严重失配都会拉垮整体分，比如 18 年的少儿保险（`term=18` 与 `μ=15` 相差 3σ 仅得 0.08）直接被过滤。这样就**避免了把"看起来沾边但其实错位"的产品当探索臂塞进去**。

### 5.5 Ask 维度选择

[slate_ccb.py:52-60](mab_demo/bandits/slate_ccb.py)：

```python
def _select_ask_dim(state):
    cands = [
        ("budget", min(1.0, state["budget_sigma"] / 4_000)),    # σ=4000 即视为"完全不确定"
        ("term",   min(1.0, state["term_sigma"] / 5.0)),
        ("risk",   _entropy(state["risk_dist"]) * 0.6),
    ]
    cands.sort(key=lambda x: -x[1])
    return cands[0]
```

- `budget` / `term`：按**归一化标准差**打分
- `risk`：按**归一化熵**打分（熵 = 0 意味着风险分布完全确定）
- `0.6` 系数：离散熵与连续 σ 的标度调和

王女士 case 中：`budget_sigma=6000 → 1.0`，`term_sigma=2.0 → 0.4`，`risk_entropy ≈ 0.4`（R1/R2 已集中）→ 选中 **budget**。

### 5.6 组合打分：利用 + 覆盖 + 跨越

[slate_ccb.py:116-129](mab_demo/bandits/slate_ccb.py)：

```python
for a, b in itertools.combinations(qualified, 2):
    qa = _base_quality(a, state)
    qb = _base_quality(b, state)
    avg_quality = (qa + qb) / 2
    # 预算覆盖宽度 (以 σ 为尺)
    budget_gap = abs(a["min_invest"] - b["min_invest"]) / state["budget_sigma"]
    coverage = min(1.0, budget_gap / 2.0)
    # 跨越均值 → 一上一下, 信息增益最大
    straddle = 1.0 if (a["min_invest"] < state["budget_mu"]) != (b["min_invest"] < state["budget_mu"]) else 0.4
    score = ALPHA_QUALITY * avg_quality + BETA_COVERAGE * coverage + GAMMA_STRADDLE * straddle
```

三项启发式：

| 分项 | 表达 | 含义 |
|---|---|---|
| `avg_quality` | 平均 base_quality | **利用**：两款都得是好产品 |
| `coverage` | \|a.amt - b.amt\| / σ | **探索**：预算跨度越大越能收集信息 |
| `straddle` | 一个低于 μ + 一个高于 μ = 1.0 | **信息增益**：用户二选一即可定位 μ 真值 |

王女士 case：P006（1.5w，低于 μ=24000）+ P007（3w，高于 μ）完美跨越，score = 0.674 夺冠。

### 5.7 利用臂 / 探索臂 角色分配

[slate_ccb.py:138-139](mab_demo/bandits/slate_ccb.py)：

```python
# 让较低预算的当"利用臂", 较高预算的当"探索臂"（探索预算上限）
if a["min_invest"] > b["min_invest"]:
    a, b = b, a
```

**业务直觉**：低预算 = 风险低 = 利用；高预算 = 挑战上限 = 探索。这一行代码把 EIG 最优的两款产品分别扣上"利用"和"探索"的语义帽子。

### 5.8 输出

```python
return {"exploit": a, "explore": b, "ask_dim": ask_dim, "eig": ask_eig, "score": score}
```

这个 slate 会交给下游"生臂"阶段去写出**对应 ask_dim 的高情商追问文案**。

---

## 6. 大模型动态生臂 (Generative Arm Bandit)

**目的**：上一阶段定了"问什么维度"（ask_dim=budget），本阶段让 LLM 动态生成 **N 条追问话术候选**，再用可插拔 Reward Model 打分选最优。

**源码**：[bandits/generative_arm.py](mab_demo/bandits/generative_arm.py)，[clients/reward_model.py](mab_demo/clients/reward_model.py)

### 6.1 两个插件的协作

```
Slate(exploit, explore, ask_dim=budget)
           │
           ▼
    ┌──────────────────┐     ┌──────────────────┐
    │  LLM Generator   │────►│  候选追问 N 条   │
    │  temperature=0.8 │     │  每条 {id, text, │
    │  高多样性        │     │       intent}    │
    └──────────────────┘     └─────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────────┐
                              │  Reward Model       │
                              │  (可插拔)           │
                              │  ├─ rule_based      │
                              │  ├─ llm_judge       │
                              │  └─ external_rm     │
                              └────────┬────────────┘
                                       │
                                       ▼
                              argmax score → best_arm
```

### 6.2 LLM 生成候选臂

[generative_arm.py:20-46](mab_demo/bandits/generative_arm.py)：

```python
def _generate_candidates_real(slate: Dict[str, Any], n: int = 4):
    a, b = slate["exploit"], slate["explore"]
    prompt = ARM_PROMPT.format(
        ask_dim=slate["ask_dim"],
        a_amt=a["min_invest"],
        b_amt=b["min_invest"],
        user_name="王女士",
        user_desc="38岁, 有 5 岁子女, 年入 60 万, 已婚家庭",
        recent_behavior="刚才点击过一款 5 万起投的教育年金险 A 秒退(嫌门槛高),"
                        " 又点过一款 1 年期教育理财 B 秒退(嫌期限短)",
        a_name=a["name"], a_risk=a["risk"], a_term=a["term_years"],
        b_name=b["name"], b_risk=b["risk"], b_term=b["term_years"],
        n=n,
    )
    raw = llm_client.chat_json(
        prompt=prompt,
        system=ARM_SYSTEM,
        temperature=0.8,          # 高温度保证多样性
    )
    if raw is None:
        raise RuntimeError("LLM returned None")
    arms = validate_ask_arms(raw, min_arms=3)
    return arms
```

**关键参数**：
- `temperature=0.8`：与阶段二先验（0.2）形成鲜明对比。此处**刻意要多样性**，让 LLM 产出传统表单、纯数字、高情商、营销等不同风格
- `min_arms=3`：Schema 强制至少 3 个候选，否则 RM 没法有效区分

### 6.3 Mock 降级：4 种风格模板

[generative_arm.py:50-69](mab_demo/bandits/generative_arm.py) 硬编码了 4 种典型风格：

| ID | 风格 | 样例（王女士 case）|
|---|---|---|
| ARM_1 | 传统硬表单 | "请填写预算: ___ 元/年" |
| ARM_2 | 纯数字探询 | "您是想每年投 1.5 万还是 3 万?" |
| ARM_3 | 高情商+锚点+痛点 | "王女士，为护航宝宝未来 15 年的教育，..." |
| ARM_4 | 营销话术 | "为孩子的未来加油！..." |

好处：Mock 也能让 RM 的打分逻辑展示出"高情商 > 纯数字 > 营销 > 硬表单"的清晰偏好。

### 6.4 可插拔 Reward Model

[clients/reward_model.py](mab_demo/clients/reward_model.py) 通过 `MAB_RM_TYPE` 环境变量三选一：

| 类型 | 延迟 | 场景 |
|---|---|---|
| `rule_based`（默认）| < 1 ms | 生产主路径，规则匹配 + 长度奖励 |
| `llm_judge` | ~500 ms | 离线评估、新 Prompt 版本对比 |
| `external_rm` | < 10 ms | 生产推荐，专用 RM 推理服务（DeBERTa 等）|

`generative_arm.run()` 只依赖 `reward_model.score(text, slate) -> {total, empathy, precision, conciseness}` 接口，**对具体实现完全不感知**——这就是可插拔的价值。

打分循环 [generative_arm.py:99-109](mab_demo/bandits/generative_arm.py)：

```python
best = None
for arm in arms:
    s = reward_model.score(arm["text"], slate)
    arm["_scores"] = s
    if best is None or s["total"] > best["_scores"]["total"]:
        best = arm
```

标准的 MAB argmax 决策，score 最高者胜出。王女士 case 中，ARM_3（高情商）得分最高。

### 6.5 真实 LLM + Mock 的三态编排

[generative_arm.py:72-84](mab_demo/bandits/generative_arm.py)：

```python
def _generate_candidates(slate):
    source = "LLM"
    if llm_client.enabled:
        try:
            return _generate_candidates_real(slate), source
        except (SchemaError, Exception) as e:
            if not CONFIG.fallback_on_fail:
                raise
            source = f"Mock (LLM 降级: {type(e).__name__})"
    else:
        source = "Mock (LLM 未启用)"
    return _generate_candidates_mock(slate), source
```

与 LLM Prior 结构一致，但参数完全不同（高温度、允许多样性、Schema 只校验 JSON 结构不校验语义）。

---

## 7. 贝叶斯后验坍缩 (Bayesian Posterior Collapse)

**目的**：把客户对追问的真实回复（点击哪款产品 + 文本"差不多每年 2 万吧，别亏钱"）转化为**观测信号**，用贝叶斯共轭更新先验，使方差剧烈收缩，最终固化为"高置信增强画像"。

**源码**：[orchestrator.py](mab_demo/orchestrator.py) 的 `stage_4_evolution()`

### 7.1 三路后验更新并行

```
客户回复 ──────────────────────────┬──────────────────────────┐
  · clicked: P006 (1.5w, R1, 15y) │                          │
  · ignored: P007 (3w, R2)         │                          │
  · text: "差不多每年 2 万吧..."   │                          │
                                   │                          │
              ┌────────────────────┼──────────────────┐       │
              │                    │                  │       │
              ▼                    ▼                  ▼       ▼
       NLU 抽 budget          点击项推 term     显式选 R1   pick over R2
       = 20000               chosen=15y        ignore R2   ignore R2
              │                    │                  │
              ▼                    ▼                  ▼
     高斯-高斯共轭更新      term 强确认坍缩     risk 重加权
     N(prior) × N(obs)       → σ = 0.5         R1 ×3.0 + 0.3
       → N(post)                                 R2 ×0.4
                                                 R3 ×0.05
                                                 R4 ×0.05
                                                 归一化
```

### 7.2 NLU 实时抽取槽位

原先 v1 版本是硬编码的，工业化版本 [orchestrator.py:27-34](mab_demo/orchestrator.py) 调用 NLU 模块：

```python
slots = nlu.extract_slots(WANG_FINAL_REPLY["text"], verbose=verbose)

prior_mu = llm_prior_dist["annual_budget"]["mu"]
prior_sigma = llm_prior_dist["annual_budget"]["sigma"]
obs_mu = slots.get("annual_budget") or prior_mu   # NLU 抽不到则不更新
obs_sigma = 1_500                                 # 客户口语回答的等效观测噪声
```

`nlu.extract_slots` 优先 LLM（JSON 输出 + Schema 校验），失败走正则兜底，进一步贯彻降级原则。

### 7.3 Budget：高斯-高斯共轭（闭式解）

[orchestrator.py:36-38](mab_demo/orchestrator.py)：

```python
post_var = 1 / (1 / prior_sigma**2 + 1 / obs_sigma**2)
post_mu = post_var * (prior_mu / prior_sigma**2 + obs_mu / obs_sigma**2)
post_sigma = math.sqrt(post_var)
```

这是教科书级别的**高斯共轭**公式：精度（1/σ²）相加，加权均值按精度加权。

代入王女士数值：
- Prior: N(24000, 6000²) → 精度 = 1/36M
- Obs:   N(20000, 1500²) → 精度 = 1/2.25M（观测精度是先验的 16 倍）
- Post:  精度 = 1/36M + 1/2.25M = 17/36M
- `post_sigma ≈ 1455`（**σ 从 6000 → 1455，缩 4.1 倍**）
- `post_mu ≈ 20235`（被观测强势拉向 20000）

### 7.4 Term：强确认坍缩

[orchestrator.py:47-50](mab_demo/orchestrator.py)：

```python
chosen_term = next(p["term_years"] for p in PRODUCTS if p["id"] == WANG_FINAL_REPLY["clicked"])
term_post_sigma = 0.5   # 强确认 → 极小方差
term_post_mu = chosen_term
```

客户点击 P006（15 年）是对 LLM 先验"term=15"的**强确认**，直接把 σ 从 2.0 → 0.5。严格说这是**启发式坍缩**而非严格共轭，因为点击行为不是严格的"term 维度观测样本"。生产建议用 Truncated Gaussian 或 Bayesian Logistic Regression 做更严格的建模。

### 7.5 Risk：乘法更新 + 归一化

[orchestrator.py:53-58](mab_demo/orchestrator.py)：

```python
risk_dist_post = copy.deepcopy(decayed_risk)
risk_dist_post["R1"] = risk_dist_post.get("R1", 0) * 3.0 + 0.3   # clicked → 强力放大
risk_dist_post["R2"] = risk_dist_post.get("R2", 0) * 0.4         # ignored → 轻压
risk_dist_post["R3"] = risk_dist_post.get("R3", 0) * 0.05        # 严重压制
risk_dist_post["R4"] = risk_dist_post.get("R4", 0) * 0.05
s = sum(risk_dist_post.values())
risk_dist_post = {k: v / s for k, v in risk_dist_post.items()}
```

**三档乘法因子**：
- 选中的 R1：`×3.0 + 0.3`（偏移项保证即使 α=0 也能抬起来）
- 展示但未选的 R2：`×0.4`（轻度压制，因为用户看了一眼）
- 未展示的 R3 / R4：`×0.05`（重度压制，视为"显式不需要"）

归一化后：R1 ≈ 0.93，R3 ≈ 0.00，彻底消除原画像的 R3 幻觉。

### 7.6 固化为增强画像

[orchestrator.py:60-76](mab_demo/orchestrator.py)：

```python
enhanced = UserProfile(
    name=profile.name, age=profile.age, gender=profile.gender,
    annual_income=profile.annual_income, family=profile.family,
    risk_level="R1",                                    # 主风险从 R3 → R1
    holdings=profile.holdings,
    avg_holding_months=profile.avg_holding_months,
    risk_dist=risk_dist_post,                           # 后验分布
    term_years_mu=term_post_mu,   term_years_sigma=term_post_sigma,     # 15 / 0.5
    annual_budget_mu=post_mu,     annual_budget_sigma=post_sigma,       # 20235 / 1455
    intent_tag="子女教育金·15年定投·R1保本",            # 意图升级
    confidence=0.97,                                    # 0.35 → 0.97
)
```

这个增强画像会被回写到用户画像存储，**后续所有推荐请求都能直接拿到它**——这就是"闭环进化"：一次会话的增值沉淀下来，让 MAB 的学习永久生效。

### 7.7 量化效果

| 维度 | 原始画像 | MAB 增强后 | 缩放 / 提升 |
|---|---|---|---|
| 置信度 | 0.35 | 0.97 | **2.8×** |
| R1 概率 | 0.05 | 0.93 | **18.6×** |
| R3 概率 | 0.60 | ≈0 | ÷∞ |
| annual_budget σ | 20000（常量 baseline）| 1455 | **缩 13.7×** |
| term σ | N/A（硬标签）| 0.5 | 极小 |
| 目标产品命中率 | 0.00% | **91.33%** | —— |

### 7.8 为何又叫"方差坍缩"

"坍缩" 来自量子力学的波函数坍缩隐喻：MAB 在阶段二-三维持了一个**高方差**的叠加态（`budget~N(24000,6000)` 意味着从 1w 到 4w 都有可能），客户的一次真实反馈就是"观测"，瞬间把这个叠加态**塌缩为一个极窄的高置信点**（`N(20235, 1455)`）。这是 Bayesian Bandit 与 frequentist UCB 的本质区别：**它把"不确定性"当一等公民管理**。

---

## 附：算法之间的数据流

```
┌─────────────────────────────────────────────────────────────┐
│  输入: UserProfile + BEHAVIOR_LOG + USER_QUERY              │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
   ① 因果去偏 ──► clean_events + confounders
               │
               ▼
   ② 隐式负反馈 ──► penalty = {min_invest, term, risk}
               │
               ▼
   ③ LLM Prior ──► {term~N, budget~N, risk~Dirichlet}
               │
               ▼
   ④ 非平稳衰减 ──► decayed_risk (R3↓, R1↑)
                          │
                          ▼
               ⑤ Slate-CCB ──► {exploit, explore, ask_dim}
                          │
                          ▼
               ⑥ 大模型生臂 ──► best_arm (高情商追问)
                          │
                          ▼ (客户真实回复)
               ⑦ 贝叶斯后验坍缩 ──► 增强画像 (R1, 2w/年, 15y)
```

每一步的输出类型严格约束（UserProfile / dict / 分布），保证链路可测试、可替换、可观测。
