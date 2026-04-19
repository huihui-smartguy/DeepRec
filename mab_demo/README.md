# 金融场景 MAB 多臂老虎机 Demo（工业化版本）

复现《金融场景MAB技术应用设计文档.md》中"王女士 case"的全链路 MAB 算法栈，**支持真实 LLM 接入 + 优雅降级**，一键运行。

## 一、运行

### 方式 A：零依赖跑（不接 LLM，自动走 Mock 降级）
```bash
cd mab_demo
python3 run_demo.py
```

### 方式 B：接入真实 LLM（DeepSeek / Qwen / vLLM / OpenAI）
```bash
pip install -r requirements.txt
export MAB_LLM_API_KEY=sk-xxxxxxxx
export MAB_LLM_BASE_URL=https://api.deepseek.com/v1   # 按需替换
export MAB_LLM_MODEL=deepseek-chat
python3 run_demo.py
```

两种方式输出效果一致（目标产品命中率 91.33%），区别仅在于 LLM 路径的推理质量上限。

### 方式 C：测试 LLM 路径（Fake Client，不消耗真实 API 额度）
```bash
python3 tests/test_llm_path.py
```

## 二、支持的环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MAB_LLM_API_KEY` | (空) | OpenAI 兼容 API key；留空则全链路走 Mock |
| `MAB_LLM_BASE_URL` | `https://api.deepseek.com/v1` | endpoint |
| `MAB_LLM_MODEL` | `deepseek-chat` | 模型名 |
| `MAB_LLM_TIMEOUT` | `30` | 请求超时（秒） |
| `MAB_LLM_RETRIES` | `2` | 失败重试次数 |
| `MAB_RM_TYPE` | `rule_based` | 奖励模型：`rule_based` / `llm_judge` / `external_rm` |
| `MAB_RM_ENDPOINT` | (空) | 外部 RM 服务地址 |
| `MAB_USE_CACHE` | `1` | 是否启用 LLM 进程内缓存 |
| `MAB_CAUSAL_USE_LLM` | `0` | 因果去偏是否启用 LLM 兜底（生产慎开） |
| `MAB_FALLBACK_ON_FAIL` | `1` | LLM 失败时是否回退 Mock |

## 三、Demo 解决了什么问题

王女士在 APP 中输入"推荐一款教育理财产品"。但她的：
- **历史画像**：R3 中等风险，6 个月持有周期 → 与"教育金=15年R1"矛盾
- **当日行为**：点过 5 万的教育年金险又秒退、点过 1 年期理财又秒退、划过股票基金不看 → 大量噪音与漏掉的负样本

传统 SQL 过滤推荐在这种"画像-意图冲突"下，命中率几乎为 0。
本 demo 用 6 种 MAB 算法 4 阶段串联，把画像 5 秒内从 R3 短期 → R1·15年·2万/年 自动收敛。

## 四、6 种 MAB 算法对应的源码

| 阶段 | 算法 | 源码 | LLM 接入？ |
|---|---|---|---|
| ① 入场洗数 | 因果去偏老虎机 | [`bandits/causal_debias.py`](bandits/causal_debias.py) | 可选（默认关） |
| ① 入场洗数 | 隐式负反馈老虎机 | [`bandits/implicit_feedback.py`](bandits/implicit_feedback.py) | ✗ |
| ② 零样本定调 | LLM 生成先验分布 | [`bandits/llm_prior.py`](bandits/llm_prior.py) | **✓** |
| ② 零样本定调 | 非平稳衰减老虎机 | [`bandits/nonstationary.py`](bandits/nonstationary.py) | ✗ |
| ③ 组合生臂 | 对话式组合老虎机 | [`bandits/slate_ccb.py`](bandits/slate_ccb.py) | ✗ |
| ③ 组合生臂 | 大模型动态生臂 | [`bandits/generative_arm.py`](bandits/generative_arm.py) | **✓** |
| ④ 闭环进化 | 贝叶斯后验坍缩 | [`orchestrator.stage_4_evolution`](orchestrator.py) | + NLU (**✓**) |

## 五、画像对比效果（核心 KPI）

| 维度 | 原始画像 | MAB 增强后 |
|---|---|---|
| 画像置信度 | 0.35 | **0.97** |
| 意图标签 | 短期理财 | **子女教育金·15年定投·R1保本** |
| R1 风险概率 | 0.05 | **0.93** |
| R3 风险概率 | 0.60 | **0.00** |
| 年预算 σ | 20,000 | **1,455 (缩 13.7x)** |
| **目标产品命中率** | **0.00%** | **91.33%** |

## 六、工业化基础设施

- [`clients/llm_client.py`](clients/llm_client.py) — OpenAI 兼容客户端（retry / cache / timeout / 降级）
- [`clients/reward_model.py`](clients/reward_model.py) — 可插拔奖励模型（rule / LLM judge / external）
- [`schemas.py`](schemas.py) — LLM 输出 JSON Schema 严格校验（防幻觉炸管道）
- [`prompts/`](prompts/) — Prompt 模板，版本化管理
- [`nlu.py`](nlu.py) — NLU 槽位抽取（LLM + 正则兜底）
- [`config.py`](config.py) — 环境变量聚合
- [`tests/test_llm_path.py`](tests/test_llm_path.py) — LLM 路径单测

详细架构说明见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。
