"""阶段二·算法1：大模型生成先验分布 (LLM-Generated Priors)

工业化版本:
  - 主路径: 调真实 LLM (OpenAI/DeepSeek/Qwen/vLLM 均兼容), 严格 JSON 校验
  - 降级路径: LLM 不可用 / 校验失败 时, 回退到常识推理 mock, 保证 demo 可跑

打破 MAB 冷启动盲盒, 让 MAB 第一轮就锁定优质区域。
"""
import logging
from typing import Any, Dict

from clients.llm_client import llm_client
from config import CONFIG
from data import UserProfile
from prompts.llm_prior import PRIOR_PROMPT, PRIOR_SYSTEM, VERSION
from schemas import SchemaError, validate_llm_prior

log = logging.getLogger("mab_demo.llm_prior")


# -------------------------- 真实 LLM 路径 -------------------------- #
def _llm_reasoning_real(profile: UserProfile, query: str) -> Dict[str, Any]:
    """调用真实大模型, 失败抛异常 (上层捕获后降级)。"""
    prompt = PRIOR_PROMPT.format(
        name=profile.name,
        age=profile.age,
        gender=profile.gender,
        annual_income=profile.annual_income,
        family=profile.family,
        risk_level=profile.risk_level,
        holdings=profile.holdings,
        avg_holding_months=profile.avg_holding_months,
        query=query,
    )
    raw = llm_client.chat_json(
        prompt=prompt,
        system=PRIOR_SYSTEM,
        temperature=0.2,  # 先验生成偏确定性, 低温度
    )
    if raw is None:
        raise RuntimeError("LLM client returned None (network/auth error)")
    return validate_llm_prior(raw)


# -------------------------- Mock 降级路径 -------------------------- #
def _llm_reasoning_mock(profile: UserProfile, query: str) -> Dict[str, Any]:
    """常识推理 mock, 保证无 LLM 环境下 demo 完整可跑。"""
    trace = []
    trace.append(f"主诉解析: '{query}' → 场景=子女教育金规划")
    child_age = 5
    horizon = 22 - child_age
    trace.append(f"家庭结构: 5岁子女 → 至大学(22岁)需 {horizon} 年; 主流教育定投周期 15 年")
    income = profile.annual_income
    edu_lo, edu_hi = int(income * 0.03), int(income * 0.05)
    trace.append(f"收入水平: 年入{income/10000:.0f}万 → 教育定投通常占 3%-5% (即 {edu_lo}-{edu_hi}/年)")
    trace.append("教育金底线: 本金安全 > 高收益 → 建议风险等级集中在 R1, R2 占次")
    trace.append("结论: 先验落点 (期限~15年, 预算~1.8-3w, 风险≈R1) 高置信")
    return {
        "cot_trace": trace,
        "priors": {
            "term_years": {"mu": 15.0, "sigma": 2.0},
            "annual_budget": {"mu": 24_000, "sigma": 6_000},
            "risk": {"R1": 0.55, "R2": 0.35, "R3": 0.08, "R4": 0.02},
        },
    }


def _llm_reasoning(profile: UserProfile, query: str, verbose: bool = False):
    """主入口: 优先真 LLM, 异常回退 mock (并打印降级原因)。"""
    source = "LLM"
    if llm_client.enabled:
        try:
            out = _llm_reasoning_real(profile, query)
            return out, source
        except (SchemaError, Exception) as e:
            if not CONFIG.fallback_on_fail:
                raise
            source = f"Mock (LLM 降级: {type(e).__name__})"
            log.warning(f"LLM prior failed, fallback to mock: {e}")
    else:
        source = "Mock (LLM 未启用)"

    out = _llm_reasoning_mock(profile, query)
    return out, source


# -------------------------- 对外接口 -------------------------- #
def run(profile: UserProfile, query: str, verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("\n┌─[阶段二·算法1] 大模型生成先验分布 (LLM Prior) · 启动 ────────────")
        print(f"│  Prompt 版本: {VERSION}")
        print(f"│  输入: 静态画像({profile.name}, {profile.age}岁, 年入{profile.annual_income/10000:.0f}万, 5岁子女)")
        print(f"│  主诉: \"{query}\"")

    out, source = _llm_reasoning(profile, query, verbose=verbose)

    if verbose:
        print(f"│  推理来源: {source}")
        print("│  ── LLM 链式推理 (CoT) ──")
        for step in out["cot_trace"]:
            print(f"│     · {step}")
        print("│  ── 注入 MAB 的先验矩阵 ──")
        p = out["priors"]
        print(f"│     prior[term_years]    ~ N(μ={p['term_years']['mu']}, σ={p['term_years']['sigma']})")
        print(f"│     prior[annual_budget] ~ N(μ={p['annual_budget']['mu']}, σ={p['annual_budget']['sigma']})")
        print(f"│     prior[risk]          ~ {p['risk']}")
        print("└─ 完成: MAB 跳过试错, 直接命中(15年, 1.8-3w, R1/R2)优质区域")
    return out["priors"]
