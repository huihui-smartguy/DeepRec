"""阶段二·算法1：大模型生成先验分布 (LLM-Generated Priors)

打破 MAB 冷启动盲盒。让 LLM 基于静态画像 + 主诉, 通过常识推理生成先验分布矩阵,
直接注入 MAB, 让其第一轮探索就锁定优质区域。

本 demo 中 LLM 推理被 mock 成确定性逻辑（保证可复现）。
真实工业实现：LLM 输出 JSON, 包含每个维度的(均值/方差/类别概率)。
"""
from typing import Dict, Any
from data import UserProfile


def _llm_reasoning(profile: UserProfile, query: str) -> Dict[str, Any]:
    """模拟 LLM 的链式推理输出（CoT trace）。"""
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
            "term_years": {"mu": 15.0, "sigma": 2.0},          # 高斯先验
            "annual_budget": {"mu": 24_000, "sigma": 6_000},
            "risk": {"R1": 0.55, "R2": 0.35, "R3": 0.08, "R4": 0.02},
        },
    }


def run(profile: UserProfile, query: str, verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("\n┌─[阶段二·算法1] 大模型生成先验分布 (LLM Prior) · 启动 ────────────")
        print(f"│  输入: 静态画像({profile.name}, {profile.age}岁, 年入{profile.annual_income/10000:.0f}万, 5岁子女)")
        print(f"│  主诉: \"{query}\"")
    out = _llm_reasoning(profile, query)
    if verbose:
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
