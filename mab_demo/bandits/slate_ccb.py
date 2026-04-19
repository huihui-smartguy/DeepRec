"""阶段三·算法1：对话式组合老虎机 (Slate Conversational Contextual Bandit)

业务约束: 必须输出 [产品A + 产品B + 1 个追问]。
MAB 在所有候选组合中, 用 Expected Information Gain (EIG) 挑出"既能利用、又能探索、
还能获取最多信息以收敛后验"的组合臂。

设计要点:
  1. 两款产品都必须满足"教育金底线": term 与 risk 必须有可观的匹配度
     （这是工业级 Slate-CCB 的"硬底线"质量过滤, 防止把不靠谱的产品当探索臂塞进去）。
  2. 在质量过滤后的候选中, 选出一对在 budget 维度上"跨越后验均值"的组合,
     形成"低预算保底 + 高预算探索"的精算夹击, 最大化对未知维度的信息增益。
  3. ask_dim 选择最具信息增益的维度（按 σ 与 entropy 的归一化贡献排序）。
"""
import itertools
import math
from typing import Dict, List, Any


# 评分权重
ALPHA_QUALITY = 0.55     # 利用: 平均匹配度
BETA_COVERAGE = 0.30     # 探索: 预算覆盖宽度
GAMMA_STRADDLE = 0.15    # 探索: 是否跨越均值（一上一下）
QUALITY_THRESHOLD = 0.18 # 候选质量底线（防止低质量产品当探索臂）


def _match_term(p: Dict[str, Any], mu: float, sigma: float) -> float:
    return math.exp(-((p["term_years"] - mu) ** 2) / (2 * sigma ** 2 + 1e-6))


def _match_risk(p: Dict[str, Any], risk_dist: Dict[str, float]) -> float:
    return risk_dist.get(p["risk"], 0.0)


def _match_budget(p: Dict[str, Any], mu: float, sigma: float) -> float:
    return math.exp(-((p["min_invest"] - mu) ** 2) / (2 * sigma ** 2 + 1e-6))


def _base_quality(p: Dict[str, Any], state: Dict[str, Any]) -> float:
    """硬约束维度（term × risk）的乘积匹配度。"""
    return _match_term(p, state["term_mu"], state["term_sigma"]) * \
           _match_risk(p, state["risk_dist"])


def _entropy(dist: Dict[str, float]) -> float:
    s = 0.0
    for v in dist.values():
        if v > 0:
            s -= v * math.log(v + 1e-9)
    return s / math.log(4)


def _select_ask_dim(state: Dict[str, Any]) -> tuple:
    """挑出归一化不确定性最大的维度作为追问维度。"""
    cands = [
        ("budget", min(1.0, state["budget_sigma"] / 4_000)),  # σ=4000 即视为"完全不确定"
        ("term",   min(1.0, state["term_sigma"] / 5.0)),
        ("risk",   _entropy(state["risk_dist"]) * 0.6),
    ]
    cands.sort(key=lambda x: -x[1])
    return cands[0]


def run(products: List[Dict[str, Any]],
        llm_prior: Dict[str, Any],
        decayed_risk: Dict[str, float],
        penalty: Dict[str, Dict[str, float]],
        verbose: bool = True) -> Dict[str, Any]:
    state = {
        "term_mu": llm_prior["term_years"]["mu"],
        "term_sigma": llm_prior["term_years"]["sigma"],
        "budget_mu": llm_prior["annual_budget"]["mu"],
        "budget_sigma": llm_prior["annual_budget"]["sigma"],
        "risk_dist": decayed_risk,
    }
    if verbose:
        print("\n┌─[阶段三·算法1] 对话式组合 MAB (Slate-CCB) · 启动 ────────────────")
        print(f"│  当前后验状态: term~N({state['term_mu']},{state['term_sigma']}), "
              f"budget~N({state['budget_mu']},{state['budget_sigma']})")

    # —— Step 1: 应用阶段一隐式负反馈做硬过滤 —— #
    invest_blacklist = {k for k, v in penalty.get("min_invest", {}).items() if v <= -0.8}
    term_blacklist = {k for k, v in penalty.get("term", {}).items() if v <= -0.8}

    def in_blacklist(p):
        invest_b = "≥5万" if p["min_invest"] >= 50_000 else ("1-5万" if p["min_invest"] >= 10_000 else "<1万")
        term_b = "≤1年" if p["term_years"] <= 1 else ("1-5年" if p["term_years"] <= 5 else
                ("5-10年" if p["term_years"] <= 10 else ">10年"))
        return invest_b in invest_blacklist or term_b in term_blacklist

    after_neg_filter = [p for p in products if not in_blacklist(p)]
    if verbose:
        print(f"│  Step 1 隐式负反馈硬过滤: {len(products)} → {len(after_neg_filter)} 款 "
              f"(剔除 {invest_blacklist | term_blacklist})")

    # —— Step 2: 质量底线过滤(教育金 must-have: 长期 + 低风险) —— #
    qualified = [p for p in after_neg_filter if _base_quality(p, state) >= QUALITY_THRESHOLD]
    if verbose:
        print(f"│  Step 2 质量底线过滤(term×risk ≥ {QUALITY_THRESHOLD}): "
              f"{len(after_neg_filter)} → {len(qualified)} 款")
        for p in qualified:
            bq = _base_quality(p, state)
            print(f"│     · {p['id']} {p['name']:<24} risk={p['risk']} term={p['term_years']:>4}年 "
                  f"起投={p['min_invest']:>6}  base_quality={bq:.3f}")

    # —— Step 3: ask 维度选择 —— #
    ask_dim, ask_eig = _select_ask_dim(state)

    # —— Step 4: 枚举 2 款组合, 计算 EIG —— #
    if verbose:
        print(f"│  Step 3 ask 维度选择: 不确定性最大维度 = {ask_dim} (EIG≈{ask_eig:.3f})")
        print(f"│  Step 4 组合臂枚举与打分:")

    best_slate = None
    best_score = -math.inf
    rows = []
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
        rows.append((score, a, b, avg_quality, coverage, straddle))
        if score > best_score:
            best_score = score
            best_slate = (a, b, score, avg_quality, coverage, straddle)
    rows.sort(key=lambda x: -x[0])
    if verbose:
        for s, a, b, q, cv, st in rows[:5]:
            print(f"│     score={s:.3f}  [{a['id']:<5}+{b['id']:<5}]  "
                  f"avg_quality={q:.3f}  coverage={cv:.3f}  straddle={st:.1f}")

    a, b, score, avg_quality, coverage, straddle = best_slate
    # 让较低预算的当"利用臂", 较高预算的当"探索臂"（探索预算上限）
    if a["min_invest"] > b["min_invest"]:
        a, b = b, a

    if verbose:
        print(f"│  ✅ 最优组合: {a['id']}({a['name']}) [利用臂] + {b['id']}({b['name']}) [探索臂]")
        print(f"│              + 询问臂: 探询维度 = {ask_dim}")
        print("└─ 完成: 组合臂(2 产品 + 1 追问)已下发给 LLM 渲染层")
    return {"exploit": a, "explore": b, "ask_dim": ask_dim, "eig": ask_eig, "score": score}
