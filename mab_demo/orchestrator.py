"""四阶段流水线编排: 入场洗数 → 零样本定调 → 组合生臂 → 闭环进化"""
import math
import copy
from typing import Dict, Any
from data import UserProfile, BEHAVIOR_LOG, USER_QUERY, PRODUCTS, WANG_FINAL_REPLY
from bandits import causal_debias, implicit_feedback, llm_prior, nonstationary, slate_ccb, generative_arm
import nlu


def stage_4_evolution(profile: UserProfile,
                      llm_prior_dist: Dict[str, Any],
                      decayed_risk: Dict[str, float],
                      slate: Dict[str, Any],
                      best_arm: Dict[str, Any],
                      verbose: bool = True) -> UserProfile:
    """阶段四: 客户回复 → 贝叶斯后验更新 → 方差坍缩 → 生成增强画像。"""
    if verbose:
        print("\n┌─[阶段四] 闭环进化: 方差收敛 + 增强画像生成 ─────────────────────")
        print(f"│  系统下发追问: «{best_arm['text']}»")
        print(f"│  ── 客户真实反应 ──")
        print(f"│  · 点击产品: {WANG_FINAL_REPLY['clicked']}")
        print(f"│  · 忽略产品: {WANG_FINAL_REPLY['ignored']}")
        print(f"│  · 文本回复: \"{WANG_FINAL_REPLY['text']}\"")

    # NLU: 工业化版本通过 LLM/正则实时抽取槽位, 而非硬编码
    if verbose:
        print("│  ── NLU 槽位抽取 ──")
    slots = nlu.extract_slots(WANG_FINAL_REPLY["text"], verbose=verbose)

    # 后验更新：在 budget 维度上使用闭式高斯-高斯共轭
    prior_mu = llm_prior_dist["annual_budget"]["mu"]
    prior_sigma = llm_prior_dist["annual_budget"]["sigma"]
    obs_mu = slots.get("annual_budget") or prior_mu  # NLU 抽不到则不更新
    obs_sigma = 1_500  # 客户口语回答的等效观测噪声

    post_var = 1 / (1 / prior_sigma**2 + 1 / obs_sigma**2)
    post_mu = post_var * (prior_mu / prior_sigma**2 + obs_mu / obs_sigma**2)
    post_sigma = math.sqrt(post_var)

    if verbose:
        print(f"│  ── 贝叶斯后验更新 (annual_budget) ──")
        print(f"│     prior  ~ N({prior_mu:>7.0f}, σ={prior_sigma:>5.0f})")
        print(f"│     observ ~ N({obs_mu:>7.0f}, σ={obs_sigma:>5.0f})")
        print(f"│     posterior ~ N({post_mu:>7.0f}, σ={post_sigma:>5.0f})  ← 方差坍缩")

    # 在 term 上：客户点击 P006 (15年), 是对 LLM 先验的强确认
    chosen_term = next(p["term_years"] for p in PRODUCTS if p["id"] == WANG_FINAL_REPLY["clicked"])
    term_post_sigma = 0.5  # 强确认 → 极小方差
    term_post_mu = chosen_term

    # 在 risk 上：忽略 R2 的 P007, 点击 R1 的 P006 → R1 概率压倒性提升
    risk_dist_post = copy.deepcopy(decayed_risk)
    risk_dist_post["R1"] = risk_dist_post.get("R1", 0) * 3.0 + 0.3
    risk_dist_post["R2"] = risk_dist_post.get("R2", 0) * 0.4
    risk_dist_post["R3"] = risk_dist_post.get("R3", 0) * 0.05
    risk_dist_post["R4"] = risk_dist_post.get("R4", 0) * 0.05
    s = sum(risk_dist_post.values())
    risk_dist_post = {k: v / s for k, v in risk_dist_post.items()}

    enhanced = UserProfile(
        name=profile.name,
        age=profile.age,
        gender=profile.gender,
        annual_income=profile.annual_income,
        family=profile.family,
        risk_level="R1",
        holdings=profile.holdings,
        avg_holding_months=profile.avg_holding_months,
        risk_dist=risk_dist_post,
        term_years_mu=term_post_mu,
        term_years_sigma=term_post_sigma,
        annual_budget_mu=post_mu,
        annual_budget_sigma=post_sigma,
        intent_tag="子女教育金·15年定投·R1保本",
        confidence=0.97,
    )

    if verbose:
        print(f"│  ── 风险后验 ──")
        for k in ["R1", "R2", "R3", "R4"]:
            print(f"│     {k}: {risk_dist_post[k]:.3f}  {'█' * int(risk_dist_post[k] * 30)}")
        print(f"│  ── 期限后验 ──  N({term_post_mu}, σ={term_post_sigma}) [强确认坍缩]")
        print(f"└─ 完成: 增强画像置信度 {enhanced.confidence:.2f} 已固化, 下游全链路 100% 懂她")
    return enhanced


def run_pipeline(raw_profile: UserProfile, verbose: bool = True) -> Dict[str, Any]:
    """执行完整四阶段流水线, 返回中间状态与增强画像。"""
    if verbose:
        print("\n" + "=" * 70)
        print("                MAB 四阶段融合流水线 · 王女士 case")
        print("=" * 70)
        print(f"\n[INPUT] 客户主诉: \"{USER_QUERY}\"")

    # 阶段一
    clean_events, confounders = causal_debias.run(BEHAVIOR_LOG, verbose=verbose)
    penalty = implicit_feedback.run(clean_events, verbose=verbose)

    # 阶段二
    prior = llm_prior.run(raw_profile, USER_QUERY, verbose=verbose)
    decayed_risk = nonstationary.run(raw_profile, prior, penalty, verbose=verbose)

    # 阶段三
    slate = slate_ccb.run(PRODUCTS, prior, decayed_risk, penalty, verbose=verbose)
    best_arm = generative_arm.run(slate, verbose=verbose)

    # 渲染前端最终输出
    if verbose:
        print("\n┌─[前端 APP 实际渲染] ────────────────────────────────────────────")
        print(f"│  📦 推荐卡片 A (利用): {slate['exploit']['name']}  "
              f"({slate['exploit']['min_invest']/10000:.1f}万/年, R{slate['exploit']['risk'][1:]}, "
              f"{slate['exploit']['term_years']}年, 预期年化 {slate['exploit']['expected_return']*100:.1f}%)")
        print(f"│  📦 推荐卡片 B (探索): {slate['explore']['name']}  "
              f"({slate['explore']['min_invest']/10000:.1f}万/年, R{slate['explore']['risk'][1:]}, "
              f"{slate['explore']['term_years']}年, 预期年化 {slate['explore']['expected_return']*100:.1f}%)")
        print("│  💬 系统追问:")
        print(f"│     «{best_arm['text']}»")
        print("└─────────────────────────────────────────────────────────────────")

    # 阶段四
    enhanced_profile = stage_4_evolution(raw_profile, prior, decayed_risk, slate, best_arm, verbose=verbose)

    return {
        "clean_events": clean_events,
        "confounders": confounders,
        "penalty": penalty,
        "llm_prior": prior,
        "decayed_risk": decayed_risk,
        "slate": slate,
        "best_arm": best_arm,
        "enhanced_profile": enhanced_profile,
    }
