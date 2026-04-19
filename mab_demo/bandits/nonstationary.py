"""阶段二·算法2：非平稳老虎机 (Non-stationary MAB / Discounted Bandit)

检测到强烈的"概念漂移"信号(本轮主诉与历史画像冲突)后, 启动指数级时间衰减,
让 LLM 先验主导风险维度, 实现"秒级画像转身"。

实现思路: 用漂移强度生成融合权重 α∈(0,1), 输出
  fused[k] = (1-α) * historical[k] + α * llm_prior[k]
α 越大表示历史越不可信。
"""
import math
from typing import Dict, Any
from data import UserProfile


DRIFT_THRESHOLD = 0.4
KL_TO_ALPHA_GAIN = 1.5  # KL → α 的灵敏度


def _kl(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-6
    s = 0.0
    for k in set(p.keys()) | set(q.keys()):
        pk = p.get(k, eps)
        qk = q.get(k, eps)
        s += pk * math.log((pk + eps) / (qk + eps))
    return max(0.0, s)


def run(profile: UserProfile, llm_prior: Dict[str, Any], penalty: Dict[str, Dict[str, float]],
        verbose: bool = True) -> Dict[str, float]:
    if verbose:
        print("\n┌─[阶段二·算法2] 非平稳衰减 MAB · 启动 ─────────────────────────────")
    drift = _kl(llm_prior["risk"], profile.risk_dist)
    triggered = drift > DRIFT_THRESHOLD
    if verbose:
        print(f"│  概念漂移强度 KL(LLM_prior || historical) = {drift:.3f}")
        print(f"│  漂移阈值 = {DRIFT_THRESHOLD}  → {'触发衰减' if triggered else '不触发'}")

    fused = dict(profile.risk_dist)
    if triggered:
        # 用 sigmoid(KL) 把漂移转化为融合权重 α∈(0,1)
        alpha = 1 / (1 + math.exp(-KL_TO_ALPHA_GAIN * (drift - DRIFT_THRESHOLD)))
        # 主动划过的隐式负反馈也参与: 若 R4 已被显式扣分, 额外把 R4 压到 0
        for k in ["R1", "R2", "R3", "R4"]:
            old = profile.risk_dist.get(k, 0)
            new = (1 - alpha) * old + alpha * llm_prior["risk"].get(k, 0)
            # 叠加显式负反馈(主动划过 R4 → 0)
            if penalty.get("risk", {}).get(k, 0) <= -0.5:
                new *= 0.05
            fused[k] = new
        s = sum(fused.values()) or 1.0
        fused = {k: v / s for k, v in fused.items()}
        if verbose:
            print(f"│  融合权重 α = {alpha:.3f} (历史权重 {1-alpha:.3f}, 先验权重 {alpha:.3f})")
            print("│  ── 风险后验融合结果 ──")
            for k in ["R1", "R2", "R3", "R4"]:
                old = profile.risk_dist.get(k, 0)
                new = fused.get(k, 0)
                bar = "█" * int(new * 30)
                arrow = "↑" if new > old else "↓"
                print(f"│     {k}: {old:.2f} ──融合→ {new:.2f} {arrow}  {bar}")
    if verbose:
        print("└─ 完成: 旧 R3 主导坍缩, R1 主导成型, 主动划过的 R4 被双重压制")
    return fused
