"""阶段一·算法2：隐式负反馈老虎机 (Implicit Negative Feedback MAB)

把"秒退"和"主动划过"翻译成强烈的负向 reward, 写入特征惩罚矩阵。

【因果归因】秒退的根因往往集中在某一维度（金额过高 OR 期限过短）, 而不是全维度都不好。
本模块用"教育金常识基线"反查偏离最大的维度作为主责, 把惩罚集中打到该维度,
避免把 R1 (本身是好风险等级) 这种次要维度也打到, 造成阶段三候选空集。

主动划过则通常是对显式风险等级的拒绝, 单独扣 risk 维度。
"""
from collections import defaultdict
from typing import List, Dict, Any


QUICK_EXIT_THRESHOLD = 5
QUICK_EXIT_PENALTY = -1.0
SWIPE_PENALTY = -0.8

# 教育金场景的常识基线（用于因果归因, 实际可由 LLM 输出）
EDU_BASELINE = {"min_invest": 20_000, "term_years": 15}


def _bucket_invest(amount: int) -> str:
    if amount >= 50_000:
        return "≥5万"
    if amount >= 10_000:
        return "1-5万"
    return "<1万"


def _bucket_term(years: float) -> str:
    if years <= 1:
        return "≤1年"
    if years <= 5:
        return "1-5年"
    if years <= 10:
        return "5-10年"
    return ">10年"


def _attribute_root_cause(attrs: Dict[str, Any]) -> str:
    """归因到偏离基线最大的那个维度。"""
    invest_dev = abs(attrs["min_invest"] - EDU_BASELINE["min_invest"]) / EDU_BASELINE["min_invest"]
    term_dev = abs(attrs["term_years"] - EDU_BASELINE["term_years"]) / EDU_BASELINE["term_years"]
    return "min_invest" if invest_dev >= term_dev else "term"


def run(clean_events: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Dict[str, float]]:
    penalty: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    if verbose:
        print("\n┌─[阶段一·算法2] 隐式负反馈 MAB · 启动 ─────────────────────────────")
    for ev in clean_events:
        attrs = ev.get("product_attrs")
        if not attrs:
            continue

        # 主动划过: 显式拒绝该风险等级
        if ev.get("silent_ignore"):
            penalty["risk"][attrs["risk"]] += SWIPE_PENALTY
            if verbose:
                print(f"│  ⛔ [{ev['time']}] {ev['target']:<22}  主动划过        → "
                      f"惩罚 risk={attrs['risk']} (Δ={SWIPE_PENALTY:+.1f})")
            continue

        # 秒退: 因果归因到主责维度
        if ev.get("quick_exit") or (ev.get("dwell_seconds", 999) < QUICK_EXIT_THRESHOLD):
            root = _attribute_root_cause(attrs)
            invest_b = _bucket_invest(attrs["min_invest"])
            term_b = _bucket_term(attrs["term_years"])

            if root == "min_invest":
                penalty["min_invest"][invest_b] += QUICK_EXIT_PENALTY
                penalty["term"][term_b] += QUICK_EXIT_PENALTY * 0.2  # 次要责任轻微扣分
                root_str = f"min_invest={invest_b} (主责)"
            else:
                penalty["term"][term_b] += QUICK_EXIT_PENALTY
                penalty["min_invest"][invest_b] += QUICK_EXIT_PENALTY * 0.2
                root_str = f"term={term_b} (主责)"

            if verbose:
                print(f"│  ⛔ [{ev['time']}] {ev['target']:<22}  秒退({ev.get('dwell_seconds')}s)  "
                      f"→ 因果归因: {root_str}")

    if verbose:
        print("│  ── 累计惩罚矩阵 (因果归因后) ──")
        for dim in ("min_invest", "term", "risk"):
            for v, r in sorted(penalty.get(dim, {}).items()):
                print(f"│     penalty[{dim}][{v:<6}] = {r:+.2f}")
        print("└─ 完成: 主责维度强势拉黑(高门槛/短期), risk 维度交由阶段二处理")
    return {k: dict(v) for k, v in penalty.items()}
