"""阶段三·算法2：大模型动态生臂 (Generative Arm Bandit)

LLM 接到 MAB 下发的硬核机器指令 [Action: Target_Budget; Anchor: 1.5w vs 3w] 后,
实时生成 N 句不同语气的追问候选 (动态臂); MAB 用极速预估打分挑出胜率最高的一句。

每个候选臂的 reward 由三个子分构成:
  empathy : 情绪价值(是否提到客户痛点/家人/孩子)
  precision: 是否锚定具体数字（1.5w / 3w）
  conciseness: 长度惩罚(过长扣分)
"""
from typing import Dict, Any, List


# LLM 生成的候选追问臂（mock）
def _generate_candidates(slate: Dict[str, Any]) -> List[Dict[str, Any]]:
    a, b = slate["exploit"], slate["explore"]
    return [
        {
            "id": "ARM_1",
            "text": f"请填写预算: ___ 元/年",
            "_intent": "传统硬表单",
        },
        {
            "id": "ARM_2",
            "text": f"您是想每年投 {a['min_invest']/10000:.1f} 万还是 {b['min_invest']/10000:.0f} 万?",
            "_intent": "纯数字探询",
        },
        {
            "id": "ARM_3",
            "text": (
                f"王女士, 为护航宝宝未来 15 年的教育, 我为您优选了这两款专属定投。"
                f"刚才注意到那款 5 万起投的可能单笔资金占用偏大。"
                f"为了兼顾您的日常品质生活, 咱们是更倾向于每年 "
                f"{a['min_invest']/10000:.1f} 万的轻松节奏, "
                f"还是稍微增加预算到 {b['min_invest']/10000:.0f} 万去争取长期丰厚回报呢?"
            ),
            "_intent": "高情商 + 锚点 + 痛点回应",
        },
        {
            "id": "ARM_4",
            "text": f"为孩子的未来加油! 您打算每年投多少呢? 越多收益越好哦~",
            "_intent": "营销话术",
        },
    ]


def _score_arm(arm: Dict[str, Any], slate: Dict[str, Any]) -> Dict[str, float]:
    text = arm["text"]
    # empathy: 提到痛点/客户/孩子
    empathy_keywords = ["王女士", "宝宝", "孩子", "品质生活", "护航", "节奏"]
    empathy = sum(1 for k in empathy_keywords if k in text) / len(empathy_keywords)
    # precision: 是否同时含 A、B 两个金额锚点
    a_amt = f"{slate['exploit']['min_invest']/10000:.1f}"
    b_amt = f"{slate['explore']['min_invest']/10000:.0f}"
    precision = (1.0 if a_amt in text else 0.0) * 0.5 + (1.0 if b_amt in text else 0.0) * 0.5
    # conciseness: 长度惩罚, 100-180 字最佳
    L = len(text)
    if L < 30:
        conciseness = 0.4
    elif L < 100:
        conciseness = 0.85
    elif L < 200:
        conciseness = 1.0
    else:
        conciseness = max(0.3, 1.0 - (L - 200) / 200)
    total = 0.5 * empathy + 0.3 * precision + 0.2 * conciseness
    return {"empathy": empathy, "precision": precision, "conciseness": conciseness, "total": total}


def run(slate: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("\n┌─[阶段三·算法2] 大模型动态生臂 (Generative Arms) · 启动 ──────────")
        print(f"│  指令(机器格式): [Action: Target_{slate['ask_dim']}; "
              f"Anchor: {slate['exploit']['min_invest']/10000:.1f}w vs {slate['explore']['min_invest']/10000:.0f}w]")
    arms = _generate_candidates(slate)
    if verbose:
        print(f"│  LLM 实时生成 {len(arms)} 个候选追问臂, MAB 打分:")
    best = None
    for arm in arms:
        s = _score_arm(arm, slate)
        arm["_scores"] = s
        if verbose:
            print(f"│     [{arm['id']}] total={s['total']:.3f} "
                  f"(empathy={s['empathy']:.2f}, precision={s['precision']:.2f}, "
                  f"concise={s['conciseness']:.2f})  «{arm['_intent']}»")
        if best is None or s["total"] > best["_scores"]["total"]:
            best = arm
    if verbose:
        print(f"│  ✅ MAB 选中: {best['id']} «{best['_intent']}»")
        print("└─ 完成: 高情商话术已生成")
    return best
