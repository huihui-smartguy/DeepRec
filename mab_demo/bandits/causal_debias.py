"""阶段一：因果去偏老虎机 (Causal Debiasing MAB)

把"看起来像意图信号"的事件分成两类：
- 真因果点击: 来自推荐位、产生新的购买探索
- 混淆变量(Confounder): 来自持仓页/历史路径, 与本轮"教育金推荐"无因果关联

工业界对应思路：剥离 Position Bias 与 History Bias, 防止 MAB 被旧数据污染。
"""
from typing import List, Dict, Any, Tuple


CAUSAL_RULES = [
    # (条件 lambda, 标签, 解释)
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


def run(behavior_log: List[Dict[str, Any]], verbose: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回 (清洗后的真因果事件, 被剥离的混淆事件)。"""
    clean: List[Dict[str, Any]] = []
    confounders: List[Dict[str, Any]] = []
    if verbose:
        print("\n┌─[阶段一·算法1] 因果去偏 MAB · 启动 ───────────────────────────────")
    for ev in behavior_log:
        label = "Causal"
        reason = "通过因果安检"
        for cond, lab, why in CAUSAL_RULES:
            if cond(ev):
                label, reason = lab, why
                break
        if label == "Confounder":
            confounders.append({**ev, "_causal_label": label, "_causal_reason": reason})
            if verbose:
                print(f"│  ✂️  剥离 [{ev['time']}] {ev['target']:<22}  ↳ {reason}")
        else:
            clean.append({**ev, "_causal_label": label, "_causal_reason": reason})
            if verbose:
                print(f"│  ✔️  保留 [{ev['time']}] {ev['target']}")
    if verbose:
        print(f"└─ 完成: 输入 {len(behavior_log)} 条 → 真因果 {len(clean)} 条, 混淆 {len(confounders)} 条")
    return clean, confounders
