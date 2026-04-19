"""阶段一：因果去偏老虎机 (Causal Debiasing MAB)

工业化版本:
  - 默认走规则引擎 (低延迟 <5ms, 生产主路径)
  - 可选 LLM 兜底: 规则未命中的事件若有歧义信号 → 调 LLM 做二分类
  - 开启方式: 环境变量 MAB_CAUSAL_USE_LLM=1 (默认关闭, 避免延迟灾难)

剥离 Position Bias 与 History Bias, 防止 MAB 被旧数据污染。
"""
import logging
from typing import Any, Dict, List, Tuple

from config import CONFIG

log = logging.getLogger("mab_demo.causal")


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


def _llm_classify_confounder(event: Dict[str, Any]) -> Tuple[bool, str]:
    """LLM 兜底分类: 规则未命中但事件可疑时调用。"""
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


def run(behavior_log: List[Dict[str, Any]], verbose: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回 (清洗后的真因果事件, 被剥离的混淆事件)。"""
    clean: List[Dict[str, Any]] = []
    confounders: List[Dict[str, Any]] = []
    if verbose:
        mode = "规则 + LLM 兜底" if CONFIG.causal_use_llm else "纯规则(低延迟)"
        print(f"\n┌─[阶段一·算法1] 因果去偏 MAB · 启动 · 模式={mode} ─────────────────")
    for ev in behavior_log:
        label = "Causal"
        reason = "通过因果安检"
        # 规则引擎优先
        for cond, lab, why in CAUSAL_RULES:
            if cond(ev):
                label, reason = lab, why
                break
        # 规则未命中, 且启用了 LLM 兜底, 且事件有 ambiguity 信号 → 调 LLM
        if label == "Causal" and CONFIG.causal_use_llm and ev.get("note"):
            is_conf, llm_reason = _llm_classify_confounder(ev)
            if is_conf:
                label, reason = "Confounder", f"LLM 兜底: {llm_reason}"

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
