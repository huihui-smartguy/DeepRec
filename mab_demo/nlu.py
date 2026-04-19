"""客户自然语言回复 → 结构化槽位抽取。

LLM 为主、正则为兜底。生产环境建议:
 - 高 QPS 路径: 用小参数量模型(Qwen-1.5B)部署成专用 NLU 服务
 - 低 QPS 路径: 直接用主 LLM
"""
import logging
import re
from typing import Any, Dict

from clients.llm_client import llm_client
from config import CONFIG
from prompts.nlu import NLU_PROMPT, NLU_SYSTEM
from schemas import SchemaError, validate_nlu_slots

log = logging.getLogger("mab_demo.nlu")


_UNIT_MAP = {"万": 10_000, "w": 10_000, "W": 10_000, "千": 1_000}


def _extract_slots_regex(text: str) -> Dict[str, Any]:
    """正则兜底: 无 LLM 时也能跑通主流程。"""
    slots: Dict[str, Any] = {
        "annual_budget": None,
        "loss_aversion": False,
        "term_preference": None,
        "other_concerns": [],
    }
    # 金额: "2 万", "1.5 万", "2w" 等
    m = re.search(r"([\d.]+)\s*([万wW千])", text)
    if m:
        val = float(m.group(1))
        unit = _UNIT_MAP.get(m.group(2), 1)
        slots["annual_budget"] = int(val * unit)
    else:
        m2 = re.search(r"(\d{4,7})", text)
        if m2:
            slots["annual_budget"] = int(m2.group(1))

    # 损失厌恶
    if any(k in text for k in ("别亏", "保本", "安全", "不亏", "稳一点", "稳点")):
        slots["loss_aversion"] = True

    # 期限
    m3 = re.search(r"(\d+)\s*年", text)
    if m3:
        slots["term_preference"] = int(m3.group(1))

    return slots


def extract_slots(text: str, verbose: bool = False) -> Dict[str, Any]:
    """主入口: 优先调 LLM, 失败则正则兜底。"""
    if llm_client.enabled:
        raw = llm_client.chat_json(
            NLU_PROMPT.format(text=text),
            system=NLU_SYSTEM,
            temperature=0.0,
        )
        if raw is not None:
            try:
                slots = validate_nlu_slots(raw)
                if verbose:
                    print(f"│  [NLU·LLM] 抽取槽位: {slots}")
                return slots
            except SchemaError as e:
                log.warning(f"NLU schema invalid, fallback to regex: {e}")

    # Fallback
    slots = _extract_slots_regex(text)
    if verbose:
        tag = "LLM 未启用" if not llm_client.enabled else "LLM 失败"
        print(f"│  [NLU·Regex兜底·{tag}] 抽取槽位: {slots}")
    return slots
