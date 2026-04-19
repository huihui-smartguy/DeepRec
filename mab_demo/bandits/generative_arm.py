"""阶段三·算法2：大模型动态生臂 (Generative Arm Bandit)

工业化版本:
  - 候选话术生成: 主路径调真实 LLM(高温度 0.8 保证多样性), 降级走 mock
  - 打分 RM:     可插拔 (rule_based / llm_judge / external_rm), 详见 clients/reward_model.py
"""
import logging
from typing import Any, Dict, List

from clients.llm_client import llm_client
from clients.reward_model import reward_model
from config import CONFIG
from prompts.generative_arm import ARM_PROMPT, ARM_SYSTEM, VERSION
from schemas import SchemaError, validate_ask_arms

log = logging.getLogger("mab_demo.generative_arm")


# -------------------------- 真实 LLM 生成候选臂 -------------------------- #
def _generate_candidates_real(slate: Dict[str, Any], n: int = 4) -> List[Dict[str, Any]]:
    a, b = slate["exploit"], slate["explore"]
    prompt = ARM_PROMPT.format(
        ask_dim=slate["ask_dim"],
        a_amt=a["min_invest"],
        b_amt=b["min_invest"],
        user_name="王女士",
        user_desc="38岁, 有 5 岁子女, 年入 60 万, 已婚家庭",
        recent_behavior="刚才点击过一款 5 万起投的教育年金险 A 秒退(嫌门槛高),"
                        " 又点过一款 1 年期教育理财 B 秒退(嫌期限短)",
        a_name=a["name"],
        a_risk=a["risk"],
        a_term=a["term_years"],
        b_name=b["name"],
        b_risk=b["risk"],
        b_term=b["term_years"],
        n=n,
    )
    raw = llm_client.chat_json(
        prompt=prompt,
        system=ARM_SYSTEM,
        temperature=0.8,  # 高温度保证多样性
    )
    if raw is None:
        raise RuntimeError("LLM returned None")
    arms = validate_ask_arms(raw, min_arms=3)
    return arms


# -------------------------- Mock 降级 -------------------------- #
def _generate_candidates_mock(slate: Dict[str, Any]) -> List[Dict[str, Any]]:
    a, b = slate["exploit"], slate["explore"]
    return [
        {"id": "ARM_1", "text": "请填写预算: ___ 元/年", "_intent": "传统硬表单"},
        {"id": "ARM_2",
         "text": f"您是想每年投 {a['min_invest']/10000:.1f} 万还是 {b['min_invest']/10000:.0f} 万?",
         "_intent": "纯数字探询"},
        {"id": "ARM_3",
         "text": (
             f"王女士, 为护航宝宝未来 15 年的教育, 我为您优选了这两款专属定投。"
             f"刚才注意到那款 5 万起投的可能单笔资金占用偏大。"
             f"为了兼顾您的日常品质生活, 咱们是更倾向于每年 "
             f"{a['min_invest']/10000:.1f} 万的轻松节奏, "
             f"还是稍微增加预算到 {b['min_invest']/10000:.0f} 万去争取长期丰厚回报呢?"
         ),
         "_intent": "高情商 + 锚点 + 痛点回应"},
        {"id": "ARM_4",
         "text": f"为孩子的未来加油! 您打算每年投多少呢? 越多收益越好哦~",
         "_intent": "营销话术"},
    ]


def _generate_candidates(slate: Dict[str, Any]):
    source = "LLM"
    if llm_client.enabled:
        try:
            return _generate_candidates_real(slate), source
        except (SchemaError, Exception) as e:
            if not CONFIG.fallback_on_fail:
                raise
            source = f"Mock (LLM 降级: {type(e).__name__})"
            log.warning(f"Generative arm fallback to mock: {e}")
    else:
        source = "Mock (LLM 未启用)"
    return _generate_candidates_mock(slate), source


# -------------------------- 对外接口 -------------------------- #
def run(slate: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("\n┌─[阶段三·算法2] 大模型动态生臂 (Generative Arms) · 启动 ──────────")
        print(f"│  Prompt 版本: {VERSION}  |  奖励模型: {type(reward_model).__name__}")
        print(f"│  指令(机器格式): [Action: Target_{slate['ask_dim']}; "
              f"Anchor: {slate['exploit']['min_invest']/10000:.1f}w vs {slate['explore']['min_invest']/10000:.0f}w]")

    arms, source = _generate_candidates(slate)
    if verbose:
        print(f"│  生成来源: {source}  →  {len(arms)} 条候选追问臂, RM 打分:")

    best = None
    for arm in arms:
        s = reward_model.score(arm["text"], slate)
        arm["_scores"] = s
        if verbose:
            intent = arm.get("_intent", "未标注")
            print(f"│     [{arm['id']:<6}] total={s['total']:.3f} "
                  f"(empathy={s['empathy']:.2f}, precision={s['precision']:.2f}, "
                  f"concise={s['conciseness']:.2f})  «{intent}»")
        if best is None or s["total"] > best["_scores"]["total"]:
            best = arm

    if verbose:
        intent = best.get("_intent", "未标注")
        print(f"│  ✅ MAB 选中: {best['id']} «{intent}»")
        print("└─ 完成: 高情商话术已生成")
    return best
