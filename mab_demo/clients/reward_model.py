"""可插拔奖励模型 (Reward Model): 给追问话术打分。

生产环境: 推荐训练一个 DeBERTa/Qwen-0.5B 的专用 RM, P99 < 10ms。
Demo 默认用规则模型, 方便无依赖验证; 也提供 LLM-as-Judge 兜底(仅离线评估)。

接口: score(text, slate_ctx) -> {"empathy": float, "precision": float,
                                "conciseness": float, "total": float}
"""
import logging
from typing import Any, Dict

from config import CONFIG

log = logging.getLogger("mab_demo.rm")


class RuleBasedRewardModel:
    """规则版 RM: 零依赖, 确定性。"""

    EMPATHY_KEYWORDS = ["王女士", "宝宝", "孩子", "品质生活", "护航", "节奏"]

    def score(self, text: str, slate_ctx: Dict[str, Any]) -> Dict[str, float]:
        a_amt = f"{slate_ctx['exploit']['min_invest']/10000:.1f}"
        b_amt = f"{slate_ctx['explore']['min_invest']/10000:.0f}"
        empathy = sum(1 for k in self.EMPATHY_KEYWORDS if k in text) / len(self.EMPATHY_KEYWORDS)
        precision = (0.5 if a_amt in text else 0.0) + (0.5 if b_amt in text else 0.0)
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
        return {"empathy": empathy, "precision": precision,
                "conciseness": conciseness, "total": total}


class LLMJudgeRewardModel:
    """LLM-as-Judge: 仅供离线评估或极低 QPS 场景, 不建议上线路径。"""

    JUDGE_PROMPT = """你是一位金融投顾 UX 评审专家。请对以下追问话术从三个维度打分:
- empathy (0-1): 是否体现情绪价值(提到客户、家人、痛点)
- precision (0-1): 是否锚定具体数字(A/B 两个金额)
- conciseness (0-1): 简洁度(100-180 字最佳)

话术: "{text}"
A 卡片金额: {a_amt}  B 卡片金额: {b_amt}

输出 JSON: {{"empathy": <0-1>, "precision": <0-1>, "conciseness": <0-1>, "total": <加权和>}}
加权公式: total = 0.5*empathy + 0.3*precision + 0.2*conciseness
"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.fallback = RuleBasedRewardModel()

    def score(self, text: str, slate_ctx: Dict[str, Any]) -> Dict[str, float]:
        if not self.llm_client.enabled:
            return self.fallback.score(text, slate_ctx)
        prompt = self.JUDGE_PROMPT.format(
            text=text,
            a_amt=slate_ctx["exploit"]["min_invest"],
            b_amt=slate_ctx["explore"]["min_invest"],
        )
        raw = self.llm_client.chat_json(prompt, temperature=0.0)
        if raw is None:
            return self.fallback.score(text, slate_ctx)
        try:
            return {
                "empathy": float(raw.get("empathy", 0)),
                "precision": float(raw.get("precision", 0)),
                "conciseness": float(raw.get("conciseness", 0)),
                "total": float(raw.get("total", 0)),
            }
        except (TypeError, ValueError) as e:
            log.warning(f"LLM judge output invalid, fallback: {e}")
            return self.fallback.score(text, slate_ctx)


class ExternalRewardModel:
    """HTTP 外部 RM: 对接专用的 RM 推理服务。"""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.fallback = RuleBasedRewardModel()

    def score(self, text: str, slate_ctx: Dict[str, Any]) -> Dict[str, float]:
        try:
            import urllib.request
            import json
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps({"text": text, "context": slate_ctx}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            log.warning(f"External RM call failed, fallback: {e}")
            return self.fallback.score(text, slate_ctx)


def get_reward_model():
    rm_type = CONFIG.rm_type
    if rm_type == "llm_judge":
        from clients.llm_client import llm_client
        return LLMJudgeRewardModel(llm_client)
    if rm_type == "external_rm" and CONFIG.rm_endpoint:
        return ExternalRewardModel(CONFIG.rm_endpoint)
    return RuleBasedRewardModel()


reward_model = get_reward_model()
