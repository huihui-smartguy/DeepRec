"""LLM JSON 输出的严格 Schema: 防止大模型幻觉导致后续管道炸掉。

设计原则: 轻量级, 纯标准库实现, 不强制引入 pydantic 依赖。
如项目中已有 pydantic 可替换成 BaseModel(推荐生产环境用 pydantic v2)。
"""
from typing import Any, Dict, List


class SchemaError(ValueError):
    """LLM 输出不符合预期 schema。"""
    pass


def _require(obj: Dict[str, Any], key: str, path: str = ""):
    if key not in obj:
        raise SchemaError(f"缺少必需字段: {path + key}")
    return obj[key]


def _check_type(v: Any, expected: type, path: str):
    if not isinstance(v, expected):
        raise SchemaError(f"字段 {path} 类型错误: 期望 {expected.__name__}, 实得 {type(v).__name__}")


def _check_range(v: float, lo: float, hi: float, path: str):
    if not (lo <= v <= hi):
        raise SchemaError(f"字段 {path} 取值越界: {v} 不在 [{lo}, {hi}]")


# ============ LLM 先验分布输出 ============ #
def validate_llm_prior(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    期望格式:
    {
      "cot_trace": ["step1", "step2", ...],
      "priors": {
        "term_years":    {"mu": float, "sigma": float},
        "annual_budget": {"mu": int,   "sigma": int},
        "risk": {"R1": float, "R2": float, "R3": float, "R4": float}
      }
    }
    """
    _check_type(raw, dict, "$")
    trace = _require(raw, "cot_trace", "$.")
    _check_type(trace, list, "$.cot_trace")
    if len(trace) < 1:
        raise SchemaError("cot_trace 至少包含 1 条推理")

    priors = _require(raw, "priors", "$.")
    _check_type(priors, dict, "$.priors")

    # term_years
    ty = _require(priors, "term_years", "$.priors.")
    ty_mu = float(_require(ty, "mu", "$.priors.term_years."))
    ty_sigma = float(_require(ty, "sigma", "$.priors.term_years."))
    _check_range(ty_mu, 0.1, 40.0, "$.priors.term_years.mu")
    _check_range(ty_sigma, 0.01, 20.0, "$.priors.term_years.sigma")

    # annual_budget
    ab = _require(priors, "annual_budget", "$.priors.")
    ab_mu = float(_require(ab, "mu", "$.priors.annual_budget."))
    ab_sigma = float(_require(ab, "sigma", "$.priors.annual_budget."))
    _check_range(ab_mu, 500, 1_000_000, "$.priors.annual_budget.mu")
    _check_range(ab_sigma, 100, 500_000, "$.priors.annual_budget.sigma")

    # risk
    risk = _require(priors, "risk", "$.priors.")
    _check_type(risk, dict, "$.priors.risk")
    for k in ("R1", "R2", "R3", "R4"):
        v = float(_require(risk, k, f"$.priors.risk."))
        _check_range(v, 0.0, 1.0, f"$.priors.risk.{k}")
    total = sum(float(risk[k]) for k in ("R1", "R2", "R3", "R4"))
    if abs(total - 1.0) > 0.05:
        # 允许小范围漂移, 自动归一化
        pass

    # 归一化 risk 分布
    risk_norm = {k: float(risk[k]) / total for k in ("R1", "R2", "R3", "R4")}

    return {
        "cot_trace": [str(s) for s in trace],
        "priors": {
            "term_years": {"mu": ty_mu, "sigma": ty_sigma},
            "annual_budget": {"mu": ab_mu, "sigma": ab_sigma},
            "risk": risk_norm,
        },
    }


# ============ 追问话术生成输出 ============ #
def validate_ask_arms(raw: Any, min_arms: int = 3) -> List[Dict[str, Any]]:
    """
    期望格式: [{"id": str, "text": str, "intent": str}, ...]
    """
    if isinstance(raw, dict) and "arms" in raw:
        raw = raw["arms"]
    _check_type(raw, list, "$")
    if len(raw) < min_arms:
        raise SchemaError(f"候选臂数量不足: 期望至少 {min_arms}, 实得 {len(raw)}")
    arms = []
    for i, item in enumerate(raw):
        _check_type(item, dict, f"$[{i}]")
        arms.append({
            "id": str(_require(item, "id", f"$[{i}].")),
            "text": str(_require(item, "text", f"$[{i}].")),
            "_intent": str(item.get("intent", item.get("_intent", "未标注"))),
        })
    return arms


# ============ NLU 槽位抽取输出 ============ #
def validate_nlu_slots(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    期望格式:
    {
      "annual_budget": int | null,
      "loss_aversion": bool,
      "term_preference": int | null,
      "other_concerns": [str, ...]
    }
    """
    _check_type(raw, dict, "$")
    ab = raw.get("annual_budget")
    if ab is not None:
        ab = int(ab)
        _check_range(ab, 100, 10_000_000, "$.annual_budget")
    la = bool(raw.get("loss_aversion", False))
    tp = raw.get("term_preference")
    if tp is not None:
        tp = int(tp)
        _check_range(tp, 0, 50, "$.term_preference")
    oc = raw.get("other_concerns", [])
    if not isinstance(oc, list):
        oc = []
    return {
        "annual_budget": ab,
        "loss_aversion": la,
        "term_preference": tp,
        "other_concerns": [str(x) for x in oc],
    }


# ============ 因果混淆分类器输出 ============ #
def validate_causal_label(raw: Dict[str, Any]) -> Dict[str, Any]:
    _check_type(raw, dict, "$")
    return {
        "is_confounder": bool(_require(raw, "is_confounder", "$.")),
        "reason": str(raw.get("reason", "")),
    }
