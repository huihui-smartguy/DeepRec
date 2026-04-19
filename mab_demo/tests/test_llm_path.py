"""验证 LLM 路径: 用 monkey-patch 伪造 LLM 返回, 测试整条链路能正确走 LLM 分支。
用法: python3 tests/test_llm_path.py
"""
import os
import sys

# 伪造环境变量让 llm_client 进入 enabled 状态
os.environ["MAB_LLM_API_KEY"] = "fake-key-for-testing"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients import llm_client as _llm_client_module  # noqa: E402
from bandits import llm_prior, generative_arm, causal_debias  # noqa: E402
import nlu  # noqa: E402
from data import WANG_RAW_PROFILE, BEHAVIOR_LOG, USER_QUERY, PRODUCTS  # noqa: E402
from schemas import validate_llm_prior, validate_ask_arms, validate_nlu_slots  # noqa: E402


class FakeLLMClient:
    """伪造 LLM 客户端, 按 prompt 中的关键字返回不同 mock 响应。"""
    enabled = True

    def chat_json(self, prompt, system=None, model=None, temperature=None, retries=None, force_json=True):
        if "先验分布" in prompt or "priors" in prompt:
            return {
                "cot_trace": [
                    "主诉判定为教育金规划",
                    "5岁子女→15年周期",
                    "60万年收入→2-3万预算",
                    "教育金场景→R1主导"
                ],
                "priors": {
                    "term_years":    {"mu": 15.0, "sigma": 2.0},
                    "annual_budget": {"mu": 25000, "sigma": 7000},
                    "risk": {"R1": 0.6, "R2": 0.3, "R3": 0.08, "R4": 0.02}
                }
            }
        if "追问话术" in prompt or "ARM_1" in prompt:
            return [
                {"id": "ARM_1", "text": "请输入预算: ___", "intent": "传统表单"},
                {"id": "ARM_2", "text": "您想投 1.5 万还是 3 万?", "intent": "纯数字"},
                {"id": "ARM_3", "text": "王女士, 考虑到宝宝未来教育, 我们提供了两款定投方案, "
                                        "兼顾您品质生活的节奏, 建议在 1.5 万和 3 万间做选择, "
                                        "您倾向轻松的 1.5 万还是 3 万?", "intent": "高情商"},
                {"id": "ARM_4", "text": "孩子未来加油, 您多投多受益!", "intent": "营销"},
            ]
        if "annual_budget" in prompt:
            return {
                "annual_budget": 20000,
                "loss_aversion": True,
                "term_preference": None,
                "other_concerns": []
            }
        if "is_confounder" in prompt:
            return {"is_confounder": False, "reason": "看似因果"}
        return {}


# 把全局 llm_client 替换为 fake
_llm_client_module.llm_client = FakeLLMClient()
# 下游模块已经 `from clients.llm_client import llm_client` 了, 需要同步替换
import bandits.llm_prior as lp_mod
import bandits.generative_arm as ga_mod
import bandits.causal_debias as cd_mod
import nlu as nlu_mod
lp_mod.llm_client = FakeLLMClient()
ga_mod.llm_client = FakeLLMClient()
cd_mod.__dict__  # noop
nlu_mod.llm_client = FakeLLMClient()


def test_llm_prior_path():
    print("\n[TEST] bandits.llm_prior  真 LLM 路径")
    priors = llm_prior.run(WANG_RAW_PROFILE, USER_QUERY, verbose=False)
    assert priors["term_years"]["mu"] == 15.0, "LLM 返回的 term_mu 应为 15.0"
    assert priors["annual_budget"]["mu"] == 25000, "LLM 返回的 budget_mu 应为 25000"
    assert abs(sum(priors["risk"].values()) - 1.0) < 1e-6, "risk 概率和必须为 1"
    print(f"  ✔ term_mu={priors['term_years']['mu']}, budget_mu={priors['annual_budget']['mu']}")
    print(f"  ✔ risk={priors['risk']}")


def test_generative_arm_path():
    print("\n[TEST] bandits.generative_arm  真 LLM 路径")
    slate = {
        "exploit": PRODUCTS[5],  # P006 1.5w
        "explore": PRODUCTS[6],  # P007 3w
        "ask_dim": "budget",
        "eig": 1.0,
    }
    best = generative_arm.run(slate, verbose=False)
    assert best["id"] in ("ARM_1", "ARM_2", "ARM_3", "ARM_4")
    print(f"  ✔ 选中 {best['id']}: {best['text'][:40]}...")


def test_nlu_path():
    print("\n[TEST] nlu.extract_slots  真 LLM 路径")
    slots = nlu.extract_slots("差不多每年 2 万吧, 别亏钱。", verbose=False)
    assert slots["annual_budget"] == 20000
    assert slots["loss_aversion"] is True
    print(f"  ✔ slots = {slots}")


def test_schema_rejection():
    print("\n[TEST] schemas 拒绝非法 LLM 输出")
    from schemas import SchemaError
    bad_cases = [
        {"missing": "cot_trace"},
        {"cot_trace": [], "priors": {"term_years": {"mu": -1, "sigma": 0},
                                      "annual_budget": {"mu": 0, "sigma": 0},
                                      "risk": {"R1": 2, "R2": 0, "R3": 0, "R4": 0}}},
    ]
    for case in bad_cases:
        try:
            validate_llm_prior(case)
            print(f"  ✗ FAILED 未抛异常: {case}")
            return False
        except (SchemaError, KeyError, ValueError, TypeError):
            pass
    print(f"  ✔ 所有 {len(bad_cases)} 条非法输出都被正确拦截")


def main():
    test_llm_prior_path()
    test_generative_arm_path()
    test_nlu_path()
    test_schema_rejection()
    print("\n✅ 所有 LLM 路径测试通过")


if __name__ == "__main__":
    main()
