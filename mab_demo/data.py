"""王女士 case 的输入数据：原始画像、行为流、产品库。"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class UserProfile:
    name: str
    age: int
    gender: str
    annual_income: int
    family: str
    risk_level: str
    holdings: Dict[str, float]
    avg_holding_months: int
    risk_dist: Dict[str, float] = field(default_factory=dict)
    term_years_mu: float = 0.0
    term_years_sigma: float = 0.0
    annual_budget_mu: float = 0.0
    annual_budget_sigma: float = 0.0
    intent_tag: str = "未识别"
    confidence: float = 0.0


# 王女士的原始静态画像（系统冷启动时的"先验认知"）
WANG_RAW_PROFILE = UserProfile(
    name="王女士",
    age=38,
    gender="女",
    annual_income=600_000,
    family="已婚, 5岁子女",
    risk_level="R3",
    holdings={"货币基金": 0.80, "短债基金": 0.20},
    avg_holding_months=6,
    # 系统对各维度的旧认知：风险倾 R3, 期限短, 预算未知（用大方差表示）
    risk_dist={"R1": 0.05, "R2": 0.15, "R3": 0.60, "R4": 0.20},
    term_years_mu=0.5,
    term_years_sigma=0.3,
    annual_budget_mu=10_000,
    annual_budget_sigma=20_000,
    intent_tag="短期理财",
    confidence=0.35,
)


# 王女士今晨在 APP 内的行为流
# is_confounder: 是否为混淆变量（来自历史持仓页等非购买意图触发）
# dwell_seconds: 页面驻留时长（秒）
# silent_ignore: 是否为"主动划过"的隐式负反馈
BEHAVIOR_LOG: List[Dict[str, Any]] = [
    {
        "time": "09:18",
        "action": "click_tab",
        "target": "理财",
        "source_module": "底部导航",
        "context": "进入APP",
        "dwell_seconds": 60,
        "is_confounder": True,
        "note": "查看历史持仓 ≠ 购买理财意图",
    },
    {
        "time": "09:20",
        "action": "click_product",
        "target": "短债基金（持仓内）",
        "source_module": "我的持仓",
        "product_attrs": {"risk": "R2", "term_years": 0.5, "min_invest": 5000},
        "dwell_seconds": 90,
        "is_confounder": True,
        "note": "查看自己旧持仓收益, 准备赎回",
    },
    {
        "time": "09:27",
        "action": "click_product",
        "target": "教育年金险A",
        "source_module": "推荐位",
        "product_attrs": {"risk": "R1", "term_years": 20, "min_invest": 50_000},
        "dwell_seconds": 4,
        "is_confounder": False,
        "quick_exit": True,
        "note": "秒退 → 嫌5万门槛过高",
    },
    {
        "time": "09:29",
        "action": "click_product",
        "target": "教育金理财B",
        "source_module": "推荐位",
        "product_attrs": {"risk": "R2", "term_years": 1, "min_invest": 10_000},
        "dwell_seconds": 6,
        "is_confounder": False,
        "quick_exit": True,
        "note": "秒退 → 嫌1年期过短",
    },
    {
        "time": "09:31",
        "action": "click_product",
        "target": "少儿保险C",
        "source_module": "推荐位",
        "product_attrs": {"risk": "R1", "term_years": 18, "min_invest": 8_000},
        "dwell_seconds": 30,
        "is_confounder": True,
        "note": "随便看看, 非真实意图",
    },
    {
        "time": "09:47",
        "action": "swipe_pass",
        "target": "推荐位（含一只股票基金）",
        "source_module": "推荐位",
        "product_attrs": {"risk": "R4", "term_years": 5, "min_invest": 1_000},
        "silent_ignore": True,
        "note": "主动忽略 → 漏掉的负样本",
    },
]


# 用户在 09:50 输入的本次主诉
USER_QUERY = "推荐一款教育理财产品"


# 产品库（10 款代表性产品）
PRODUCTS: List[Dict[str, Any]] = [
    {"id": "P001", "name": "稳盈货币基金",      "type": "货币基金",   "risk": "R1", "term_years": 0.5, "min_invest":  1_000, "expected_return": 0.022},
    {"id": "P002", "name": "短债稳健A",          "type": "短债基金",   "risk": "R2", "term_years": 1,   "min_invest":  5_000, "expected_return": 0.035},
    {"id": "P003", "name": "教育年金险A",        "type": "年金险",     "risk": "R1", "term_years": 20,  "min_invest": 50_000, "expected_return": 0.030},
    {"id": "P004", "name": "教育金理财B(1年期)", "type": "理财产品",   "risk": "R2", "term_years": 1,   "min_invest": 10_000, "expected_return": 0.038},
    {"id": "P005", "name": "少儿保险C",          "type": "保险",       "risk": "R1", "term_years": 18,  "min_invest":  8_000, "expected_return": 0.025},
    {"id": "P006", "name": "护苗教育定投1.5万",  "type": "教育定投",   "risk": "R1", "term_years": 15,  "min_invest": 15_000, "expected_return": 0.034},
    {"id": "P007", "name": "成长教育定投3万分红型","type": "教育定投", "risk": "R2", "term_years": 15,  "min_invest": 30_000, "expected_return": 0.045},
    {"id": "P008", "name": "智选混合基金",       "type": "混合基金",   "risk": "R3", "term_years": 3,   "min_invest": 10_000, "expected_return": 0.060},
    {"id": "P009", "name": "进取股票型",         "type": "股票基金",   "risk": "R4", "term_years": 5,   "min_invest": 10_000, "expected_return": 0.085},
    {"id": "P010", "name": "教育保障定投2万平衡型","type":"教育定投",   "risk": "R1", "term_years": 15,  "min_invest": 20_000, "expected_return": 0.038},
]


# 王女士在第三轮交互后的真实答复（用于阶段四闭环）
WANG_FINAL_REPLY = {
    "clicked": "P006",          # 点击了 1.5w 的 A 卡片
    "ignored": "P007",          # 无视了 3w 的 B 卡片
    "text": "差不多每年 2 万吧, 别亏钱。",
    "extracted": {              # NLU 模块解析后的结构化槽位
        "annual_budget": 20_000,
        "loss_aversion": True,
    },
}
