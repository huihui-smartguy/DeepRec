"""主入口: 一键演示 MAB 全链路。
用法: cd mab_demo && python run_demo.py
"""
from data import WANG_RAW_PROFILE
from orchestrator import run_pipeline
from profile_compare import render


def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█       金融场景 MAB 多臂老虎机 · 王女士 case 全链路演示          █")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n[原始画像] (系统冷启动认知, 与真实需求严重错位):")
    p = WANG_RAW_PROFILE
    print(f"   {p.name}, {p.age}岁, 年入 {p.annual_income/10000:.0f}万, {p.family}")
    print(f"   稳态风险测评: {p.risk_level}  历史持仓: {p.holdings}  历史持有期: {p.avg_holding_months} 月")
    print(f"   系统旧意图标签: '{p.intent_tag}' (置信度仅 {p.confidence:.2f})")

    result = run_pipeline(p, verbose=True)
    render(p, result["enhanced_profile"], target_product_id="P010")

    print("\n[结论] MAB 四阶段流水线把客户的'画像-意图冲突'与'兴趣漂移'全部消解,")
    print("       从一个迷雾画像收敛到 R1·15年·2万 的精准画像, 演示完毕。\n")


if __name__ == "__main__":
    main()
