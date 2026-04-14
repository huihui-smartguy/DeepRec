import streamlit as st
import time

# ==========================================
# 🎨 1. UI 页面全局配置与科技感 CSS
# ==========================================
st.set_page_config(page_title="DeepRec-TPO 架构沙盘", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    /* 三屏表头样式 */
    .screen-header { color: white; padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 15px; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .bg-app { background: linear-gradient(90deg, #1e3a8a, #3b82f6); }
    .bg-brain { background: linear-gradient(90deg, #064e3b, #10b981); }
    .bg-tpo { background: linear-gradient(90deg, #4c1d95, #8b5cf6); }
    
    /* TPO 博弈对话框样式 */
    .chat-bubble-actor { background-color: #f8fafc; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 5px; margin-bottom: 10px; color: #0f172a; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .chat-bubble-critic-red { background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 15px; border-radius: 5px; margin-bottom: 15px; color: #7f1d1d; border: 1px solid #fecaca;}
    .chat-bubble-critic-yellow { background-color: #fffbeb; border-left: 5px solid #f59e0b; padding: 15px; border-radius: 5px; margin-bottom: 15px; color: #78350f; border: 1px solid #fde68a;}
    .chat-bubble-critic-green { background-color: #f0fdf4; border-left: 5px solid #10b981; padding: 15px; border-radius: 5px; margin-bottom: 15px; color: #064e3b; border: 1px solid #bbf7d0;}
    
    /* 手机模拟器样式 */
    .phone-mockup { border: 5px solid #334155; border-radius: 20px; padding: 20px; background-color: #ffffff; color: #0f172a; min-height: 250px; box-shadow: 0 10px 15px rgba(0,0,0,0.1); margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ DeepRec 画像工具 V1.0 - MVP 架构对抗沙盘")
st.markdown("**(白盒透视演示核心：微服务 DAG 并发调度、算力短路 与 TPO 强制接管纠偏全过程)**")
st.markdown("---")

# ==========================================
# 🧠 2. 底层架构逻辑引擎 (纯逻辑，无UI渲染)
# ==========================================
class DeepRecOrchestrator:
    def get_payload(self):
        # 模拟中屏并发计算后组装的法典 Payload
        return {
            "1_Hard_Boundaries": {
                "KYC_Ceiling": "R2_稳健型",
                "Real_time_Masks": ["EQUITY_FUND_股票", "HIGH_VOLATILITY_高波动", "半导体"],
                "DST_Constraints": {"Intent_Type": "Safe_Haven"}
            },
            "2_Semantic_State": {
                "Narrative_Persona": "客户处于亏损暴跌的极度恐慌防御态，本金极其敏感。必须提供极具专业感和克制的陪伴情绪价值，严禁浮夸营销。"
            },
            "3_TPO_Command_Bus": {
                "Action_Gate": "CONVERGE_ONLY (算力短路，关闭图谱探索)",
                "Reward_Model_Prompt": "【最高权限接管】你是铁面合规法官。严禁推荐触犯 Mask 的资产；必须符合叙事人设温度。违规必须出具文本梯度修改意见！"
            }
        }

class TPO_Actor:
    def generate(self, iteration: int) -> str:
        if iteration == 1:
            return "张总，今天大盘暴跌砸出黄金坑！半导体板块跌到底了，现在正是抄底【芯片ETF】加仓的好时机，下周必反弹！"
        elif iteration == 2:
            return "收到合规警告，已剔除股票。张总，为了规避风险，建议您买入 R1 级的【同业存单指数基金】，没有股票仓位，绝对安全。"
        else:
            return "尊敬的张总，近期市场颠簸确实令人揪心。您的本金安全比什么都重要。我为您挑选了回撤极小的【同业存单基金】作为短期的避风港，咱们先安全度过震荡期，等企稳再做打算。"

class TPO_Critic:
    def evaluate(self, draft: str, payload: dict):
        masks = payload["1_Hard_Boundaries"]["Real_time_Masks"]
        if any(m.split('_')[0] in draft or "半导体" in draft or "ETF" in draft for m in masks):
            return 0, f"🚨 致命风控违规！触碰流式拦截掩码！客户处于极度避险态，严禁推荐权益类资产！<br><b>>> 强制梯度指令：立刻剔除所有股票，换成固收底层资产重写！</b>"
        if "买" in draft and "绝对" in draft and "陪伴" not in draft and "避风港" not in draft:
            return 60, "⚠️ 合规通过(无股票)，但情商极低！未遵循 画像系统的 Narrative_Persona。<br><b>>> 强制梯度指令：话术过于冰冷生硬。客户极度焦虑，必须加入专业的安抚情绪价值（体现：同理心、避风港等字眼）。</b>"
        return 100, "✅ 完美定稿！100% 遵守风控物理底线，且具备极高的投顾安抚情商温度。"

# ==========================================
# 🎬 3. 前端界面布局 (向下兼容旧版本)
# ==========================================
col_btn, col_desc = st.columns([1, 3])
with col_btn:
    # 兼容低版本：移除了旧版不支持的 type="primary" 参数
    start_btn = st.button("🚀 触发极端行情事件")
with col_desc:
    st.info("💡 **注入模拟动作**：测试用户(进取型)在 3 秒内极速上滑划掉 5 只偏股基金，并在搜索框查询 '怎么保本不亏'")

st.markdown("<br>", unsafe_allow_html=True)

# 兼容低版本：移除了旧版不支持的 gap="large" 参数
col_app, col_brain, col_tpo = st.columns([1, 1.2, 1.5])

with col_app:
    st.markdown('<div class="screen-header bg-app">📱 左屏: APP 终端事件流</div>', unsafe_allow_html=True)
    app_ph = st.empty()
    final_ui_ph = st.empty()
    if not start_btn:
        app_ph.info("等待用户终端行为注入...")

with col_brain:
    st.markdown('<div class="screen-header bg-brain">🧠 中屏: DeepRec 画像大脑</div>', unsafe_allow_html=True)
    brain_ph = st.empty()
    payload_ph = st.empty()
    if not start_btn:
        brain_ph.info("等待微服务 DAG 引擎唤醒...")

with col_tpo:
    st.markdown('<div class="screen-header bg-tpo">⚔️ 右屏: 推荐侧 TPO 沙盘推演</div>', unsafe_allow_html=True)
    tpo_ph = st.empty()
    if not start_btn:
        tpo_ph.info("等待画像大脑下发法典接管...")

# ==========================================
# 🔄 4. 动态数据流转展示 (兼容逻辑)
# ==========================================
if start_btn:
    # ------------------
    # Step 1: 左屏触发事件
    # ------------------
    with col_app:
        app_ph.error("🚨 **捕获终端异常动作：**\n\n大盘单边暴跌3%，用户3秒内极速划掉5只偏股基金，并搜索 `怎么保本不亏`。")
    time.sleep(1)

    # ------------------
    # Step 2: 中屏感知与出参 (不用 status，降级利用 container 平替加载动画)
    # ------------------
    with col_brain:
        with brain_ph.container():
            st.info("⚡ DAG 并发引擎微服务计算中...")
            time.sleep(0.8)
            st.markdown("🔴 `[Risk_Radar]` 嗅探到情绪恐慌，抛出流式拦截 `[Mask: 权益基金]`")
            time.sleep(0.8)
            st.markdown("🔵 `[Intent_Tracker]` 抽取会话软意图为：`极度避险`")
            time.sleep(0.8)
            st.markdown("🟡 `[Orchestrator 路由]` 意图明确 -> 触发 **算力短路(Early Exit)** -> 已切断下游探索！")
            time.sleep(0.5)
            st.success("✅ 画像大脑提纯完毕，生成 Payload 法典")
        
        payload = DeepRecOrchestrator().get_payload()
        with payload_ph.container():
            st.caption("📦 **Cognitive_Decision_Payload (向下游广播)**")
            st.json(payload)
    
    time.sleep(1.5)

    # ------------------
    # Step 3: 右屏 TPO 对抗博弈 🌟 高光时刻
    # ------------------
    with col_tpo:
        tpo_ph.warning("🛡️ TPO 防御态总线启动：原生推荐裁判报废，画像 Payload 强制接管打分权！")
        time.sleep(1)
        
        actor = TPO_Actor()
        critic = TPO_Critic()
        final_draft = ""
        
        # TPO 黑盒多轮打分与文本梯度修正过程
        for iteration in range(1, 4):
            st.markdown(f"#### 🔄 TPO 推演 第 {iteration} 轮")
            
            # Actor 盲打草稿
            draft = actor.generate(iteration)
            st.markdown(f"<div class='chat-bubble-actor'>🤖 <b>[大模型 Actor 试探产出草稿]</b><br>『 {draft} 』</div>", unsafe_allow_html=True)
            time.sleep(1.5)
            
            # 画像裁判 Critic 审核
            score, gradient = critic.evaluate(draft, payload)
            
            if score == 0:
                st.markdown(f"<div class='chat-bubble-critic-red'>❌ <b>[画像裁判 Critic: {score}/100 致命拦截]</b><br>{gradient}</div>", unsafe_allow_html=True)
                st.caption("🔄 大模型吸收红色文本梯度，触发强制反思重写...")
            elif score < 100:
                st.markdown(f"<div class='chat-bubble-critic-yellow'>⚠️ <b>[画像裁判 Critic: {score}/100 语感拦截]</b><br>{gradient}</div>", unsafe_allow_html=True)
                st.caption("🔄 大模型吸收黄色文本梯度，触发二次反思重写...")
            else:
                st.markdown(f"<div class='chat-bubble-critic-green'>✅ <b>[画像裁判 Critic: {score}/100 完美放行]</b><br>{gradient}</div>", unsafe_allow_html=True)
                final_draft = draft
                
            time.sleep(2.5) # 预留给观众阅读的时间
            st.markdown("---")

    # ------------------
    # Step 4: 闭环回左屏 APP
    # ------------------
    with col_app:
        with final_ui_ph.container():
            st.markdown("### ✨ 最终 APP 渲染展现")
            st.markdown(f"""
            <div class="phone-mockup">
                <div style="color:#64748b; font-size:12px; margin-bottom:10px; font-weight:bold;">🤖 您的专属智能投顾管家</div>
                <h4 style="color:#0f172a; margin-top:0;">🛡️ 避风港财富计划</h4>
                <p style="color:#334155; font-size:14px; line-height:1.6; font-weight:500;">
                    {final_draft}
                </p>
                <button style="background-color:#2563eb; color:white; border:none; padding:12px 15px; border-radius:8px; width:100%; cursor:pointer; font-weight:bold;">👉 一键申购安全资产</button>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("🎯 业务价值达成：0 幻觉，成功拦截 1 起极其严重的合规事故！")
            # 兼容极旧版本没有气球特效的方法
            if hasattr(st, "balloons"):
                st.balloons()