"""客户文本回复的 NLU 槽位抽取 Prompt。"""

VERSION = "v1.0.2026-04-19"

NLU_SYSTEM = """你是金融对话系统的 NLU 模块, 任务是把客户的自然语言回复
抽取成结构化槽位, 输出严格 JSON。不要返回任何解释, 只返回 JSON 对象。"""


NLU_PROMPT = """客户回复: "{text}"

请抽取以下结构化槽位, 输出 JSON:
{{
  "annual_budget":   <客户明示或暗示的年投资金额元, 无则 null>,
  "loss_aversion":   <是否表达了'保本/别亏/安全'等诉求, bool>,
  "term_preference": <客户明示的持有年限, 无则 null>,
  "other_concerns":  [<其他表达的顾虑字符串列表>]
}}

注意:
- "2 万" → 20000, "两万五" → 25000, "1.5 个 w" → 15000
- "别亏钱/保本/安全/稳一点" → loss_aversion = true
- 未出现的字段明确返回 null 或空数组"""
