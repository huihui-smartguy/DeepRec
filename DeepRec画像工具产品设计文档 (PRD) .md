### 1. 项目的 MVP（最小可行产品）定义

* **业务愿景：** 为金融行业（及强合规重决策行业）的推荐系统提供可热插拔的“实时认知大脑与风控统帅”，彻底解决极端场景下“合规易穿透、AI易产生越界幻觉”的痛点。

* **🎯 MVP（Phase 1 / V1.0）核心边界：**

    * **战略聚焦 \(In\-Scope\)：** 砍掉业务发散与跨界探索，将所有研发资源 All\-in 于构建\*\*“绝对防线（流式阻断）”**与**“TPO内审接管（Defensive TPO 防御态）”\*\*。

    * **功能包含：** `Orchestrator 编排总控（含🌟TPO认知接管总线🌟）` + `Risk_Radar 风控小Skill` + `Intent_Tracker 意图小Skill` + `Semantic_Context 语义认知（降级为离线预计算）`。

    * **暂缓上线 \(Out\-of\-Scope\)：** `KG_Explorer (外部图谱探路)` 延期至 V2.0；毫秒级在线 Target Attention 延期至 V1.5。

### 2. 核心功能点与优先级清单 \(MoSCoW 规范\)

|**所属微服务模块**|**核心功能点描述 \(Feature\)**|**研发优先级**|**MVP 状态**|
|---|---|---|---|
|**Orchestrator\(总控\)**|**DAG 并发编排引擎**：多线程极速唤起底层小 Skill，无阻塞汇聚特征。|**P0 \(Must\)**|✅ 包含|
|👑 **Orchestrator\(总控\)**|👑 **TPO 认知接管总线 \(TPO Director\)**：将画像的风控特征翻译为大模型裁判法典（Prompt规则），打包下发，**强制接管下游推荐系统 TPO 的内部博弈**。|**P0 \(Must\)**|✅ **包含**|
|**Orchestrator\(总控\)**|**算力动态路由 \(Gating\)**：计算信息熵，本期强制执行 Early Exit（算力短路），限制发散。|**P0 \(Must\)**|✅ 包含|
|**Risk\_Radar\(风控\)**|**流式负反馈防线**：基于行为流计算情绪突变，秒级抛出 `[硬拦截 Mask]`。|**P0 \(Must\)**|✅ 包含|
|**Risk\_Radar\(风控\)**|**静态底线对齐**：加载底层关系型数据库中的 KYC/反洗钱等不可逾越红线。|**P0 \(Must\)**|✅ 包含|
|**Intent\_Tracker\(意图\)**|**Session 状态机**：多轮 DST 抽取结构化约束槽位与生成式情绪软意图。|**P0 \(Must\)**|✅ 包含|
|**Semantic\_Context\(认知\)**|**离线画像提取**：毫秒级检索离线预计算生成的“叙事型画像基调”。|**P1 \(Should\)**|✅ 包含|
|**KG\_Explorer\(图谱\)**|**外部逻辑图谱推理**：基于知识图谱输出安全的跨界扩展白名单节点。|**P2 \(Could\)**|❌ 暂缓|

### 3. 系统技术栈选择 \(Tech Stack\)

保障核心关键路径 **P99 &lt; 50ms** 的极致性能选型：

* **中枢编排层：** `Golang` 或 `Python (FastAPI + LangGraph)`，专职负责微服务 DAG 极速调度与 Payload 组装。

* **流式感知层：** `Apache Flink`（计算行为事件窗口与驻留一阶导数） + `Kafka`（高并发接入端流）。

* **极速状态层：** `Redis Cluster`（存储毫秒级工作记忆、瞬时 Mask，自带 TTL 阅后即焚） + `Milvus`（存储与检索文本形式的叙事画像）。

* **大模型底座层：** 部署 `DeepSeek-V3` / `Qwen-Max` 等开源模型 + `vLLM` 推理加速框架（确保金融数据 100% 私有化不出域）。

### 4. 核心系统业务流转 \(User &amp; System Workflow\)

*\(泳道图流转逻辑，限时 &lt; 50ms\)*

1. **\[T=0ms 触发\]** APP 终端发生动作。网关同步调起 Orchestrator API；异步打入 Kafka 供 Flink 并发消费。

2. **\[T=5ms 并发收集\]** Orchestrator 启动 DAG，**起 3 个协程并行调用**：向 `Risk_Radar` 查缓存 Mask/KYC，向 `Intent_Tracker` 调取 DST 意图，向 `Semantic_Context` 提纯画像。

3. **\[T=30ms 算力短路与总线接管\]** Orchestrator 判定当前为防守态（切断图谱探索）。**TPO 认知接管总线** 启动，将收集到的所有约束，翻译成推荐端 TPO 能读懂的“铁面内审员规则”。

4. **\[T=40ms 组装出栈\]** 统一组装 `Cognitive_Decision_Payload` JSON 协议包，作为最高指令返回给下游推荐系统，本次会话生命周期结束。
