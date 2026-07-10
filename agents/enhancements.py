"""
Agent 智能增强模块
在 Agent 基础推理之上，叠加 7 个智能维度的增强能力:

1. 用户画像自适应检测 — 根据用户语言特征推断法律知识水平/年龄/紧迫度，动态调整回复风格
2. 信息完整性追踪 — 追踪对话中已提供/缺失的关键信息，计算完整度百分比，驱动主动追问
3. 思维链可视化 — 从 Agent 推理过程中提取决策链路，向用户展示工具选择理由
4. Agent 自反思与质量校验 — 生成回答后自检法律准确性/完整性/语气，不合格时自动修正
5. 置信度标注 — 评估回答的置信度（高/中/低），低置信度时建议咨询专业律师
6. 知识边界感知 — 检测问题是否超出 Agent 能力范围，诚实声明不确定性
7. 维权进度追踪 — 跨对话状态机，记录用户维权走到哪一步，推荐下一步行动
8. 对话摘要导出 — 将多轮对话压缩为结构化纪要文档

设计原则:
- 每个增强模块独立，可单独开关
- 尽量用轻量级规则匹配，减少额外 LLM 调用（只有自反思和摘要需要额外 LLM 调用）
- 所有模块返回结构化数据，由 base.py 决定如何呈现给用户
"""
import json
import os
import re
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import ARK_API_KEY, ARK_BASE_URL, MODEL_ID, WORD_REPORTS_DIR


# ============================================================
# 1. 用户画像自适应检测
# ============================================================

# 法律知识水平判定关键词
_LEGAL_EXPERT_KEYWORDS = [
    "法条", "条款", "第几条", "诉讼时效", "举证责任", "抗辩", "违约金",
    "格式条款", "缔约过失", "欺诈", "退一赔三", "退一赔十", "惩罚性赔偿",
    "消法", "食安法", "民法典", "管辖权", "小额诉讼",
]
_LEGAL_NOVICE_KEYWORDS = [
    "怎么办", "不知道", "第一次", "不懂", "能退吗", "能赔吗",
    "合法吗", "可以吗", "有没有用", "来得及吗", "找谁",
]
_URGENCY_KEYWORDS = [
    "急", "马上", "立刻", "今天", "明天", "过期", "来不及",
    "快", "紧急", "最后期限",
]


class UserProfileDetector:
    """
    用户画像自适应检测

    通过分析用户输入的语言特征，推断:
    - 法律知识水平: expert / intermediate / novice
    - 紧迫程度: urgent / normal
    - 用户类型: consumer / merchant / student

    根据画像动态调整 Agent 的回复风格:
    - expert: 使用精确法律术语，引用具体法条编号
    - novice: 通俗解释，避免术语，多用类比
    - urgent: 先给结论，再解释原因
    """

    @staticmethod
    def detect(user_input: str, history: list = None) -> dict:
        """
        检测用户画像

        返回:
            {
                "legal_level": "expert" / "intermediate" / "novice",
                "urgency": "urgent" / "normal",
                "user_type": "consumer" / "merchant" / "student",
                "style_hint": "给Agent的风格调整提示",
            }
        """
        text = user_input or ""
        # 也参考历史对话
        if history:
            for item in history[-3:]:  # 最近3轮
                if isinstance(item, dict):
                    text += " " + str(item.get("content", ""))

        # 法律知识水平
        expert_hits = sum(1 for kw in _LEGAL_EXPERT_KEYWORDS if kw in text)
        novice_hits = sum(1 for kw in _LEGAL_NOVICE_KEYWORDS if kw in text)

        if expert_hits >= 2:
            legal_level = "expert"
            style_hint = "用户具备一定法律知识，可使用专业术语和法条编号，回答可以更精炼"
        elif novice_hits >= 2:
            legal_level = "novice"
            style_hint = "用户法律知识较少，请用通俗易懂的语言解释，避免过多术语，多用生活化比喻"
        else:
            legal_level = "intermediate"
            style_hint = "用户有一定基础但不精通，适度使用法律术语并附带解释"

        # 紧迫程度
        urgency_hits = sum(1 for kw in _URGENCY_KEYWORDS if kw in text)
        urgency = "urgent" if urgency_hits >= 1 else "normal"
        if urgency == "urgent":
            style_hint += "；用户比较着急，请先给出结论和核心建议，再展开解释"

        # 用户类型
        user_type = "consumer"
        if any(kw in text for kw in ["我是商家", "我是卖家", "经营", "店铺"]):
            user_type = "merchant"
        elif any(kw in text for kw in ["作业", "课程", "论文", "研究", "答辩"]):
            user_type = "student"

        return {
            "legal_level": legal_level,
            "urgency": urgency,
            "user_type": user_type,
            "style_hint": style_hint,
        }


# ============================================================
# 2. 信息完整性追踪
# ============================================================

# 投诉信所需关键信息字段
_COMPLAINT_FIELDS = {
    "complainant_name": {"label": "投诉人姓名", "keywords": ["我叫", "我是", "姓名", "本人"]},
    "contact": {"label": "联系方式", "keywords": ["电话", "手机", "微信", "联系"]},
    "merchant_name": {"label": "商家名称", "keywords": ["商家", "店铺", "超市", "店", "平台", "永辉", "美团", "淘宝", "京东", "拼多多"]},
    "purchase_time": {"label": "购买时间", "keywords": ["今天", "昨天", "上周", "月", "日", "号买的", "购买"]},
    "product_name": {"label": "商品/服务名称", "keywords": ["买的", "购买的", "商品", "食品", "手机", "鞋子", "卡", "服务"]},
    "dispute_detail": {"label": "纠纷经过", "keywords": ["过期", "假货", "质量", "坏了", "不退", "拒绝", "欺诈", "虚假"]},
    "demand": {"label": "具体诉求", "keywords": ["退款", "赔偿", "退货", "道歉", "换货", "投诉"]},
}

# 条款审查所需字段
_REVIEW_FIELDS = {
    "clause_content": {"label": "条款内容", "keywords": ["条款", "合同", "协议", "写着", "规定", "约定"]},
    "context": {"label": "使用场景", "keywords": ["会员卡", "健身", "租房", "电商", "预付", "消费"]},
}


class CompletenessTracker:
    """
    信息完整性追踪

    追踪对话中已提供和缺失的关键信息字段，
    计算信息完整度百分比，生成主动追问建议。
    """

    @staticmethod
    def track(history: list, current_input: str, agent_type: str = "qa") -> dict:
        """
        追踪信息完整性

        参数:
            history: 对话历史
            current_input: 当前用户输入
            agent_type: agent类型 (qa/complaint/review)

        返回:
            {
                "completeness": 85,  # 完整度百分比
                "provided_fields": ["商家名称", "纠纷经过"],
                "missing_fields": ["投诉人姓名", "联系方式"],
                "should_ask": True/False,  # 是否应该主动追问
                "ask_hint": "请补充以下信息...",
            }
        """
        # 合并所有对话文本
        all_text = current_input or ""
        if history:
            for item in history:
                if isinstance(item, dict):
                    content = item.get("content", "")
                    if item.get("role") == "user":
                        all_text += " " + str(content)

        # 选择字段集
        if agent_type == "complaint":
            fields = _COMPLAINT_FIELDS
        elif agent_type == "review":
            fields = _REVIEW_FIELDS
        else:
            # QA模式不需要严格追踪，返回基本状态
            return {
                "completeness": 100,
                "provided_fields": [],
                "missing_fields": [],
                "should_ask": False,
                "ask_hint": "",
            }

        provided = []
        missing = []

        for field_key, field_info in fields.items():
            label = field_info["label"]
            keywords = field_info["keywords"]
            if any(kw in all_text for kw in keywords):
                provided.append(label)
            else:
                missing.append(label)

        total = len(fields)
        completeness = int(len(provided) / total * 100) if total > 0 else 100

        # 缺失关键字段时建议追问
        should_ask = len(missing) > 0 and completeness < 70

        if should_ask:
            ask_hint = f"信息完整度 {completeness}%，还缺少: {'、'.join(missing[:3])}"
        else:
            ask_hint = ""

        return {
            "completeness": completeness,
            "provided_fields": provided,
            "missing_fields": missing,
            "should_ask": should_ask,
            "ask_hint": ask_hint,
        }


# ============================================================
# 3. 思维链可视化
# ============================================================

# 工具选择理由模板
_TOOL_REASONING = {
    "search_law": "用户咨询法律问题，需要检索相关法条作为依据",
    "search_case": "需要查找类似案例，为用户提供参考",
    "estimate_compensation": "用户关心赔偿金额，需要计算法定赔偿",
    "generate_evidence_checklist": "用户准备维权，需要知道收集哪些证据",
    "plan_rights_path": "用户需要维权路径，规划分步骤行动方案",
    "merchant_tactics_response": "用户遇到商家推诿，需要法律反驳依据",
    "rights_deadline_reminder": "涉及维权时效，需要计算剩余天数",
    "multi_platform_complaint": "用户需要向不同平台投诉，生成对应格式文书",
    "package_evidence": "用户需要打包维权材料",
    "trap_warning": "用户想了解消费陷阱，查询行业预警",
    "check_merchant_reputation": "用户想了解商家信誉，查询历史投诉",
    "create_complaint_report": "信息已齐备，生成正式投诉信文档",
    "create_review_report": "审查完成，生成审查报告文档",
    "handoff_to_agent": "检测到用户需求变化，切换到更合适的智能体",
}


class ReasoningChainExtractor:
    """
    思维链可视化

    从 Agent 的执行结果中提取推理过程:
    - Agent 调用了哪些工具，每个工具的选择理由
    - Agent 的推理步骤
    - 最终回答的依据

    生成可读的思维链展示文本，让用户看到 Agent 的"思考过程"。
    """

    @staticmethod
    def extract(result_messages: list, user_input: str, emotion: str = "neutral") -> str:
        """
        从 Agent 执行结果中提取思维链

        参数:
            result_messages: agent.invoke() 返回的 messages 列表
            user_input: 用户原始输入
            emotion: 检测到的情绪

        返回:
            思维链展示文本（Markdown格式）
        """
        steps = []

        # 步骤1: 意图理解
        steps.append(("理解意图", f"分析用户输入「{user_input[:50]}...」的核心诉求"))

        # 步骤2: 情绪感知（如果有）
        if emotion != "neutral":
            emotion_labels = {"anger": "愤怒", "anxiety": "焦虑", "sad": "委屈"}
            steps.append(("情绪感知", f"检测到用户情绪为{emotion_labels.get(emotion, emotion)}，优先安抚"))

        # 步骤3: 工具调用推理
        tool_calls_seen = []
        for msg in result_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "")
                    if tool_name and tool_name not in [t[0] for t in tool_calls_seen]:
                        reason = _TOOL_REASONING.get(tool_name, f"调用工具 {tool_name}")
                        tool_calls_seen.append((tool_name, reason))

        for tool_name, reason in tool_calls_seen:
            tool_label = {
                "search_law": "检索法律条文",
                "search_case": "检索相关案例",
                "estimate_compensation": "预估赔偿金额",
                "generate_evidence_checklist": "生成证据清单",
                "plan_rights_path": "规划维权路径",
                "merchant_tactics_response": "分析商家话术",
                "rights_deadline_reminder": "提醒维权时效",
                "multi_platform_complaint": "生成平台投诉文书",
                "package_evidence": "打包维权材料",
                "trap_warning": "查询消费陷阱",
                "check_merchant_reputation": "查询商家信誉",
                "create_complaint_report": "生成投诉信文档",
                "create_review_report": "生成审查报告",
                "handoff_to_agent": "切换智能体",
            }.get(tool_name, tool_name)
            steps.append((f"调用工具: {tool_label}", reason))

        # 步骤4: 综合回答
        if tool_calls_seen:
            steps.append(("综合分析", "整合工具返回结果和法律依据，组织回答"))

        # 生成思维链文本
        lines = ["> **Agent 思维链**\n>"]
        for i, (action, reason) in enumerate(steps, 1):
            lines.append(f"> {i}. {action} — {reason}")
        return "\n".join(lines)


# ============================================================
# 4. Agent 自反思与质量校验
# ============================================================

_REFLECTION_SYSTEM_PROMPT = """你是一个消费维权回答质量审查员。请对以下AI回答进行质量评估。

评估维度（每项1-5分）:
1. 法律准确性: 法条引用是否正确、适用是否恰当
2. 完整性: 是否充分回答了用户问题，有无遗漏关键信息
3. 语气适当性: 语气是否专业、温和、不机械
4. 可操作性: 建议是否具体、可执行
5. 风险提示: 是否包含必要的免责声明和风险提示

请输出JSON格式:
{"pass": true/false, "score": 总分, "issues": ["问题1", "问题2"], "suggestion": "改进建议（如有）"}

如果总分>=16分且无严重问题，pass为true。否则pass为false。
只输出JSON，不要其他内容。"""


class SelfReflector:
    """
    Agent 自反思与质量校验

    在 Agent 生成回答后，调用 LLM 对回答进行质量自检:
    - 法律准确性: 法条引用是否正确
    - 完整性: 是否充分回答了用户问题
    - 语气适当性: 语气是否专业温和
    - 可操作性: 建议是否具体可执行
    - 风险提示: 是否包含免责声明

    如果质量不达标（总分<16或有严重问题），追加改进提示。
    """

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=MODEL_ID,
                api_key=ARK_API_KEY,
                base_url=ARK_BASE_URL,
                temperature=0.0,
            )
        return self._llm

    def reflect(self, user_question: str, agent_answer: str) -> dict:
        """
        对 Agent 回答进行自反思

        参数:
            user_question: 用户的原始问题
            agent_answer: Agent 生成的回答

        返回:
            {
                "pass": True/False,
                "score": 18,
                "issues": ["缺少风险提示"],
                "suggestion": "建议在末尾添加免责声明",
                "quality_label": "优质" / "合格" / "需改进",
            }
        """
        llm = self._get_llm()

        review_input = f"用户问题: {user_question}\n\nAI回答:\n{agent_answer[:2000]}"

        try:
            response = llm.invoke([
                SystemMessage(content=_REFLECTION_SYSTEM_PROMPT),
                HumanMessage(content=review_input),
            ])
            result_text = response.content.strip()

            # 尝试解析JSON
            # 清理可能的markdown代码块标记
            result_text = re.sub(r"^```json\s*", "", result_text)
            result_text = re.sub(r"\s*```$", "", result_text)

            result = json.loads(result_text)

            score = result.get("score", 0)
            if score >= 18:
                quality_label = "优质"
            elif score >= 16:
                quality_label = "合格"
            else:
                quality_label = "需改进"

            return {
                "pass": result.get("pass", True),
                "score": score,
                "issues": result.get("issues", []),
                "suggestion": result.get("suggestion", ""),
                "quality_label": quality_label,
            }
        except (json.JSONDecodeError, Exception) as e:
            # 解析失败时默认通过，不阻塞回答
            print(f"[SelfReflector] 自反思失败(已降级): {type(e).__name__}: {e}")
            return {
                "pass": True,
                "score": 16,
                "issues": [],
                "suggestion": "",
                "quality_label": "合格",
            }


# ============================================================
# 5. 置信度标注 + 6. 知识边界感知
# ============================================================

# 知识边界关键词 — 这些问题可能超出 Agent 能力
_BOUNDARY_RISK_KEYWORDS = [
    "刑事案件", "离婚", "继承", "房产纠纷", "劳动仲裁", "交通事故",
    "医疗事故", "税务", "专利", "商标", "著作权",
    "具体判几年", "帮我打官司", "代写诉状", "出庭",
]

# 高置信度信号 — 回答中有具体法条引用
_HIGH_CONFIDENCE_SIGNALS = [
    "第", "条", "消费者权益保护法", "食品安全法", "产品质量法",
    "电子商务法", "民法典",
]


class ConfidenceEvaluator:
    """
    置信度标注 + 知识边界感知

    评估 Agent 回答的置信度:
    - 高: 回答有明确法条依据，问题在能力范围内
    - 中: 回答有一定依据，但部分内容需要用户自行判断
    - 低: 问题可能超出能力范围，或回答缺乏法条支撑

    同时检测知识边界:
    - 如果用户问题涉及刑事案件、离婚等超出消费维权范围的话题，
      Agent 应诚实声明这超出了自己的能力范围。
    """

    @staticmethod
    def evaluate(user_question: str, agent_answer: str, has_law_reference: bool) -> dict:
        """
        评估回答置信度和知识边界

        参数:
            user_question: 用户原始问题
            agent_answer: Agent 回答
            has_law_reference: 回答中是否引用了法条

        返回:
            {
                "confidence": "high" / "medium" / "low",
                "confidence_label": "高" / "中" / "低",
                "confidence_emoji": "🟢" / "🟡" / "🔴",
                "reason": "置信度评估理由",
                "boundary_warning": "知识边界警告（如果超出范围）",
                "is_out_of_scope": True/False,
            }
        """
        question = user_question or ""
        answer = agent_answer or ""

        # 知识边界检测
        is_out_of_scope = any(kw in question for kw in _BOUNDARY_RISK_KEYWORDS)
        boundary_warning = ""
        if is_out_of_scope:
            boundary_warning = (
                "⚠️ 您的问题可能超出了消费维权智能助手的专业范围。"
                "本系统专注于消费者权益保护领域的法律咨询、投诉信起草和条款审查。"
                "建议您咨询专业律师或拨打12348法律服务热线获取更准确的法律帮助。"
            )

        # 置信度评估
        law_signals = sum(1 for sig in _HIGH_CONFIDENCE_SIGNALS if sig in answer)
        answer_length = len(answer)

        if is_out_of_scope:
            confidence = "low"
            confidence_label = "低"
            confidence_emoji = "🔴"
            reason = "问题可能超出系统专业范围"
        elif has_law_reference and law_signals >= 2 and answer_length > 100:
            confidence = "high"
            confidence_label = "高"
            confidence_emoji = "🟢"
            reason = "回答引用了具体法条，有明确法律依据"
        elif law_signals >= 1 or answer_length > 50:
            confidence = "medium"
            confidence_label = "中"
            confidence_emoji = "🟡"
            reason = "回答有一定依据，但部分内容建议进一步确认"
        else:
            confidence = "low"
            confidence_label = "低"
            confidence_emoji = "🔴"
            reason = "回答缺乏明确法律依据，建议咨询专业律师"

        return {
            "confidence": confidence,
            "confidence_label": confidence_label,
            "confidence_emoji": confidence_emoji,
            "reason": reason,
            "boundary_warning": boundary_warning,
            "is_out_of_scope": is_out_of_scope,
        }


# ============================================================
# 7. 维权进度追踪
# ============================================================

# 维权阶段定义
_RIGHTS_STAGES = [
    {"stage": 1, "key": "consult", "label": "咨询了解", "desc": "了解自己的权益和维权方式"},
    {"stage": 2, "key": "negotiate", "label": "与商家协商", "desc": "直接联系商家沟通解决"},
    {"stage": 3, "key": "platform", "label": "平台投诉", "desc": "通过电商平台或12315投诉"},
    {"stage": 4, "key": "mediate", "label": "消协调解", "desc": "向消费者协会申请调解"},
    {"stage": 5, "key": "litigate", "label": "提起诉讼", "desc": "向人民法院提起民事诉讼"},
    {"stage": 6, "key": "resolved", "label": "维权完成", "desc": "问题已解决"},
]

# 阶段关键词
_STAGE_KEYWORDS = {
    "consult": ["咨询", "了解", "问一下", "怎么办", "能维权吗", "合法吗"],
    "negotiate": ["协商", "沟通", "找了商家", "联系了", "商家说", "商家拒绝", "商家同意"],
    "platform": ["12315", "平台投诉", "已经投诉", "投诉了", "举报"],
    "mediate": ["消协", "调解", "消费者协会"],
    "litigate": ["起诉", "法院", "诉讼", "打官司", "立案"],
    "resolved": ["解决了", "退了", "赔了", "已退款", "已赔偿", "搞定了"],
}


class CaseProgressTracker:
    """
    维权进度追踪

    跨对话追踪用户的维权进度，通过分析对话内容判断用户当前处于哪个维权阶段，
    推荐下一步行动。以状态机的形式呈现维权全流程。
    """

    @staticmethod
    def track(history: list, current_input: str) -> dict:
        """
        追踪维权进度

        参数:
            history: 对话历史
            current_input: 当前输入

        返回:
            {
                "current_stage": 2,
                "current_label": "与商家协商",
                "stages_completed": [1],
                "next_action": "建议向12315平台投诉",
                "progress_bar": "█░░░░░ 17%",
                "full_path": [...],  # 完整路径
            }
        """
        # 合并所有用户消息
        all_text = current_input or ""
        if history:
            for item in history:
                if isinstance(item, dict) and item.get("role") == "user":
                    all_text += " " + str(item.get("content", ""))

        # 检测用户处于哪个阶段
        detected_stage = 1  # 默认在咨询阶段
        stages_hit = set()

        for stage in _RIGHTS_STAGES:
            keywords = _STAGE_KEYWORDS.get(stage["key"], [])
            if any(kw in all_text for kw in keywords):
                stages_hit.add(stage["stage"])
                if stage["stage"] > detected_stage:
                    detected_stage = stage["stage"]

        # 构建进度条
        total_stages = len(_RIGHTS_STAGES)
        progress_percent = int(detected_stage / total_stages * 100)
        filled = int(progress_percent / 100 * total_stages)
        progress_bar = "█" * filled + "░" * (total_stages - filled) + f" {progress_percent}%"

        # 推荐下一步
        if detected_stage < total_stages:
            next_stage = _RIGHTS_STAGES[detected_stage]  # 当前阶段的下一个
            next_action = f"下一步: {next_stage['label']} — {next_stage['desc']}"
        else:
            next_action = "维权流程已完成"

        # 构建完整路径（标记已完成/当前/未来）
        full_path = []
        for stage in _RIGHTS_STAGES:
            if stage["stage"] < detected_stage:
                status = "completed"
            elif stage["stage"] == detected_stage:
                status = "current"
            else:
                status = "upcoming"
            full_path.append({
                "stage": stage["stage"],
                "label": stage["label"],
                "desc": stage["desc"],
                "status": status,
            })

        return {
            "current_stage": detected_stage,
            "current_label": _RIGHTS_STAGES[detected_stage - 1]["label"],
            "stages_completed": sorted([s for s in stages_hit if s < detected_stage]),
            "next_action": next_action,
            "progress_bar": progress_bar,
            "full_path": full_path,
        }

    @staticmethod
    def format_progress(progress: dict) -> str:
        """将进度信息格式化为展示文本"""
        lines = ["> **维权进度追踪**\n>"]
        lines.append(f"> {progress['progress_bar']}\n>")
        lines.append(f"> 当前阶段: {progress['current_label']}\n>")
        lines.append(f"> {progress['next_action']}")
        return "\n".join(lines)


# ============================================================
# 8. 对话摘要导出
# ============================================================

_SUMMARY_SYSTEM_PROMPT = """你是一个对话摘要助手。请将以下消费维权咨询对话整理为结构化纪要。

输出格式:
## 咨询纪要

### 基本信息
- 咨询时间: （自动填写）
- 咨询类型: （法律问答/投诉信起草/条款审查）
- 涉及商家: （如有）

### 问题描述
（简要描述用户的消费纠纷情况）

### 法律依据
（列出对话中引用的相关法条）

### Agent建议
（列出Agent给出的维权建议和行动步骤）

### 关键结论
（核心结论摘要）

### 后续行动
（用户接下来应该做的事情）

请只输出纪要内容，不要添加额外说明。"""


class ConversationSummarizer:
    """
    对话摘要导出

    将多轮对话压缩为结构化纪要文档，包含:
    - 基本信息（时间、类型、涉及商家）
    - 问题描述
    - 法律依据
    - Agent建议
    - 关键结论
    - 后续行动建议

    可导出为文本文件保存。
    """

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=MODEL_ID,
                api_key=ARK_API_KEY,
                base_url=ARK_BASE_URL,
                temperature=0.3,
            )
        return self._llm

    def summarize(self, history: list) -> str:
        """
        生成对话摘要

        参数:
            history: Gradio对话历史

        返回:
            结构化纪要文本
        """
        if not history or len(history) == 0:
            return "暂无对话内容可摘要"

        # 拼接对话文本
        dialog_text = ""
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if isinstance(content, list):
                    # 处理多模态内容
                    parts = []
                    for block in content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, dict) and "text" in block:
                            parts.append(str(block["text"]))
                    content = "\n".join(parts)
                if role == "user":
                    dialog_text += f"用户: {content}\n\n"
                elif role == "assistant":
                    dialog_text += f"助手: {content[:500]}\n\n"

        if not dialog_text.strip():
            return "暂无对话内容可摘要"

        llm = self._get_llm()

        try:
            response = llm.invoke([
                SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n对话内容:\n{dialog_text[:4000]}"),
            ])
            return response.content
        except Exception as e:
            # 降级: 生成简单摘要
            return self._simple_summary(history)

    def _simple_summary(self, history: list) -> str:
        """LLM调用失败时的降级简单摘要"""
        lines = ["## 咨询纪要", "", f"### 基本信息", f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
        lines.append("### 对话概要")
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = str(item.get("content", ""))[:200]
                if role == "user":
                    lines.append(f"- 用户: {content}")
                elif role == "assistant":
                    lines.append(f"- 助手: {content[:100]}...")
        return "\n".join(lines)

    def save_summary(self, summary_text: str) -> str:
        """将摘要保存为文件"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"咨询纪要_{timestamp}.txt"
        filepath = os.path.join(WORD_REPORTS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(summary_text)
        return filepath
