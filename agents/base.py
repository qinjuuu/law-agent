"""
Agent 基类模块
统一处理流式输出 + 全维度数据库记录，确保回复实时到达前端
所有具体 Agent 继承此类，只需提供 llm / tools / system_prompt

支持:
- 真正的 LLM 流式输出: astream(stream_mode="messages") 逐字 yield
- 多轮对话记忆: 将 Gradio 传入的对话历史转换为 LangChain 消息列表
- 情绪感知与安抚: 检测用户负面情绪，先安抚再联动维权路径规划
- 用户画像自适应: 检测法律知识水平/紧迫度，动态调整回复风格
- 信息完整性追踪: 追踪缺失信息，驱动主动追问
- 思维链可视化: 展示Agent推理过程和工具选择理由
- Agent自反思: 生成后自检法律准确性/完整性/语气
- 置信度标注: 评估回答置信度，低置信度时建议咨询律师
- 知识边界感知: 检测问题是否超出能力范围，诚实声明
- 维权进度追踪: 跨对话状态机，记录维权走到哪一步
- 全流程数据库记录: 对话/工具/情绪/自反思/置信度/进度全部入库
"""
import asyncio
import time
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk


# ============================================================
# 工具名称到中文标签的映射
# ============================================================
_TOOL_LABELS = {
    "search_law": "检索法律条文",
    "search_case": "检索相关案例",
    "search_web": "联网搜索",
    "search_latest_regulation": "搜索最新法规",
    "search_merchant_info": "联网搜索商家信息",
    "create_complaint_report": "生成投诉信文档",
    "create_review_report": "生成审查报告",
    "create_file": "创建文件",
    "read_file": "读取文件",
    "delete_file": "删除文件",
    "zip_files": "压缩文件",
    "estimate_compensation": "预估赔偿金额",
    "generate_evidence_checklist": "生成证据清单",
    "plan_rights_path": "规划维权路径",
    "merchant_tactics_response": "分析商家话术",
    "rights_deadline_reminder": "提醒维权时效",
    "multi_platform_complaint": "生成平台投诉文书",
    "package_evidence": "打包维权材料",
    "trap_warning": "查询消费陷阱",
    "check_merchant_reputation": "查询商家信誉",
    "handoff_to_agent": "切换智能体",
    "get_rights_progress": "查询维权进度",
    "send_email": "发送邮件",
}


# ============================================================
# 情绪感知模块
# ============================================================

_ANGER_KEYWORDS = [
    "气死", "气炸", "气疯", "愤怒", "恼火", "火大", "可恶", "无良", "黑心",
    "奸商", "骗子", "坑人", "太坑", "坑爹", "不要脸", "无耻", "混蛋", "恶心",
    "投诉他", "举报他", "告他", "气不过", "忍不了", "受不了", "过分", "离谱",
]

_ANXIETY_KEYWORDS = [
    "怎么办", "怎么办啊", "着急", "急死", "害怕", "担心", "慌", "焦虑",
    "不知道该怎么办", "无助", "迷茫", "没经验", "第一次遇到", "不知道找谁",
    "能维权吗", "来得及吗", "还有用吗", "能退吗", "能赔吗",
]

_SAD_KEYWORDS = [
    "难过", "伤心", "委屈", "倒霉", "倒霉透顶", "郁闷", "郁闷死了", "心累",
    "太惨了", "好惨", "可怜", "心寒", "失望", "绝望", "崩溃",
]

_EMOTION_RESPONSES = {
    "anger": {
        "label": "愤怒",
        "emoji": "😤",
        "soothe": "完全理解您的心情，遇到这种情况确实让人气愤。您别着急，这种情况完全可以依法维权，我来帮您一步步解决。",
        "action_hint": "建议先深呼吸冷静一下，您的权益有法律保障。接下来我帮您规划维权路径，让法律替您说话。",
    },
    "anxiety": {
        "label": "焦虑",
        "emoji": "😰",
        "soothe": "别担心，遇到消费纠纷很常见，您来咨询就是迈出了正确的第一步。维权没有想象中那么复杂，我来帮您理清思路。",
        "action_hint": "这种情况完全可以维权，我帮您看看该怎么走，一步一步来，不用怕。",
    },
    "sad": {
        "label": "委屈",
        "emoji": "😢",
        "soothe": "遇到这种事确实很让人难受，但这不是您的错。消费维权是您的合法权利，我来帮您讨回公道。",
        "action_hint": "别灰心，法律会保护您的权益。我来帮您分析一下接下来该怎么做。",
    },
}


def _detect_emotion(text: str) -> str:
    if not text:
        return "neutral"
    anger_count = sum(1 for kw in _ANGER_KEYWORDS if kw in text)
    anxiety_count = sum(1 for kw in _ANXIETY_KEYWORDS if kw in text)
    sad_count = sum(1 for kw in _SAD_KEYWORDS if kw in text)
    max_count = max(anger_count, anxiety_count, sad_count)
    if max_count == 0:
        return "neutral"
    if anger_count == max_count:
        return "anger"
    elif anxiety_count == max_count:
        return "anxiety"
    else:
        return "sad"


def _get_emotion_keywords(text: str, emotion: str) -> list:
    """获取匹配到的情绪关键词"""
    if emotion == "anger":
        kws = _ANGER_KEYWORDS
    elif emotion == "anxiety":
        kws = _ANXIETY_KEYWORDS
    elif emotion == "sad":
        kws = _SAD_KEYWORDS
    else:
        return []
    return [kw for kw in kws if kw in text]


def _build_emotion_prefix(emotion: str) -> str:
    if emotion == "neutral":
        return ""
    resp = _EMOTION_RESPONSES.get(emotion)
    if not resp:
        return ""
    return f"> {resp['emoji']} 情绪感知: 检测到您可能感到{resp['label']}\n\n" \
           f"**{resp['soothe']}**\n\n" \
           f"---\n\n"


def _build_emotion_instruction(emotion: str) -> str:
    if emotion == "neutral":
        return ""
    resp = _EMOTION_RESPONSES.get(emotion)
    if not resp:
        return ""
    return f"""

【情绪感知指令】
系统检测到用户当前情绪为「{resp['label']}」，请在回答中注意以下事项:
1. 开头先用温和、共情的语气回应，让用户感受到被理解。可以这样开头: "{resp['soothe']}"
2. 必须调用 plan_rights_path 工具为用户规划维权路径，让用户知道接下来该怎么走
3. 如果条件允许，也调用 estimate_compensation 工具预估赔偿金额，给用户信心
4. 整体语气要积极、有力量，让用户感到维权有希望、有方向
5. 不要机械地罗列法条，要结合用户的情绪状态组织语言
"""


# ============================================================
# 对话历史转换
# ============================================================

def _content_to_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content)


def _gradio_history_to_messages(history):
    if not history:
        return []
    messages = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if not content:
                continue
            content_str = _content_to_str(content)
            if not content_str:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content_str))
            elif role == "assistant":
                clean = _clean_agent_output(content_str)
                if clean:
                    messages.append(AIMessage(content=clean))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg:
                messages.append(HumanMessage(content=_content_to_str(user_msg)))
            if bot_msg:
                clean = _clean_agent_output(_content_to_str(bot_msg))
                if clean:
                    messages.append(AIMessage(content=clean))
    return messages


def _clean_agent_output(text) -> str:
    if not text:
        return ""
    text = _content_to_str(text)
    if not text:
        return ""
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        if line.strip().startswith("> "):
            continue
        if line.strip() == "---":
            continue
        if line.strip().startswith("正在") and line.strip().endswith("..."):
            continue
        if "意图识别完成" in line:
            continue
        if "情绪感知" in line:
            continue
        if "正在接入" in line:
            continue
        if "Agent思维链" in line:
            continue
        if "置信度" in line and line.strip().startswith(">"):
            continue
        if "维权进度" in line and line.strip().startswith(">"):
            continue
        if "自反思" in line and line.strip().startswith(">"):
            continue
        if "信息完整度" in line and line.strip().startswith(">"):
            continue
        if "用户画像" in line and line.strip().startswith(">"):
            continue
        clean_lines.append(line)
    result = "\n".join(clean_lines).strip()
    return result


# ============================================================
# Agent 基类
# ============================================================

class BaseAgent:
    """
    Agent 基类

    子类需设置:
        self.llm           - ChatOpenAI 实例
        self.tools         - 工具列表
        self.system_prompt  - 系统提示词字符串
        self.agent_type     - agent类型 (qa/complaint/review)

    特性:
        - 真正的 LLM 流式输出: astream 逐字 yield
        - 多轮对话记忆: 自动将 Gradio history 转换为消息上下文
        - 情绪感知: 检测负面情绪时先安抚再联动维权路径
        - 用户画像自适应: 检测法律知识水平，动态调整回复风格
        - 信息完整性追踪: 追踪缺失信息，驱动主动追问
        - 思维链可视化: 展示Agent推理过程和工具选择理由
        - Agent自反思: 生成后自检质量，不合格时追加提示
        - 置信度标注: 高/中/低置信度 + 知识边界感知
        - 维权进度追踪: 跨对话状态机
        - 全流程数据库记录: 所有处理步骤自动入库
    """

    llm = None
    tools = []
    system_prompt = ""
    agent_type = "qa"

    async def chat(self, user_input: str, history: list = None):
        """
        流式问答对话（全增强版 + 真正流式输出 + 数据库记录）

        处理流程:
            1. 用户画像自适应检测
            2. 情绪感知
            3. 信息完整性追踪
            4. 构建 Agent
            5. astream 流式输出 — 逐字 yield LLM 生成的文本
            6. 思维链提取
            7. Agent自反思
            8. 置信度标注
            9. 维权进度追踪
            10. 组装最终输出（前缀 + 正文 + 后缀）
            11. 数据库记录全流程
        """
        from langchain.agents import create_agent
        from agents.enhancements import (
            UserProfileDetector,
            CompletenessTracker,
            ReasoningChainExtractor,
            SelfReflector,
            ConfidenceEvaluator,
            CaseProgressTracker,
        )

        # 数据库实例（非阻塞，失败不影响主流程）
        try:
            from database import db
        except Exception:
            db = None

        # 创建对话记录
        conversation_id = 0
        if db:
            try:
                import uuid
                session_id = str(uuid.uuid4())[:8]
                conversation_id = db.create_conversation(session_id, self.agent_type, user_input[:50])
            except Exception:
                pass

        # ========== 1. 用户画像自适应检测 ==========
        profile = UserProfileDetector.detect(user_input, history)
        if db and conversation_id:
            try:
                db.log_user_profile(conversation_id, 0, profile)
            except Exception:
                pass

        # ========== 2. 情绪感知 ==========
        emotion = _detect_emotion(user_input)
        emotion_prefix = _build_emotion_prefix(emotion)
        emotion_instruction = _build_emotion_instruction(emotion)

        if emotion != "neutral":
            emotion_kw = _get_emotion_keywords(user_input, emotion)
            resp = _EMOTION_RESPONSES.get(emotion, {})
            if db and conversation_id:
                try:
                    db.log_emotion(
                        conversation_id, 0, emotion,
                        resp.get("label", ""), emotion_kw,
                        resp.get("soothe", ""), resp.get("action_hint", ""),
                    )
                except Exception:
                    pass

        # ========== 3. 信息完整性追踪 ==========
        completeness = CompletenessTracker.track(history, user_input, self.agent_type)
        completeness["agent_type"] = self.agent_type
        if db and conversation_id:
            try:
                db.log_completeness(conversation_id, 0, completeness)
            except Exception:
                pass

        # ========== 4. 构建 Agent ==========
        now = datetime.now()
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        current_time = now.strftime("%Y年%m月%d日 %H:%M:%S")
        current_weekday = weekday_names[now.weekday()]
        enhancement_instruction = ""
        enhancement_instruction += (
            f"\n【当前时间】现在是 {current_time} {current_weekday}。"
            f"请根据当前时间判断用户提到的相对时间（如'昨天'、'上周'、'上个月'）的具体日期，"
            f"计算维权时效是否过期、剩余天数等。"
            f"如果用户提到购买日期或纠纷发生日期，请结合当前时间计算已过天数和剩余时效。\n"
        )
        enhancement_instruction += f"\n【用户画像】{profile['style_hint']}\n"

        if completeness["should_ask"]:
            enhancement_instruction += (
                f"\n【信息完整性】当前完整度 {completeness['completeness']}%，"
                f"还缺少: {', '.join(completeness['missing_fields'][:3])}。"
                f"请在回答末尾提醒用户补充以上信息。\n"
            )

        enhancement_instruction += (
            "\n【知识边界感知】"
            "如果用户的问题超出消费维权范围（如刑事案件、离婚、劳动纠纷等），"
            "请诚实告知这超出了你的专业范围，建议咨询专业律师或拨打12348。\n"
        )

        full_system_prompt = self.system_prompt + emotion_instruction + enhancement_instruction
        agent = create_agent(
            self.llm, self.tools, system_prompt=full_system_prompt
        )

        # 构建完整消息列表
        history_messages = _gradio_history_to_messages(history)
        current_message = HumanMessage(content=user_input)
        all_messages = history_messages + [current_message]
        messages_input = {"messages": all_messages}

        # ========== 显示初始状态 ==========
        if emotion_prefix:
            yield emotion_prefix + "\n\u23f3 正在为您分析情况..."
            await asyncio.sleep(0.3)
        else:
            yield "\u23f3 正在分析您的问题，请稍候..."

        # ========== 5. astream 流式输出 ==========
        final_answer = ""
        tool_calls_made = []
        tool_status_labels = []
        all_streamed_chunks = []

        print(f"[Agent] 流式调用 LLM API (模型: {self.llm.model})...")
        t0 = time.time()

        try:
            async for event in agent.astream(messages_input, stream_mode="messages"):
                # 兼容不同版本: event 可能是 tuple 或直接是 message
                if isinstance(event, tuple) and len(event) == 2:
                    chunk = event[0]
                else:
                    chunk = event

                all_streamed_chunks.append(chunk)

                # 收集工具调用
                if isinstance(chunk, AIMessageChunk):
                    if getattr(chunk, "tool_calls", None):
                        for tc in chunk.tool_calls:
                            name = tc.get("name", "")
                            if name and name not in tool_calls_made:
                                tool_calls_made.append(name)
                                label = _TOOL_LABELS.get(name, name)
                                if label not in tool_status_labels:
                                    tool_status_labels.append(label)
                                    # 更新显示: 展示已调用的工具
                                    tool_text = "\n".join(f"> {l}" for l in tool_status_labels)
                                    display = (emotion_prefix or "") + tool_text + "\n\n\u23f3 正在生成回答..."
                                    yield display

                    # 流式输出文本内容（非工具调用的 chunk）
                    if chunk.content and not getattr(chunk, "tool_calls", None):
                        final_answer += chunk.content
                        # 构建当前显示内容
                        tool_text = ""
                        if tool_status_labels:
                            tool_text = "\n".join(f"> {l}" for l in tool_status_labels) + "\n\n"
                        display = (emotion_prefix or "") + tool_text + "---\n\n" + final_answer
                        yield display

                # 也处理完整的 AIMessage（某些版本会 yield 完整消息）
                elif hasattr(chunk, "type") and chunk.type == "ai":
                    if getattr(chunk, "tool_calls", None):
                        for tc in chunk.tool_calls:
                            name = tc.get("name", "")
                            if name and name not in tool_calls_made:
                                tool_calls_made.append(name)
                                label = _TOOL_LABELS.get(name, name)
                                if label not in tool_status_labels:
                                    tool_status_labels.append(label)

                    if chunk.content and not getattr(chunk, "tool_calls", None):
                        if not final_answer:
                            final_answer = chunk.content
                            tool_text = ""
                            if tool_status_labels:
                                tool_text = "\n".join(f"> {l}" for l in tool_status_labels) + "\n\n"
                            display = (emotion_prefix or "") + tool_text + "---\n\n" + final_answer
                            yield display

        except Exception as e:
            print(f"[Agent] 流式调用失败，回退到 invoke: {e}")
            # 回退到 invoke
            try:
                result = await asyncio.to_thread(agent.invoke, messages_input)
                for msg in result["messages"]:
                    all_streamed_chunks.append(msg)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.get("name", "")
                            if name and name not in tool_calls_made:
                                tool_calls_made.append(name)
                                label = _TOOL_LABELS.get(name, name)
                                tool_status_labels.append(label)
                    if (hasattr(msg, "type") and msg.type == "ai"
                            and not getattr(msg, "tool_calls", None)
                            and hasattr(msg, "content") and msg.content):
                        final_answer = msg.content
            except Exception as e2:
                print(f"[Agent] invoke 回退也失败: {e2}")
                error_msg = f"处理时出错: {str(e2)}"
                if emotion_prefix:
                    yield emotion_prefix + error_msg
                else:
                    yield error_msg
                return

        elapsed = time.time() - t0
        print(f"[Agent] API 调用完成 ({elapsed:.1f}s)")

        if tool_calls_made:
            print(f"[Agent] 工具调用: {', '.join(tool_calls_made)}")
        else:
            print(f"[Agent] 无工具调用，直接生成回答")

        if not final_answer:
            final_answer = "未能生成有效回复，请尝试换个问法。"

        # 向前端发送"正在自检"状态，避免后处理期间长时间无响应
        _tool_text = ""
        if tool_status_labels:
            _tool_text = "\n".join(f"> {l}" for l in tool_status_labels) + "\n\n"
        _pre_display = (emotion_prefix or "") + _tool_text + "---\n\n" + final_answer + "\n\n⏳ 正在自检回答质量，评估置信度..."
        yield _pre_display

        # 数据库: 记录消息和工具调用 + 业务数据提取
        msg_id = 0
        if db and conversation_id:
            try:
                msg_id = db.log_message(
                    conversation_id, "user", user_input, self.agent_type,
                )
                db.log_message(
                    conversation_id, "assistant", final_answer, self.agent_type,
                    response_time_ms=int(elapsed * 1000),
                )
                for tc_name in tool_calls_made:
                    db.log_tool_call(
                        conversation_id, msg_id, tc_name,
                        _TOOL_LABELS.get(tc_name, tc_name),
                        duration_ms=int(elapsed * 1000 / max(len(tool_calls_made), 1)),
                    )

                # ===== 业务数据提取入库 =====
                self._log_business_data(db, conversation_id, msg_id, tool_calls_made, user_input, final_answer)
            except Exception:
                pass

        # ========== 6. 思维链提取 ==========
        # 从收集到的 chunks 构建思维链
        reasoning_chain = ReasoningChainExtractor.extract(
            all_streamed_chunks, user_input, emotion
        )
        if db and conversation_id:
            try:
                db.log_reasoning_chain(
                    conversation_id, 0, reasoning_chain, tool_calls_made,
                    emotion, len(tool_calls_made) + 2,
                )
            except Exception:
                pass

        # ========== 7. Agent自反思 ==========
        reflector = SelfReflector()
        print(f"[Agent] 自反思 API 调用中...")
        t1 = time.time()
        reflection = await asyncio.to_thread(
            reflector.reflect, user_input, final_answer
        )
        print(f"[Agent] 自反思完成 ({time.time()-t1:.1f}s) — 评分: {reflection['score']}/20 ({reflection['quality_label']})")

        if not reflection["pass"] and reflection["suggestion"]:
            final_answer += f"\n\n> **自反思提示**: {reflection['suggestion']}"

        if db and conversation_id:
            try:
                db.log_reflection(conversation_id, 0, reflection)
            except Exception:
                pass

        # ========== 8. 置信度标注 ==========
        has_law_ref = any(
            kw in final_answer
            for kw in ["第", "条", "法", "规定"]
        )
        confidence = ConfidenceEvaluator.evaluate(
            user_input, final_answer, has_law_ref
        )
        confidence["has_law_ref"] = has_law_ref
        confidence["law_signal_count"] = sum(
            1 for sig in ["第", "条", "消费者权益保护法", "食品安全法", "产品质量法", "电子商务法", "民法典"]
            if sig in final_answer
        )

        if db and conversation_id:
            try:
                db.log_confidence(conversation_id, 0, confidence)
            except Exception:
                pass

        # ========== 9. 维权进度追踪 ==========
        progress = CaseProgressTracker.track(history, user_input)
        progress["progress_pct"] = int(progress["current_stage"] / 6 * 100)
        progress_text = CaseProgressTracker.format_progress(progress)

        if db and conversation_id:
            try:
                db.log_progress(conversation_id, 0, progress)
            except Exception:
                pass

        # ========== 10. 组装最终输出 ==========
        # 构建完整前缀
        prefix = emotion_prefix or ""

        # 画像标签
        profile_tags = []
        level_labels = {"expert": "专业", "intermediate": "进阶", "novice": "新手"}
        profile_tags.append(f"画像: {level_labels.get(profile['legal_level'], profile['legal_level'])}")
        if profile["urgency"] == "urgent":
            profile_tags.append("紧迫")
        if profile_tags:
            prefix += f"> 用户画像: {' / '.join(profile_tags)}\n"

        # 工具调用状态
        if tool_status_labels:
            status_block = "\n".join(f"> {label}" for label in tool_status_labels)
            prefix += status_block + "\n"

        # 信息完整度
        if completeness["completeness"] < 100 and completeness["missing_fields"]:
            prefix += f"> 信息完整度: {completeness['completeness']}% (缺少: {', '.join(completeness['missing_fields'][:3])})\n"

        # 思维链
        prefix += reasoning_chain + "\n"

        prefix += "\n---\n\n"

        # 构建后缀
        suffix_parts = []

        suffix_parts.append(
            f"\n\n---\n"
            f"> {confidence['confidence_emoji']} 置信度: **{confidence['confidence_label']}** — {confidence['reason']}"
        )

        if reflection["quality_label"] == "优质":
            suffix_parts.append(f"> **自反思**: 质量评估通过 ({reflection['score']}/20)")
        elif reflection["quality_label"] == "需改进":
            suffix_parts.append(f"> **自反思**: {reflection['quality_label']} ({reflection['score']}/20)")

        if confidence["is_out_of_scope"]:
            suffix_parts.append(f"> {confidence['boundary_warning']}")

        suffix_parts.append(progress_text)

        suffix = "\n".join(suffix_parts)

        # ========== 流式输出最终完整结果 ==========
        # 逐字输出，模拟打字机效果，让用户感受到AI在"思考"
        full_output = prefix + final_answer + suffix
        total_len = len(full_output)
        # 根据内容长度动态调整打字速度，目标总时间约3秒
        target_time = 3.0
        base_delay = max(0.008, min(0.03, target_time / max(total_len, 1)))
        # 动态步长：保证总迭代次数约400次，总时间稳定在3-4秒
        step = max(1, int(total_len / 400))
        for i in range(0, total_len, step):
            yield full_output[:i + step]
            await asyncio.sleep(base_delay)

        # 确保最终完整输出
        yield full_output

    def _log_business_data(self, db, conv_id: int, msg_id: int,
                           tool_calls: list, user_input: str, answer: str):
        """
        从工具调用和回答内容中提取业务数据，写入数据库各业务表

        覆盖: 赔偿预估/证据清单/维权路径/商家话术/时效提醒/陷阱预警/信誉查询/
              文档生成(投诉信/审查报告)/投诉记录/条款审查/对话摘要
        """
        import re as _re

        # --- 赔偿预估 ---
        if "estimate_compensation" in tool_calls:
            try:
                amounts = _re.findall(r"[\d,]+\.?\d*", answer.replace(",", ""))
                est = float(amounts[0]) if amounts else 0
                dispute_type = "未知"
                for dt in ["食品安全", "欺诈", "人身损害", "预付款", "产品质量"]:
                    if dt in answer:
                        dispute_type = dt
                        break
                purchase = float(_re.search(r"购买金额.*?(\d+\.?\d*)", answer).group(1)) if _re.search(r"购买金额.*?(\d+\.?\d*)", answer) else 0
                db.log_compensation(conv_id, dispute_type, purchase, 0, est, "", "", answer[:500])
            except Exception:
                pass

        # --- 证据清单 ---
        if "generate_evidence_checklist" in tool_calls:
            try:
                dispute_type = "未知"
                for dt in ["食品安全", "网购欺诈", "格式条款", "预付卡", "服务质量"]:
                    if dt in answer:
                        dispute_type = dt
                        break
                items = [line.strip() for line in answer.split("\n") if line.strip() and line.strip()[0:1].isdigit()]
                if items:
                    db.log_evidence_checklist(conv_id, dispute_type, items)
            except Exception:
                pass

        # --- 维权路径 ---
        if "plan_rights_path" in tool_calls:
            try:
                path_type = "通用"
                if any(kw in answer for kw in ["食品安全", "过期", "变质"]):
                    path_type = "食品安全"
                elif any(kw in answer for kw in ["网购", "七天无理由"]):
                    path_type = "网购"
                steps = []
                for m in _re.finditer(r"第(\d+)步[:\s]*(.+?)(?:\n|$)", answer):
                    steps.append({"step": int(m.group(1)), "action": m.group(2).strip()})
                db.log_rights_path(conv_id, path_type, user_input[:200], steps)
            except Exception:
                pass

        # --- 商家话术应对 ---
        if "merchant_tactics_response" in tool_calls:
            try:
                # 从回答中提取商家话术和反驳
                statement = user_input[:500]
                db.log_merchant_tactics(conv_id, statement, answer[:2000])
            except Exception:
                pass

        # --- 时效提醒 ---
        if "rights_deadline_reminder" in tool_calls:
            try:
                from datetime import datetime, timedelta
                deadline_type = "未知"
                for dt in ["七天无理由退货", "质量问题退货", "质量保修", "民事诉讼时效", "12315投诉时效"]:
                    if dt in answer:
                        deadline_type = dt
                        break
                remaining = 0
                m = _re.search(r"剩余天数[:\s]*(\d+)", answer)
                if m:
                    remaining = int(m.group(1))
                urgency = "充裕"
                if "紧急" in answer:
                    urgency = "紧急"
                elif "过期" in answer:
                    urgency = "已过期"
                law = ""
                m = _re.search(r"《(.+?)》", answer)
                if m:
                    law = f"《{m.group(1)}》"
                db.log_deadline(conv_id, deadline_type, "", "", remaining, urgency, law)
            except Exception:
                pass

        # --- 陷阱预警 ---
        if "trap_warning" in tool_calls:
            try:
                industry = "未知"
                for ind in ["预付卡", "电商", "食品", "教育", "租房", "医美", "通信"]:
                    if ind in answer or ind in user_input:
                        industry = ind
                        break
                db.log_trap_warning(conv_id, industry, answer[:2000])
            except Exception:
                pass

        # --- 商家信誉 ---
        if "check_merchant_reputation" in tool_calls:
            try:
                merchant = ""
                for name in ["永辉", "美团", "淘宝", "京东", "拼多多"]:
                    if name in answer:
                        merchant = name
                        break
                if merchant:
                    db.log_merchant_reputation(merchant, {})
            except Exception:
                pass

        # --- 文档生成: 投诉信 ---
        if "create_complaint_report" in tool_calls:
            try:
                import os
                m = _re.search(r"投诉信已生成[:\s]*(\S+)", answer)
                filename = m.group(1) if m else ""
                from config import WORD_REPORTS_DIR
                filepath = os.path.join(WORD_REPORTS_DIR, filename) if filename else ""
                file_size = os.path.getsize(filepath) if filepath and os.path.exists(filepath) else 0
                if filename:
                    db.log_document(conv_id, "complaint", filename, filepath, file_size)
                # 记录投诉信业务数据
                db.log_complaint(conv_id, {
                    "complainant_name": "",
                    "merchant_name": "",
                    "dispute_detail": user_input[:500],
                    "demand": "",
                    "legal_basis": "",
                }, filepath, file_size)
            except Exception:
                pass

        # --- 文档生成: 审查报告 ---
        if "create_review_report" in tool_calls:
            try:
                import os
                m = _re.search(r"审查报告已生成[:\s]*(\S+)", answer)
                filename = m.group(1) if m else ""
                from config import WORD_REPORTS_DIR
                filepath = os.path.join(WORD_REPORTS_DIR, filename) if filename else ""
                file_size = os.path.getsize(filepath) if filepath and os.path.exists(filepath) else 0
                risk_level = "中"
                for rl in ["低", "中", "高"]:
                    if rl in answer:
                        risk_level = rl
                        break
                if filename:
                    db.log_document(conv_id, "review", filename, filepath, file_size)
                db.log_clause_review(conv_id, {
                    "contract_title": user_input[:200],
                    "clause_content": user_input[:1000],
                    "risk_level": risk_level,
                    "risk_score": 5,
                    "review_results": answer[:1000],
                }, filepath, file_size)
            except Exception:
                pass

        # --- 多平台投诉 ---
        if "multi_platform_complaint" in tool_calls:
            try:
                db.log_document(conv_id, "platform_complaint", "", "", 0)
            except Exception:
                pass

        # --- 证据打包 ---
        if "package_evidence" in tool_calls:
            try:
                import os
                m = _re.search(r"维权材料包已生成[:\s]*(\S+)", answer)
                filename = m.group(1) if m else ""
                from config import WORD_REPORTS_DIR
                filepath = os.path.join(WORD_REPORTS_DIR, filename) if filename else ""
                file_size = os.path.getsize(filepath) if filepath and os.path.exists(filepath) else 0
                if filename:
                    db.log_document(conv_id, "evidence_package", filename, filepath, file_size)
            except Exception:
                pass

        # --- 邮件发送 ---
        if "send_email" in tool_calls:
            try:
                from config import SMTP_USER
                # 尝试从回答中提取收件人邮箱
                to_email = ""
                m = _re.search(r"收件人[:\s]*([\w.+-]+@[\w-]+\.[\w.-]+)", answer)
                if m:
                    to_email = m.group(1)
                # 尝试提取邮件主题
                subject = ""
                m = _re.search(r"主题[:\s]*(.+?)(?:\n|$)", answer)
                if m:
                    subject = m.group(1).strip()
                # 判断发送状态
                if "✅" in answer and "发送成功" in answer:
                    status = "sent"
                elif "❌" in answer and "发送失败" in answer:
                    status = "failed"
                else:
                    status = "preview"  # 仅预览，未发送
                # 提取附件信息
                attachments = []
                m = _re.search(r"附件[:\s]*(.+?)(?:\n|$)", answer)
                if m and "无" not in m.group(1):
                    attachments = [a.strip() for a in m.group(1).split(",") if a.strip()]
                db.log_email(
                    conv_id, msg_id,
                    from_email=SMTP_USER or "",
                    to_email=to_email,
                    subject=subject,
                    body_preview=answer[:500],
                    attachments=attachments,
                    status=status,
                    error_message="" if status != "failed" else answer[:500],
                )
            except Exception:
                pass

        # --- Agent交接 ---
        if "handoff_to_agent" in tool_calls:
            try:
                m = _re.search(r"切换目标[:\s]*(.+?)(?:\n|$)", answer)
                target = m.group(1).strip() if m else ""
                agent_map = {"投诉信起草专家": "complaint", "条款审查专家": "review", "消费法律问答助手": "qa"}
                to_agent = agent_map.get(target, "")
                if to_agent:
                    db.log_handoff(conv_id, msg_id, self.agent_type, to_agent, answer[:300])
            except Exception:
                pass

