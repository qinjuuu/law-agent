"""
消费维权智能助手 - Gradio 主界面
亮色主题 + 橙色点缀，五 Tab 交互界面
Tab1 综合问答(自动路由+情绪感知) Tab2 投诉信起草 Tab3 条款审查
Tab4 维权工具箱(赔偿计算器/证据清单/陷阱查询/信誉查询/时效提醒/话术应对)
Tab5 系统信息
"""
import os
import sys

# ============================================================
# HuggingFace 镜像配置（必须在 import gradio 之前设置）
# gradio 内部依赖 huggingface_hub，import 时会缓存 HF_ENDPOINT
# 用国内镜像 + 关闭 SSL 校验，嵌入模型从镜像加载或本地缓存读取
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

import warnings

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gradio as gr
from config import MODEL_ID, EMBEDDING_MODEL, VECTORS_DIR, WORD_REPORTS_DIR, WORK_ROOT, MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, SMTP_ENABLED, SMTP_SERVER, SMTP_USER

# ============================================================
# 亮色主题 CSS（橙色点缀）
# ============================================================
ARK_CSS = """
/* ===== 全局背景与字体 ===== */
.gradio-container {
    background: #f5f5f7 !important;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif !important;
}

/* ===== 标题区域 ===== */
#ark-header {
    background: linear-gradient(135deg, #ffffff 0%, #fff5ee 50%, #ffffff 100%);
    border-bottom: 3px solid #ff8c42 !important;
    padding: 20px 24px !important;
    margin-bottom: 0 !important;
    border-radius: 6px 6px 0 0 !important;
}
#ark-header h1 {
    color: #1a1a2e !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    margin: 0 !important;
}
#ark-header .ark-subtitle {
    color: #888 !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    margin-top: 6px !important;
}

/* ===== Tab 区域 ===== */
.gradio-container .tab-nav {
    background: #ffffff !important;
    border-bottom: 2px solid #e0e0e0 !important;
}
.gradio-container .tab-nav button {
    color: #888 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.gradio-container .tab-nav button:hover {
    color: #ff8c42 !important;
}
.gradio-container .tab-nav button.selected {
    color: #ff8c42 !important;
    border-bottom: 3px solid #ff8c42 !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* ===== 面板/卡片 ===== */
.gr-panel {
    background: #ffffff !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 6px !important;
}

/* ===== 聊天界面 ===== */
.gradio-container .message-wrap,
.gradio-container .chat-window {
    background: #ffffff !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 6px !important;
}
.gradio-container .message {
    background: #f9f9fb !important;
    color: #333 !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
}
.gradio-container .message.user {
    background: #fff5ee !important;
    border-left: 4px solid #ff8c42 !important;
}
.gradio-container .message.bot {
    background: #f0f4ff !important;
    border-left: 4px solid #4a90d9 !important;
}

/* ===== 输入框 ===== */
.gradio-container textarea,
.gradio-container input[type="text"] {
    background: #ffffff !important;
    color: #333 !important;
    border: 1px solid #d0d0d8 !important;
    border-radius: 6px !important;
    font-size: 14px !important;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
    border-color: #ff8c42 !important;
    box-shadow: 0 0 0 2px rgba(255, 140, 66, 0.12) !important;
}

/* ===== 按钮 ===== */
.gradio-container button.primary,
.gradio-container button.btn-primary {
    background: #ff8c42 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}
.gradio-container button.primary:hover {
    background: #ff7022 !important;
}
.gradio-container button:not(.primary) {
    background: #ffffff !important;
    color: #555 !important;
    border: 1px solid #d0d0d8 !important;
    border-radius: 6px !important;
}
.gradio-container button:not(.primary):hover {
    border-color: #ff8c42 !important;
    color: #ff8c42 !important;
}

/* ===== Markdown 区域 ===== */
.gradio-container .prose {
    color: #333 !important;
}
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: #1a1a2e !important;
}
.gradio-container .prose strong {
    color: #ff7022 !important;
}
.gradio-container .prose code {
    background: #f0f0f4 !important;
    color: #c45500 !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 3px !important;
    padding: 2px 6px !important;
}
.gradio-container .prose blockquote {
    border-left: 4px solid #ff8c42 !important;
    background: #fff8f3 !important;
    color: #666 !important;
}

/* ===== 状态指示器 ===== */
#ark-status-bar {
    background: #ffffff !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
    color: #666 !important;
}
.ark-status-dot {
    display: inline-block !important;
    width: 8px !important;
    height: 8px !important;
    border-radius: 50% !important;
    margin-right: 6px !important;
    vertical-align: middle !important;
}
.ark-dot-green { background: #4ade80 !important; }
.ark-dot-orange { background: #ff8c42 !important; }
.ark-dot-gray { background: #ccc !important; }

/* ===== 装饰线 ===== */
.ark-divider {
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #e0e0e0, #ff8c4260, #e0e0e0, transparent) !important;
    margin: 12px 0 !important;
    border: none !important;
}

/* ===== 工具箱卡片 ===== */
.tool-card {
    background: #ffffff !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    transition: all 0.2s !important;
}
.tool-card:hover {
    border-color: #ff8c42 !important;
    box-shadow: 0 2px 8px rgba(255, 140, 66, 0.1) !important;
}
.tool-card h3 {
    color: #ff8c42 !important;
    font-size: 16px !important;
    margin-bottom: 8px !important;
}
"""

# ============================================================
# 懒加载智能体（避免启动时加载模型）
# ============================================================
_router = None
_qa_agent = None
_complaint_agent = None
_review_agent = None
_rag_status = None
_last_history = None  # 保存最近一次对话历史，供摘要导出使用


def _get_router():
    global _router
    if _router is None:
        from agents.router import IntentRouter
        _router = IntentRouter()
    return _router


def _get_qa_agent():
    global _qa_agent
    if _qa_agent is None:
        from agents.qa_agent import ConsumerQAAgent
        _qa_agent = ConsumerQAAgent()
    return _qa_agent


def _get_complaint_agent():
    global _complaint_agent
    if _complaint_agent is None:
        from agents.complaint_agent import ComplaintAgent
        _complaint_agent = ComplaintAgent()
    return _complaint_agent


def _get_review_agent():
    global _review_agent
    if _review_agent is None:
        from agents.review_agent import ReviewAgent
        _review_agent = ReviewAgent()
    return _review_agent


def _check_rag_status():
    global _rag_status
    if _rag_status is None:
        faiss_path = os.path.join(VECTORS_DIR, "faiss_index.faiss")
        bm25_path = os.path.join(VECTORS_DIR, "bm25_index.pkl")
        faiss_ok = os.path.exists(faiss_path)
        bm25_ok = os.path.exists(bm25_path)
        if faiss_ok and bm25_ok:
            _rag_status = "ready"
        elif faiss_ok or bm25_ok:
            _rag_status = "partial"
        else:
            _rag_status = "empty"
    return _rag_status


# ============================================================
# Tab 1: 综合智能问答（自动路由 + 情绪感知）
# ============================================================
async def chat_unified(message: str, history: list):
    """
    综合问答: 自动识别用户意图，路由到对应智能体
    内置情绪感知: 检测到负面情绪时先安抚再联动维权路径
    """
    global _last_history
    router = _get_router()

    intent = router.classify(message)
    intent_labels = {
        "complaint": "投诉信起草",
        "review": "格式条款审查",
        "qa": "消费法律问答",
    }
    intent_label = intent_labels.get(intent, "消费法律问答")
    print(f"[Router] 用户意图: {intent} ({intent_label})")
    print(f"[Router] 路由到: {intent_label}智能体")

    # 数据库: 记录意图路由 + Agent交接
    try:
        from database import db
        import uuid
        session_id = str(uuid.uuid4())[:8]
        conv_id = db.create_conversation(session_id, intent, message[:50])
        if conv_id:
            msg_id = db.log_message(conv_id, "user", message, intent)
            db.log_intent_route(conv_id, msg_id, message, intent, intent, 1.0)
            # 如果路由到的Agent和默认qa不同，记录交接
            if intent != "qa":
                db.log_handoff(conv_id, msg_id, "qa", intent, f"意图识别: {intent_label}")
    except Exception:
        conv_id = 0

    if intent == "complaint":
        agent = _get_complaint_agent()
    elif intent == "review":
        agent = _get_review_agent()
    else:
        agent = _get_qa_agent()

    prefix = f"> 意图识别完成: **{intent_label}**\n\n"
    yield prefix + "\n\u23f3 正在接入对应智能体..."

    full_response = prefix
    async for chunk in agent.chat(message, history):
        full_response = prefix + chunk
        yield full_response

    # 保存对话历史供摘要导出
    _last_history = (history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": full_response}]


# ============================================================
# Tab 2: 投诉信起草
# ============================================================
async def chat_complaint(message: str, history: list):
    global _last_history
    agent = _get_complaint_agent()
    print(f"[Complaint] 收到消息: {message[:50]}...")
    full_response = ""
    async for chunk in agent.chat(message, history):
        full_response = chunk
        yield chunk
    _last_history = (history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": full_response}]


# ============================================================
# Tab 3: 格式条款审查
# ============================================================
async def chat_review(message: str, history: list):
    global _last_history
    agent = _get_review_agent()
    print(f"[Review] 收到消息: {message[:50]}...")
    full_response = ""
    async for chunk in agent.chat(message, history):
        full_response = chunk
        yield chunk
    _last_history = (history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": full_response}]


# ============================================================
# Tab 4: 维权工具箱（快捷工具，非对话式）
# ============================================================

def _get_db_conv():
    """获取数据库连接和对话ID（工具箱场景用临时对话）"""
    try:
        from database import db
        import uuid
        session_id = str(uuid.uuid4())[:8]
        conv_id = db.create_conversation(session_id, "toolbox", "工具箱操作")
        return db, conv_id
    except Exception:
        return None, 0


def tool_compensation(dispute_type, purchase_amount, actual_loss):
    """赔偿计算器"""
    from tools.innovation_tools import estimate_compensation
    if not purchase_amount:
        return "请输入购买金额"
    try:
        amount = float(purchase_amount)
        loss = float(actual_loss) if actual_loss else 0
        result = estimate_compensation.invoke({
            "dispute_type": dispute_type,
            "purchase_amount": amount,
            "actual_loss": loss,
        })
        # 数据库: 记录赔偿预估
        try:
            db, conv_id = _get_db_conv()
            if db and conv_id:
                rule = {
                    "食品安全": ("《食品安全法》第一百四十八条", 10, 3, 1000),
                    "欺诈": ("《消费者权益保护法》第五十五条", 3, 0, 500),
                    "人身损害": ("《消费者权益保护法》第四十九条", 0, 1, 0),
                    "预付款": ("《消费者权益保护法》第五十三条", 0, 0, 0),
                    "产品质量": ("《产品质量法》第四十四条", 0, 1, 0),
                }
                law, mult, loss_mult, minimum = rule.get(dispute_type, ("", 0, 0, 0))
                est_amount = max(amount * mult, loss * loss_mult, minimum) if mult > 0 else (loss if loss > 0 else minimum)
                db.log_compensation(conv_id, dispute_type, amount, loss, est_amount, law, f"max({amount}x{mult}, {loss}x{loss_mult}, {minimum})", result[:500])
        except Exception:
            pass
        return result
    except ValueError:
        return "金额格式错误，请输入数字"
    except Exception as e:
        return f"计算失败: {str(e)}"


def tool_evidence(dispute_type):
    """证据清单生成器"""
    from tools.innovation_tools import generate_evidence_checklist
    try:
        result = generate_evidence_checklist.invoke({"dispute_type": dispute_type})
        # 数据库: 记录证据清单
        try:
            db, conv_id = _get_db_conv()
            if db and conv_id:
                items = [line.strip() for line in result.split("\n") if line.strip() and line.strip()[0].isdigit()]
                db.log_evidence_checklist(conv_id, dispute_type, items)
        except Exception:
            pass
        return result
    except Exception as e:
        return f"生成失败: {str(e)}"


def tool_trap(industry):
    """消费陷阱查询"""
    from tools.innovation_tools import trap_warning
    try:
        result = trap_warning.invoke({"industry": industry})
        # 数据库: 记录陷阱查询
        try:
            db, conv_id = _get_db_conv()
            if db and conv_id:
                db.log_trap_warning(conv_id, industry, result[:2000])
        except Exception:
            pass
        return result
    except Exception as e:
        return f"查询失败: {str(e)}"


def tool_reputation(merchant_name):
    """商家信誉查询"""
    from tools.innovation_tools import check_merchant_reputation
    if not merchant_name:
        return "请输入商家名称"
    try:
        result = check_merchant_reputation.invoke({"merchant_name": merchant_name})
        # 数据库: 记录/更新商家信誉
        try:
            db, conv_id = _get_db_conv()
            if db and conv_id:
                from tools.innovation_tools import _MERCHANT_REPUTATION
                matched = None
                for name, data in _MERCHANT_REPUTATION.items():
                    if name in merchant_name or merchant_name in name:
                        matched = data
                        break
                if matched:
                    db.log_merchant_reputation(merchant_name, matched)
        except Exception:
            pass
        return result
    except Exception as e:
        return f"查询失败: {str(e)}"


def tool_deadline(deadline_type, purchase_date):
    """维权时效提醒"""
    from tools.innovation_tools import rights_deadline_reminder
    try:
        result = rights_deadline_reminder.invoke({
            "dispute_type": deadline_type,
            "purchase_date": purchase_date or "",
        })
        # 数据库: 记录时效提醒
        try:
            db, conv_id = _get_db_conv()
            if db and conv_id:
                from tools.innovation_tools import _DEADLINES
                dl = _DEADLINES.get(deadline_type, {})
                from datetime import datetime, timedelta
                pd = datetime.strptime(purchase_date, "%Y-%m-%d") if purchase_date else datetime.now()
                dd = pd + timedelta(days=dl.get("days", 0))
                remaining = (dd - datetime.now()).days
                urgency = "已过期" if remaining < 0 else ("紧急" if remaining <= 3 else ("较紧急" if remaining <= 7 else "充裕"))
                db.log_deadline(conv_id, deadline_type, purchase_date or pd.strftime("%Y-%m-%d"), dd.strftime("%Y-%m-%d"), remaining, urgency, dl.get("law", ""))
        except Exception:
            pass
        return result
    except Exception as e:
        return f"查询失败: {str(e)}"


def tool_tactics(merchant_statement):
    """商家话术应对"""
    from tools.innovation_tools import merchant_tactics_response
    if not merchant_statement:
        return "请输入商家说的话术"
    try:
        result = merchant_tactics_response.invoke({"merchant_statement": merchant_statement})
        # 数据库: 记录商家话术应对
        try:
            db, conv_id = _get_db_conv()
            if db and conv_id:
                db.log_merchant_tactics(conv_id, merchant_statement, result[:2000])
        except Exception:
            pass
        return result
    except Exception as e:
        return f"分析失败: {str(e)}"


def tool_rights_path(dispute_description):
    """维权路径规划"""
    from tools.innovation_tools import plan_rights_path
    if not dispute_description:
        return "请描述您的纠纷情况"
    try:
        result = plan_rights_path.invoke({"dispute_description": dispute_description})
        # 数据库: 记录维权路径
        try:
            db, conv_id = _get_db_conv()
            if db and conv_id:
                from tools.innovation_tools import _RIGHTS_PATHS
                desc = dispute_description.lower()
                if any(kw in desc for kw in ["食品", "过期", "变质"]):
                    path_type = "食品安全"
                elif any(kw in desc for kw in ["网购", "快递", "退货"]):
                    path_type = "网购"
                else:
                    path_type = "通用"
                steps = _RIGHTS_PATHS.get(path_type, [])
                db.log_rights_path(conv_id, path_type, dispute_description, steps)
        except Exception:
            pass
        return result
    except Exception as e:
        return f"规划失败: {str(e)}"


# ============================================================
# 邮件发送（工具箱独立功能）
# ============================================================
def tool_send_email(to_email, subject, body, attachment_file, confirm):
    """发送邮件（工具箱版，支持附件和用户确认）

    确认流程:
    1. 点击"预览邮件"按钮 → confirm=False，返回邮件预览，不实际发送
    2. 用户查看预览后，勾选"我已确认邮件内容，同意发送"复选框
    3. 点击"发送邮件"按钮 → confirm=True（复选框已勾选），实际发送邮件
    4. 如果未勾选复选框直接点击"发送邮件"，会提示用户先确认
    """
    from tools.email_tools import send_email
    from config import SMTP_ENABLED

    # SMTP 配置检查
    if not SMTP_ENABLED:
        return (
            "邮件功能未启用：SMTP 配置不完整。\n"
            "请在 .env 文件中配置 SMTP_SERVER、SMTP_PORT、SMTP_USER、SMTP_PASSWORD。"
        )

    if not to_email or "@" not in to_email:
        return "请输入有效的收件人邮箱地址"
    if not subject:
        return "请输入邮件主题"
    if not body:
        return "请输入邮件正文"

    attachments = [attachment_file] if attachment_file else None

    # 未勾选确认复选框时，不发送，返回预览 + 提示
    if not confirm:
        try:
            preview = send_email.invoke({
                "to_email": to_email,
                "subject": subject,
                "body": body,
                "attachments": attachments,
                "confirm": False,
            })
            return (
                f"{preview}\n\n"
                "⚠️ 邮件尚未发送！\n"
                "请确认以上邮件内容无误后，勾选上方的「我已确认邮件内容，同意发送」复选框，"
                "再点击「发送邮件」按钮完成发送。"
            )
        except Exception as e:
            return f"邮件预览失败: {str(e)}"

    # confirm=True，实际发送邮件
    try:
        result = send_email.invoke({
            "to_email": to_email,
            "subject": subject,
            "body": body,
            "attachments": attachments,
            "confirm": True,
        })
        return result
    except Exception as e:
        return f"邮件发送失败: {str(e)}"


def get_report_files():
    """获取 workspace/reports/ 下的 .docx 文件列表，供邮件附件下拉选择"""
    files = []
    if os.path.exists(WORD_REPORTS_DIR):
        for f in sorted(os.listdir(WORD_REPORTS_DIR)):
            if f.endswith(".docx") or f.endswith(".zip"):
                files.append(f)
    return files if files else []


# ============================================================
# 对话摘要导出
# ============================================================
def export_conversation_summary(history: list):
    """导出对话纪要"""
    from agents.enhancements import ConversationSummarizer
    try:
        summarizer = ConversationSummarizer()
        summary = summarizer.summarize(history)
        filepath = summarizer.save_summary(summary)
        # 数据库: 记录对话摘要
        try:
            from database import db
            import os
            db, conv_id = _get_db_conv()
            if db and conv_id:
                file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                db.log_conversation_summary(conv_id, summary[:2000], filepath, len(history))
                # 同时记录文档生成
                db.log_document(conv_id, "summary", os.path.basename(filepath), filepath, file_size)
        except Exception:
            pass
        return f"对话纪要已导出!\n\n保存路径: {filepath}\n\n纪要内容:\n\n{summary}"
    except Exception as e:
        return f"导出失败: {str(e)}"


# ============================================================
# Tab 5: 系统信息
# ============================================================
def get_system_info():
    rag_status = _check_rag_status()
    rag_label = {"ready": "已就绪", "partial": "部分加载", "empty": "未构建"}.get(
        rag_status, "未知"
    )

    report_files = []
    if os.path.exists(WORD_REPORTS_DIR):
        for f in sorted(os.listdir(WORD_REPORTS_DIR)):
            if f.endswith(".docx"):
                size = os.path.getsize(os.path.join(WORD_REPORTS_DIR, f))
                report_files.append(f"| {f} | {size:,} 字节 |")

    workspace_files = []
    if os.path.exists(WORK_ROOT):
        for f in sorted(os.listdir(WORK_ROOT)):
            workspace_files.append(f"| {f} |")

    # 数据库统计
    db_stats_html = ""
    try:
        from database import db
        stats = db.get_stats()
        db_size_kb = db.get_db_size() / 1024

        # 工具调用统计
        tool_stats = db.get_tool_call_stats()
        tool_stats_rows = ""
        if tool_stats:
            for ts in tool_stats[:10]:
                tool_stats_rows += f"| {ts.get('tool_label', ts.get('tool_name', ''))} | {ts['call_count']} | {ts.get('success_count', 0)} | {ts.get('avg_ms', 0)} |\n"
        else:
            tool_stats_rows = "| 暂无调用记录 | - | - | - |\n"

        # 情绪分布
        emotion_stats = db.get_emotion_stats()
        emotion_rows = ""
        if emotion_stats:
            for es in emotion_stats:
                emotion_rows += f"| {es.get('emotion_label', es.get('emotion_type', ''))} | {es['count']} |\n"
        else:
            emotion_rows = "| 暂无情绪记录 | - |\n"

        # 自反思质量
        reflection_stats = db.get_reflection_stats()
        reflection_rows = ""
        if reflection_stats:
            for rs in reflection_stats:
                reflection_rows += f"| {rs.get('quality_label', '')} | {rs['count']} | {rs.get('avg_score', 0)} |\n"
        else:
            reflection_rows = "| 暂无自反思记录 | - | - |\n"

        # 对话统计
        conv_stats = db.get_conversation_stats()
        conv_rows = ""
        if conv_stats:
            for cs in conv_stats:
                agent_labels = {"qa": "法律问答", "complaint": "投诉信起草", "review": "条款审查", "toolbox": "工具箱"}
                conv_rows += f"| {agent_labels.get(cs.get('agent_type', ''), cs.get('agent_type', ''))} | {cs.get('count', 0)} | {cs.get('messages', 0)} |\n"
        else:
            conv_rows = "| 暂无对话记录 | - | - |\n"

        # 意图路由统计
        intent_stats = db.get_intent_route_stats()
        intent_rows = ""
        if intent_stats:
            for ir in intent_stats:
                intent_rows += f"| {ir.get('detected_intent', '')} → {ir.get('routed_agent', '')} | {ir.get('count', 0)} |\n"
        else:
            intent_rows = "| 暂无路由记录 | - |\n"

        # Agent交接统计
        handoff_stats = db.get_handoff_stats()
        handoff_rows = ""
        if handoff_stats:
            for hs in handoff_stats:
                handoff_rows += f"| {hs.get('from_agent', '')} → {hs.get('to_agent', '')} | {hs.get('count', 0)} |\n"
        else:
            handoff_rows = "| 暂无交接记录 | - |\n"

        # 文档生成统计
        doc_stats = db.get_document_stats()
        doc_rows = ""
        if doc_stats:
            for ds in doc_stats:
                doc_type_labels = {"complaint": "投诉信", "review": "审查报告", "summary": "对话纪要", "evidence_package": "证据包", "platform_complaint": "平台投诉书"}
                doc_rows += f"| {doc_type_labels.get(ds.get('doc_type', ''), ds.get('doc_type', ''))} | {ds.get('count', 0)} | {ds.get('total_size', 0):,} |\n"
        else:
            doc_rows = "| 暂无文档记录 | - | - |\n"

        # 赔偿预估统计
        comp_stats = db.get_compensation_stats()
        comp_rows = ""
        if comp_stats:
            for cs2 in comp_stats:
                comp_rows += f"| {cs2.get('dispute_type', '')} | {cs2.get('count', 0)} | {cs2.get('avg_amount', 0)} | {cs2.get('max_amount', 0)} |\n"
        else:
            comp_rows = "| 暂无赔偿记录 | - | - | - |\n"

        # 维权进度统计
        progress_stats = db.get_progress_stats()
        progress_rows = ""
        if progress_stats:
            for ps in progress_stats:
                progress_rows += f"| {ps.get('current_label', '')} | {ps.get('count', 0)} | {ps.get('avg_pct', 0)} |\n"
        else:
            progress_rows = "| 暂无进度记录 | - | - |\n"

        # 时效提醒统计
        deadline_stats = db.get_deadline_stats()
        deadline_rows = ""
        if deadline_stats:
            for ds2 in deadline_stats:
                deadline_rows += f"| {ds2.get('deadline_type', '')} | {ds2.get('count', 0)} | {ds2.get('urgent_count', 0)} | {ds2.get('expired_count', 0)} |\n"
        else:
            deadline_rows = "| 暂无时效记录 | - | - | - |\n"

        # 用户画像统计
        profile_stats = db.get_user_profile_stats()
        profile_rows = ""
        if profile_stats:
            for ps2 in profile_stats:
                level_labels = {"expert": "专业", "intermediate": "进阶", "novice": "新手"}
                profile_rows += f"| {level_labels.get(ps2.get('legal_level', ''), ps2.get('legal_level', ''))} | {ps2.get('urgency', '')} | {ps2.get('user_type', '')} | {ps2.get('count', 0)} |\n"
        else:
            profile_rows = "| 暂无画像记录 | - | - | - |\n"

        # 置信度分布
        confidence_stats = db.get_confidence_stats()
        confidence_rows = ""
        if confidence_stats:
            for cs3 in confidence_stats:
                confidence_rows += f"| {cs3.get('confidence_label', '')} | {cs3.get('count', 0)} |\n"
        else:
            confidence_rows = "| 暂无置信度记录 | - |\n"

        # 信息完整性统计
        completeness_stats = db.get_completeness_stats()
        completeness_rows = ""
        if completeness_stats:
            for cs4 in completeness_stats:
                completeness_rows += f"| {cs4.get('agent_type', '')} | {cs4.get('avg_completeness', 0)} | {cs4.get('ask_count', 0)} | {cs4.get('total', 0)} |\n"
        else:
            completeness_rows = "| 暂无完整性记录 | - | - | - |\n"

        # 邮件发送统计
        email_stats = db.get_email_stats()
        email_rows = ""
        if email_stats:
            for es in email_stats:
                status_label = {"sent": "发送成功", "failed": "发送失败", "preview": "仅预览"}.get(es.get("status", ""), es.get("status", ""))
                email_rows += f"| {status_label} | {es.get('count', 0)} |\n"
        else:
            email_rows = "| 暂无邮件记录 | - |\n"

        # 扩展数据表记录数
        ext_table_rows = ""
        ext_tables = {
            "evidence_checklists": "证据清单",
            "rights_paths": "维权路径",
            "merchant_tactics": "商家话术",
            "deadline_reminders": "时效提醒",
            "merchant_reputations": "商家信誉",
            "trap_warnings": "陷阱预警",
            "conversation_summaries": "对话摘要",
            "email_logs": "邮件发送日志",
            "merchant_tactics_kb": "话术知识库",
        }
        for t, label in ext_tables.items():
            ext_table_rows += f"| {t} | {stats['tables'].get(t, 0)} | {label} |\n"

        db_stats_html = f"""
---

## 数据库统计 (36张表 / {stats['total_records']}条记录 / {db_size_kb:.0f}KB)

### 核心数据表记录数

| 数据表 | 记录数 | 说明 |
|--------|--------|------|
| users | {stats['tables'].get('users', 0)} | 用户会话 |
| conversations | {stats['tables'].get('conversations', 0)} | 对话会话 |
| messages | {stats['tables'].get('messages', 0)} | 消息记录 |
| tool_calls | {stats['tables'].get('tool_calls', 0)} | 工具调用 |
| emotion_records | {stats['tables'].get('emotion_records', 0)} | 情绪检测 |
| self_reflections | {stats['tables'].get('self_reflections', 0)} | 自反思记录 |
| confidence_assessments | {stats['tables'].get('confidence_assessments', 0)} | 置信度评估 |
| case_progress | {stats['tables'].get('case_progress', 0)} | 维权进度 |
| reasoning_chains | {stats['tables'].get('reasoning_chains', 0)} | 思维链 |
| completeness_records | {stats['tables'].get('completeness_records', 0)} | 信息完整性 |
| intent_routes | {stats['tables'].get('intent_routes', 0)} | 意图路由 |
| agent_handoffs | {stats['tables'].get('agent_handoffs', 0)} | Agent交接 |
| complaints | {stats['tables'].get('complaints', 0)} | 投诉信 |
| clause_reviews | {stats['tables'].get('clause_reviews', 0)} | 条款审查 |
| compensation_estimates | {stats['tables'].get('compensation_estimates', 0)} | 赔偿预估 |
| documents | {stats['tables'].get('documents', 0)} | 文档记录 |

### 扩展业务表记录数

| 数据表 | 记录数 | 说明 |
|--------|--------|------|
{ext_table_rows}

### 知识库表记录数

| 数据表 | 记录数 | 说明 |
|--------|--------|------|
| laws | {stats['tables'].get('laws', 0)} | 法律条文 |
| case_precedents | {stats['tables'].get('case_precedents', 0)} | 案例 |
| compensation_rules | {stats['tables'].get('compensation_rules', 0)} | 赔偿规则 |
| evidence_templates | {stats['tables'].get('evidence_templates', 0)} | 证据模板 |
| rights_path_templates | {stats['tables'].get('rights_path_templates', 0)} | 维权路径模板 |
| platform_templates | {stats['tables'].get('platform_templates', 0)} | 平台模板 |
| rights_stages | {stats['tables'].get('rights_stages', 0)} | 维权阶段 |
| deadline_rules | {stats['tables'].get('deadline_rules', 0)} | 时效规则 |
| trap_kb | {stats['tables'].get('trap_kb', 0)} | 陷阱知识 |

### 工具调用排行

| 工具 | 调用次数 | 成功次数 | 平均耗时(ms) |
|------|----------|----------|-------------|
{tool_stats_rows}

### 情绪分布

| 情绪类型 | 次数 |
|----------|------|
{emotion_rows}

### 自反思质量分布

| 质量等级 | 次数 | 平均分 |
|----------|------|--------|
{reflection_rows}

### 置信度分布

| 置信度 | 次数 |
|--------|------|
{confidence_rows}

### 对话统计

| Agent类型 | 对话数 | 消息数 |
|-----------|--------|--------|
{conv_rows}

### 意图路由统计

| 路由路径 | 次数 |
|----------|------|
{intent_rows}

### Agent交接统计

| 交接路径 | 次数 |
|----------|------|
{handoff_rows}

### 文档生成统计

| 文档类型 | 数量 | 总大小(字节) |
|----------|------|-------------|
{doc_rows}

### 赔偿预估统计

| 纠纷类型 | 次数 | 平均金额 | 最高金额 |
|----------|------|----------|----------|
{comp_rows}

### 维权进度统计

| 当前阶段 | 人数 | 平均进度 |
|----------|------|----------|
{progress_rows}

### 时效提醒统计

| 时效类型 | 查询次数 | 紧急数 | 已过期数 |
|----------|----------|--------|----------|
{deadline_rows}

### 用户画像统计

| 法律水平 | 紧迫度 | 用户类型 | 人数 |
|----------|--------|----------|------|
{profile_rows}

### 信息完整性统计

| Agent类型 | 平均完整度 | 追问次数 | 总记录数 |
|-----------|-----------|----------|----------|
{completeness_rows}

### 邮件发送统计

| 状态 | 次数 |
|------|------|
{email_rows}"""
    except Exception as e:
        db_stats_html = f"\n---\n\n## 数据库统计\n\n数据库初始化中... ({e})\n"

    email_status = f"已启用 ({SMTP_SERVER}: {SMTP_USER})" if SMTP_ENABLED else "未配置"

    info = f"""## 系统状态

| 项目 | 状态 |
|------|------|
| 大语言模型 | `{MODEL_ID}` |
| 嵌入模型 | `{EMBEDDING_MODEL}` |
| RAG 知识库 | {rag_label} (7部法律法规, 155个文本块) |
| 联网搜索 | 已启用 (DuckDuckGo API) |
| 邮件发送 | {email_status} |
| 工作目录 | `{WORK_ROOT}` |
| 报告目录 | `{WORD_REPORTS_DIR}` |
| 数据库 | MySQL `{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}` |
| 流式输出 | 已启用 (astream) |

---

## 创新功能一览（25项）

### Agent 智能维度（9项）

| 创新点 | 说明 |
|--------|------|
| 多轮对话记忆 | Agent 记住上下文，支持分条补充信息 |
| 情绪感知与安抚 | 检测愤怒/焦虑/委屈情绪，先安抚再联动维权路径 |
| 用户画像自适应 | 检测法律知识水平/紧迫度，动态调整回复风格 |
| 信息完整性追踪 | 追踪缺失信息字段，计算完整度，驱动主动追问 |
| 思维链可视化 | 展示Agent推理过程和工具选择理由 |
| Agent自反思 | 生成后自检法律准确性/完整性/语气，不合格时修正 |
| 置信度标注 | 高/中/低置信度评估，低置信度建议咨询律师 |
| 知识边界感知 | 检测问题超出能力范围时诚实声明 |
| 维权进度追踪 | 跨对话状态机：协商→投诉→调解→诉讼 |

### 多Agent协作（1项）

| 创新点 | 说明 |
|--------|------|
| 多Agent协作交接 | QA/投诉/审查Agent间带上下文智能切换 |

### 工具与功能（15项）

| 创新点 | 说明 |
|--------|------|
| 赔偿金额智能预估 | 按法条自动计算退一赔三/退一赔十/最低赔偿 |
| 证据清单生成器 | 按纠纷类型生成需收集的证据清单 |
| 维权路径规划 | 分步骤给出行动建议、预计时间和成功率 |
| 商家话术应对 | 识别商家推诿话术并给出法律反驳 |
| 维权时效提醒 | 计算法定期限剩余天数，标注紧迫程度 |
| 多平台投诉适配 | 同一纠纷生成12315/消协/法院/电商四种格式 |
| 一键证据打包 | 投诉信+证据+清单打成zip |
| 消费陷阱预警 | 按行业查询常见消费陷阱 |
| 条款风险评分 | 1-10分风险评分+红黄绿颜色标注 |
| 商家信誉查询 | 本地数据+联网搜索双模式商家信誉 |
| 对话摘要导出 | 多轮对话压缩为结构化纪要文档 |
| **邮件发送** | 通过SMTP将投诉信/审查报告等维权文档邮件发送，发送前需用户确认 |
| **联网搜索增强** | DuckDuckGo API 实时搜索最新法律法规和案例 |
| **最新法规搜索** | 专门搜索2024年最新消费法规政策动态 |
| **商家信息联网** | 本地无数据时自动联网搜索商家投诉和处罚信息 |

{db_stats_html}

---

## 已生成报告 ({len(report_files)} 个)

| 文件名 | 大小 |
|--------|------|
{chr(10).join(report_files) if report_files else "| 暂无 | - |"}

---

## 工作目录文件 ({len(workspace_files)} 个)

| 文件名 |
|--------|
{chr(10).join(workspace_files) if workspace_files else "| 暂无 |"}

---

## 使用说明

1. **综合问答** — 直接输入问题，系统自动识别意图并路由，支持情绪感知、用户画像自适应、思维链可视化、自反思和置信度标注
2. **投诉信起草** — 描述消费纠纷，系统检索法条并生成投诉信，支持多轮补充信息和信息完整性追踪
3. **格式条款审查** — 输入合同条款，系统分析风险并生成审查报告，含1-10分评分
4. **维权工具箱** — 快捷使用赔偿计算器、证据清单、陷阱查询等独立工具，支持对话纪要导出
5. 生成的文档保存在 `workspace/reports/` 目录下

---

## Agent 智能增强说明

每次对话中，Agent 会自动执行以下增强流程:
- **用户画像检测**: 推断法律知识水平，动态调整回复风格
- **信息完整性追踪**: 追踪缺失信息，完整度不足时主动追问
- **思维链可视化**: 展示Agent选择了哪些工具及选择理由
- **自反思校验**: 生成回答后自检法律准确性/完整性/语气
- **置信度标注**: 评估回答置信度，低置信度时建议咨询律师
- **知识边界感知**: 问题超出范围时诚实声明
- **维权进度追踪**: 记录维权走到哪一步，推荐下一步
- **RAG混合检索**: FAISS向量+BM25关键词双路检索，加权融合
- **联网搜索增强**: 本地知识库覆盖不了时自动联网搜索最新法规和案例
- **全流程数据库记录**: 36张表记录对话/工具/情绪/自反思/置信度/进度等全维度数据
- **真正流式输出**: LLM 生成内容逐字实时显示，无需等待完整结果
"""
    return info


def refresh_files():
    global _rag_status
    _rag_status = None
    return get_system_info()


# ============================================================
# 状态栏
# ============================================================
def get_status_html():
    rag_status = _check_rag_status()
    if rag_status == "ready":
        dot_class = "ark-dot-green"
        rag_text = "知识库就绪"
    else:
        dot_class = "ark-dot-orange"
        rag_text = "知识库未就绪"
    return f"""
    <div id="ark-status-bar">
        <span class="ark-status-dot {dot_class}"></span>
        <span>{rag_text}</span>
        <span style="margin: 0 16px; color: #ddd;">|</span>
        <span class="ark-status-dot ark-dot-green"></span>
        <span>模型: {MODEL_ID}</span>
        <span style="margin: 0 16px; color: #ddd;">|</span>
        <span class="ark-status-dot ark-dot-green"></span>
        <span>嵌入: {EMBEDDING_MODEL}</span>
        <span style="margin: 0 16px; color: #ddd;">|</span>
        <span class="ark-status-dot ark-dot-green"></span>
        <span>情绪感知: 已启用</span>
    </div>
    """


# ============================================================
# 构建界面
# ============================================================
def create_app():
    with gr.Blocks(
        title="消费维权智能助手",
    ) as app:

        # ===== 顶部标题 =====
        with gr.Row(elem_id="ark-header"):
            with gr.Column():
                gr.HTML(
                    '<h1>消费维权智能助手</h1>'
                    '<div class="ark-subtitle">'
                    'CONSUMER RIGHTS PROTECTION ASSISTANT &nbsp;|&nbsp; '
                    '基于 LangChain + RAG + 多智能体架构 &nbsp;|&nbsp; '
                    '25项创新功能 &nbsp;|&nbsp; '
                    '36表全维度数据库'
                    '</div>'
                )

        # 状态栏
        gr.HTML(get_status_html())

        gr.HTML('<div class="ark-divider"></div>')

        # ===== Tab 区域 =====
        with gr.Tabs():

            # --- Tab 1: 综合问答 ---
            with gr.Tab("综合问答"):
                gr.Markdown(
                    "### 综合智能问答\n"
                    "> 输入您的消费维权问题，系统自动识别意图，"
                    "智能路由到法律问答、投诉信起草或条款审查。\n\n"
                    "**Agent 智能增强**\n"
                    "- 用户画像自适应: 检测法律知识水平，动态调整回复风格\n"
                    "- 信息完整性追踪: 追踪缺失信息，主动追问\n"
                    "- 思维链可视化: 展示Agent推理过程和工具选择理由\n"
                    "- Agent自反思: 生成后自检质量\n"
                    "- 置信度标注: 高/中/低置信度评估\n"
                    "- 维权进度追踪: 记录维权走到哪一步\n\n"
                    "**支持的能力**\n"
                    "- 法律问答: 检索法条和案例\n"
                    "- 投诉信起草: 生成Word投诉信\n"
                    "- 条款审查: 风险评分和审查报告\n"
                    "- 赔偿预估: 智能计算赔偿金额\n"
                    "- 维权路径: 分步骤行动建议\n"
                    "- 情绪感知: 检测负面情绪并安抚\n"
                    "- 多Agent协作: 自动切换到最合适的智能体\n\n"
                    "**试试这些例子**\n"
                    "- 咨询: *网上买的手机七天内有质量问题能退货吗*\n"
                    "- 投诉: *帮我写投诉信，超市买的面包过期了*\n"
                    "- 审查: *健身卡合同写着一经售出概不退换合法吗*\n"
                    "- 情绪: *气死我了，无良商家卖过期食品给我*\n"
                    "- 最新法规: *2024年消费者权益保护法实施条例有什么新规定*\n"
                    "- 商家查询: *某某直播间的商家靠谱吗，有没有被处罚过*"
                )
                gr.ChatInterface(
                    fn=chat_unified,
                    examples=[
                        "我在网上买的手机七天内有质量问题，能退货吗？",
                        "帮我写一封投诉信，我在超市买到了过期食品",
                        "健身卡合同里写着'一经售出概不退换'，这合法吗？",
                        "气死我了，网购的手机是假货，商家还不给退，该怎么办",
                        "我买的面包过期了，能赔多少钱",
                    ],
                )

            # --- Tab 2: 投诉信起草 ---
            with gr.Tab("投诉信起草"):
                gr.Markdown(
                    "### 投诉信起草\n"
                    "> 描述您的消费纠纷情况，系统将检索相关法律条文，"
                    "生成一份正式的投诉信 Word 文档。\n"
                    "> 支持多轮对话补充信息，系统会记住您之前说的内容。\n\n"
                    "**系统还会帮您**\n"
                    "- 预估赔偿金额\n"
                    "- 生成证据收集清单\n"
                    "- 生成不同平台的投诉版本\n"
                    "- 一键打包所有维权材料\n\n"
                    "**请尽量包含以下信息**\n"
                    "- 商家名称和购买时间\n"
                    "- 商品或服务名称及金额\n"
                    "- 纠纷经过和您的诉求"
                )
                gr.ChatInterface(
                    fn=chat_complaint,
                    examples=[
                        "我在电商平台买的鞋子是假货，商家拒绝退款",
                        "超市卖的面包过期了，我吃坏了肚子",
                        "帮我写投诉信，永辉超市买的过期面包，购买价12.8元",
                    ],
                )

            # --- Tab 3: 格式条款审查 ---
            with gr.Tab("条款审查"):
                gr.Markdown(
                    "### 格式条款审查\n"
                    "> 输入消费合同或格式条款内容，系统将分析其合法性，"
                    "对每条条款进行1-10分风险评分，并生成审查报告。\n\n"
                    "**风险评分标准**\n"
                    "- 1-3分 绿色: 条款合规，无需修改\n"
                    "- 4-6分 黄色: 有风险，建议修改\n"
                    "- 7-10分 红色: 违法条款，建议删除或举报\n\n"
                    "**可审查的条款类型**\n"
                    "- 会员卡/预付卡条款\n"
                    "- 电商平台用户协议\n"
                    "- 商家店内告示/声明"
                )
                gr.ChatInterface(
                    fn=chat_review,
                    examples=[
                        "商家合同里写着'最终解释权归本店所有'，这个条款有问题吗？",
                        "我办的会员卡上写着'余额不退不换'，帮我审查一下",
                        "健身房合同写着'一经售出概不退换，最终解释权归本店所有，会员卡余额过期作废'，帮我审查这些条款",
                    ],
                )

            # --- Tab 4: 维权工具箱 ---
            with gr.Tab("维权工具箱"):
                gr.Markdown(
                    "### 维权工具箱\n"
                    "> 独立快捷工具，无需对话直接使用。选择工具类型，输入信息，点击按钮即可。"
                )

                with gr.Row():
                    # 左列: 赔偿计算器 + 证据清单
                    with gr.Column():
                        gr.HTML('<div class="tool-card"><h3>赔偿金额计算器</h3>'
                                '<p>选择纠纷类型，输入购买金额，自动计算法定赔偿</p></div>')
                        comp_type = gr.Dropdown(
                            choices=["食品安全", "欺诈", "人身损害", "预付款", "产品质量"],
                            value="食品安全",
                            label="纠纷类型",
                        )
                        comp_amount = gr.Textbox(
                            label="购买金额（元）",
                            placeholder="如 12.8",
                        )
                        comp_loss = gr.Textbox(
                            label="实际损失（元，可选）",
                            placeholder="如 85（就医费用等）",
                        )
                        comp_btn = gr.Button("计算赔偿", variant="primary")
                        comp_out = gr.Textbox(label="计算结果", lines=12)

                        gr.HTML('<div class="tool-card"><h3>证据清单生成器</h3>'
                                '<p>选择纠纷类型，获取需要收集的证据清单</p></div>')
                        ev_type = gr.Dropdown(
                            choices=["食品安全", "网购欺诈", "格式条款", "预付卡", "服务质量"],
                            value="食品安全",
                            label="纠纷类型",
                        )
                        ev_btn = gr.Button("生成清单", variant="primary")
                        ev_out = gr.Textbox(label="证据清单", lines=12)

                    # 右列: 陷阱查询 + 信誉查询
                    with gr.Column():
                        gr.HTML('<div class="tool-card"><h3>消费陷阱预警</h3>'
                                '<p>选择行业，查看常见消费陷阱和防范建议</p></div>')
                        trap_industry = gr.Dropdown(
                            choices=["预付卡", "电商", "食品", "教育", "租房", "医美", "通信"],
                            value="预付卡",
                            label="行业类型",
                        )
                        trap_btn = gr.Button("查询陷阱", variant="primary")
                        trap_out = gr.Textbox(label="陷阱预警", lines=10)

                        gr.HTML('<div class="tool-card"><h3>商家信誉查询</h3>'
                                '<p>输入商家名称，查询历史投诉和信誉评分</p></div>')
                        rep_name = gr.Textbox(
                            label="商家名称",
                            placeholder="如 永辉超市、美团、拼多多",
                        )
                        rep_btn = gr.Button("查询信誉", variant="primary")
                        rep_out = gr.Textbox(label="信誉信息", lines=10)

                gr.HTML('<div class="ark-divider"></div>')

                with gr.Row():
                    # 左列: 维权时效
                    with gr.Column():
                        gr.HTML('<div class="tool-card"><h3>维权时效提醒</h3>'
                                '<p>查询法定维权期限，计算剩余天数</p></div>')
                        dl_type = gr.Dropdown(
                            choices=["七天无理由退货", "质量问题退货", "质量保修", "民事诉讼时效", "12315投诉时效"],
                            value="七天无理由退货",
                            label="时效类型",
                        )
                        dl_date = gr.Textbox(
                            label="购买日期（可选，格式 YYYY-MM-DD）",
                            placeholder="如 2026-07-01",
                        )
                        dl_btn = gr.Button("查询时效", variant="primary")
                        dl_out = gr.Textbox(label="时效信息", lines=10)

                    # 右列: 商家话术 + 维权路径
                    with gr.Column():
                        gr.HTML('<div class="tool-card"><h3>商家话术应对</h3>'
                                '<p>输入商家说的话，获取法律反驳和应对策略</p></div>')
                        tac_input = gr.Textbox(
                            label="商家话术",
                            placeholder='如 "特价商品概不退换"',
                        )
                        tac_btn = gr.Button("分析话术", variant="primary")
                        tac_out = gr.Textbox(label="法律反驳", lines=8)

                        gr.HTML('<div class="tool-card"><h3>维权路径规划</h3>'
                                '<p>描述纠纷情况，获取分步骤维权建议</p></div>')
                        path_input = gr.Textbox(
                            label="纠纷描述",
                            placeholder="如 超市买到过期食品",
                        )
                        path_btn = gr.Button("规划路径", variant="primary")
                        path_out = gr.Textbox(label="维权路径", lines=8)

                gr.HTML('<div class="ark-divider"></div>')

                # 邮件发送
                gr.HTML('<div class="tool-card"><h3>邮件发送</h3>'
                        '<p>将生成的投诉信、审查报告等维权文档通过邮件发送。'
                        '附件从 workspace/reports/ 目录选取。</p>'
                        '<br>'
                        '<p><b>确认流程</b>: 1) 填写信息 → 2) 点击「预览邮件」查看预览 → '
                        '3) 确认无误后勾选「我已确认邮件内容，同意发送」→ 4) 点击「发送邮件」</p></div>')
                with gr.Row():
                    with gr.Column():
                        email_to = gr.Textbox(
                            label="收件人邮箱",
                            placeholder="如 consumer@example.com",
                        )
                        email_subject = gr.Textbox(
                            label="邮件主题",
                            placeholder="如 消费维权投诉信 — 某某超市",
                        )
                    with gr.Column():
                        email_attachment = gr.Dropdown(
                            choices=get_report_files(),
                            label="附件（从已生成的报告中选择）",
                            interactive=True,
                        )
                        email_confirm = gr.Checkbox(
                            label="我已确认邮件内容，同意发送",
                            value=False,
                        )
                email_body = gr.Textbox(
                    label="邮件正文",
                    placeholder="简要说明投诉事由，并提示附件为投诉信全文...",
                    lines=4,
                )
                with gr.Row():
                    email_preview_btn = gr.Button("预览邮件", variant="primary")
                    email_send_btn = gr.Button("发送邮件", variant="stop")
                email_out = gr.Textbox(label="发送结果", lines=10)

                gr.HTML('<div class="ark-divider"></div>')

                # 对话摘要导出
                gr.HTML('<div class="tool-card"><h3>对话纪要导出</h3>'
                        '<p>将最近一次对话整理为结构化纪要文档，包含问题描述、法律依据、'
                        'Agent建议和后续行动。先在上方对话Tab中对话，再回到此处导出。</p></div>')
                summary_btn = gr.Button("导出对话纪要", variant="primary")
                summary_out = gr.Textbox(label="纪要内容", lines=20)

                # 绑定事件
                comp_btn.click(fn=tool_compensation, inputs=[comp_type, comp_amount, comp_loss], outputs=comp_out)
                ev_btn.click(fn=tool_evidence, inputs=ev_type, outputs=ev_out)
                trap_btn.click(fn=tool_trap, inputs=trap_industry, outputs=trap_out)
                rep_btn.click(fn=tool_reputation, inputs=rep_name, outputs=rep_out)
                dl_btn.click(fn=tool_deadline, inputs=[dl_type, dl_date], outputs=dl_out)
                tac_btn.click(fn=tool_tactics, inputs=tac_input, outputs=tac_out)
                path_btn.click(fn=tool_rights_path, inputs=path_input, outputs=path_out)
                summary_btn.click(fn=lambda: export_conversation_summary(_last_history), outputs=summary_out)

                # 邮件发送事件绑定
                email_preview_btn.click(
                    fn=lambda t, s, b, a, c: tool_send_email(t, s, b, a, False),
                    inputs=[email_to, email_subject, email_body, email_attachment, email_confirm],
                    outputs=email_out,
                )
                email_send_btn.click(
                    fn=tool_send_email,
                    inputs=[email_to, email_subject, email_body, email_attachment, email_confirm],
                    outputs=email_out,
                )

            # --- Tab 5: 系统信息 ---
            with gr.Tab("系统信息"):
                gr.Markdown("### 系统状态与创新功能")
                with gr.Row():
                    refresh_btn = gr.Button("刷新状态", variant="primary")
                info_box = gr.Markdown(get_system_info())
                refresh_btn.click(fn=refresh_files, outputs=info_box)

    return app


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_port=7860,
        server_name="127.0.0.1",
        css=ARK_CSS,
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate"),
    )
