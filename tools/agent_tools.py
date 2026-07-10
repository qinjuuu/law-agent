"""
Agent 协作工具模块
多Agent协作交接 + 维权进度追踪工具

这些工具让 Agent 能够:
1. 在对话中主动建议切换到更合适的智能体，并携带上下文
2. 查询和更新用户的维权进度状态
"""
from langchain.tools import tool


@tool
def handoff_to_agent(
    target_agent: str,
    reason: str,
    context_summary: str = "",
) -> str:
    """
    多Agent协作交接 — 当当前Agent检测到用户需求变化时，建议切换到更合适的智能体

    参数:
        target_agent: 目标智能体类型，可选值: complaint(投诉信起草)、review(条款审查)、qa(法律问答)
        reason: 切换原因，说明为什么建议切换
        context_summary: 当前对话的上下文摘要，传递给目标Agent

    返回:
        交接建议信息，告知用户系统将切换到更适合的智能体

    适用场景:
        - QA Agent 发现用户需要写投诉信时，交接给 Complaint Agent
        - QA Agent 发现用户需要审查合同时，交接给 Review Agent
        - Complaint Agent 发现用户在问法律问题时，交接给 QA Agent
    """
    agent_labels = {
        "complaint": "投诉信起草专家",
        "review": "条款审查专家",
        "qa": "消费法律问答助手",
    }

    label = agent_labels.get(target_agent, target_agent)

    result = f"""Agent协作交接
══════════════════════════════════
切换目标: {label}
切换原因: {reason}
"""
    if context_summary:
        result += f"\n上下文传递:\n  {context_summary}\n"

    result += f"""
══════════════════════════════════
系统已为您切换到「{label}」，您之前的对话信息会自动传递，无需重复描述。
"""
    return result


@tool
def get_rights_progress() -> str:
    """
    查询当前维权进度 — 获取用户在维权流程中的当前位置和下一步建议

    返回:
        维权进度信息，包含当前阶段、已完成步骤和下一步建议

    适用场景:
        用户想了解自己维权进行到哪一步、接下来该怎么做时调用
    """
    from agents.enhancements import CaseProgressTracker

    # 这个工具主要在非对话场景使用，返回维权阶段说明
    from agents.enhancements import _RIGHTS_STAGES

    lines = ["维权进度阶段说明", "=" * 40, ""]
    lines.append("维权全流程包含以下阶段:")
    lines.append("")

    for stage in _RIGHTS_STAGES:
        lines.append(f"第{stage['stage']}步: {stage['label']}")
        lines.append(f"  说明: {stage['desc']}")
        lines.append("")

    lines.append("=" * 40)
    lines.append("提示: 系统会根据您的对话内容自动追踪维权进度。")
    lines.append("您可以在综合问答中描述您已经做了什么，系统会判断您当前处于哪个阶段。")

    return "\n".join(lines)
