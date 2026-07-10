"""
创新工具模块
消费维权智能助手的创新功能工具集
包含赔偿预估、证据清单、维权路径、商家话术应对、时效提醒、
多平台适配、证据打包、陷阱预警、商家信誉查询
"""
import os
import sys
import json
import zipfile
from datetime import datetime, timedelta
from typing import List

from langchain.tools import tool

from config import WORK_ROOT, WORD_REPORTS_DIR, _PROJECT_ROOT

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ============================================================
# 知识库数据加载
# ============================================================
_KNOWLEDGE_DIR = os.path.join(_PROJECT_ROOT, "data", "knowledge")


def _load_knowledge(filename: str) -> str:
    """加载知识库文本文件"""
    filepath = os.path.join(_KNOWLEDGE_DIR, filename)
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# 赔偿规则数据（内置，避免每次读文件）
# ============================================================
_COMPENSATION_RULES = {
    "食品安全": {
        "law": "《食品安全法》第一百四十八条",
        "multiplier": 10,
        "loss_multiplier": 3,
        "minimum": 1000,
        "description": "生产不符合食品安全标准的食品或经营明知不符合标准的食品",
        "formula": "赔偿金 = max(购买价款 × 10, 实际损失 × 3, 1000元)",
    },
    "欺诈": {
        "law": "《消费者权益保护法》第五十五条",
        "multiplier": 3,
        "loss_multiplier": 0,
        "minimum": 500,
        "description": "经营者提供商品或服务有欺诈行为",
        "formula": "赔偿金 = max(购买价款 × 3, 500元)",
    },
    "人身损害": {
        "law": "《消费者权益保护法》第四十九条",
        "multiplier": 0,
        "loss_multiplier": 1,
        "minimum": 0,
        "description": "经营者提供商品或服务造成消费者人身伤害",
        "formula": "赔偿金 = 医疗费 + 护理费 + 交通费 + 误工费（按实际损失计算）",
    },
    "预付款": {
        "law": "《消费者权益保护法》第五十三条",
        "multiplier": 0,
        "loss_multiplier": 0,
        "minimum": 0,
        "description": "经营者以预收款方式提供服务，未按约定提供",
        "formula": "退回预付余额 + 预付款利息 + 已支付的合理费用",
    },
    "产品质量": {
        "law": "《产品质量法》第四十四条",
        "multiplier": 0,
        "loss_multiplier": 1,
        "minimum": 0,
        "description": "因产品缺陷造成人身或财产损害",
        "formula": "赔偿金 = 实际损失（人身伤害 + 财产损失）",
    },
}

# ============================================================
# 证据清单模板
# ============================================================
_EVIDENCE_CHECKLISTS = {
    "食品安全": [
        "购物小票或电子订单截图（证明购买行为和购买时间）",
        "过期/问题食品实物照片（需清晰显示生产日期、保质期、包装信息）",
        "食品问题部位特写照片（如异物、变质、发霉等）",
        "食用后身体不适的就医记录和诊断证明",
        "医院收费单据和药品购买凭证",
        "与商家沟通的聊天记录或通话录音",
        "12315投诉受理截图（如已投诉）",
    ],
    "网购欺诈": [
        "商品购买页面截图（含商品描述、宣传图片、价格信息）",
        "订单详情截图（订单号、购买时间、支付金额）",
        "收到的实物照片（与宣传进行对比）",
        "与商家的聊天记录截图（商家承诺、推诿等内容）",
        "商品鉴定报告或检测报告（如涉及假冒伪劣）",
        "物流信息截图（发货时间、签收时间）",
        "支付凭证截图",
    ],
    "格式条款": [
        "合同或协议原件照片或扫描件",
        "商家宣传材料截图（含承诺内容）",
        "签约过程的聊天记录或录音",
        "付款凭证（收据、转账记录）",
        "已履行部分的证明材料",
        "因条款不公造成的损失证明",
    ],
    "预付卡": [
        "会员卡或预付卡实物照片",
        "充值凭证和消费记录",
        "合同或办卡协议",
        "商家承诺的材料（宣传单页、聊天记录）",
        "剩余余额证明",
        "商家变更或关门的证据",
    ],
    "服务质量": [
        "服务合同或协议",
        "付款凭证",
        "服务过程的照片或视频",
        "服务效果不达标的证明材料",
        "与商家沟通的记录",
        "第三方评估或鉴定报告（如有）",
    ],
}

# ============================================================
# 维权路径模板
# ============================================================
_RIGHTS_PATHS = {
    "通用": [
        {"step": 1, "action": "与商家协商", "method": "直接联系商家客服或负责人，明确表达诉求", "duration": "3个工作日", "success_rate": "约40%", "tip": "保留沟通记录，协商不成不要纠缠"},
        {"step": 2, "action": "平台投诉", "method": "通过电商平台、应用商店等渠道发起投诉", "duration": "7个工作日", "success_rate": "约60%", "tip": "提供完整的订单信息和证据截图"},
        {"step": 3, "action": "12315投诉", "method": "拨打12315或通过全国12315平台在线投诉", "duration": "15个工作日", "success_rate": "约75%", "tip": "12315是市场监管部门官方渠道，效果较好"},
        {"step": 4, "action": "消费者协会调解", "method": "向当地消费者协会申请调解", "duration": "30个工作日", "success_rate": "约70%", "tip": "消协调解不收费，但无强制执行力"},
        {"step": 5, "action": "提起诉讼", "method": "向人民法院提起民事诉讼（小额诉讼程序）", "duration": "3-6个月", "success_rate": "约85%", "tip": "小额诉讼程序简便快捷，诉讼费低（通常2.5%）"},
    ],
    "食品安全": [
        {"step": 1, "action": "保留证据并联系商家", "method": "拍照保留过期/问题食品和小票，联系商家要求赔偿", "duration": "1-3天", "success_rate": "约50%", "tip": "食品安全问题商家通常愿意快速解决"},
        {"step": 2, "action": "12315投诉", "method": "向市场监管部门投诉，可主张退一赔十不足一千赔一千", "duration": "7-15天", "success_rate": "约80%", "tip": "明确引用《食品安全法》第148条"},
        {"step": 3, "action": "举报违法行为", "method": "向市场监管部门举报商家销售不合格食品", "duration": "30天", "success_rate": "约90%", "tip": "举报和投诉可以同时进行"},
        {"step": 4, "action": "提起诉讼", "method": "如有身体损害，可向法院提起损害赔偿诉讼", "duration": "3-6个月", "success_rate": "约85%", "tip": "保留就医记录，可主张医疗费和精神损害赔偿"},
    ],
    "网购": [
        {"step": 1, "action": "申请七天无理由退货", "method": "在平台直接申请退货，无需说明理由", "duration": "7天", "success_rate": "约90%", "tip": "收到商品7天内均可申请，商品需完好"},
        {"step": 2, "action": "平台介入", "method": "商家拒绝退货时申请平台客服介入", "duration": "3-7天", "success_rate": "约80%", "tip": "提供商品照片和问题描述"},
        {"step": 3, "action": "12315投诉", "method": "平台处理不满意可向12315投诉", "duration": "15天", "success_rate": "约70%", "tip": "提供平台订单号和沟通记录"},
        {"step": 4, "action": "提起诉讼", "method": "向商家所在地或合同履行地法院起诉", "duration": "3-6个月", "success_rate": "约80%", "tip": "网购纠纷可选择小额诉讼程序"},
    ],
}

# ============================================================
# 维权时效数据
# ============================================================
_DEADLINES = {
    "七天无理由退货": {"days": 7, "law": "《消费者权益保护法》第二十五条", "note": "自收到商品之日起7日内，网购商品默认享有无理由退货权"},
    "质量问题退货": {"days": 15, "law": "部分商品三包规定", "note": "部分商品出现性能故障可要求换货或退货，具体以三包规定为准"},
    "质量保修": {"days": 365, "law": "商品三包规定", "note": "多数商品保修期为1年，具体以商品三包凭证为准"},
    "民事诉讼时效": {"days": 1095, "law": "《民法典》第一百八十八条", "note": "向人民法院请求保护民事权利的诉讼时效期间为3年"},
    "12315投诉时效": {"days": 730, "law": "《消费者权益保护法》", "note": "建议在纠纷发生后2年内投诉，超过时限可能影响处理"},
}

# ============================================================
# 多平台投诉格式模板
# ============================================================
_PLATFORM_TEMPLATES = {
    "12315": {
        "name": "全国12315平台",
        "format": "简洁叙述型",
        "max_length": "500字以内",
        "required_fields": ["投诉人姓名", "联系电话", "被投诉方名称", "被投诉方地址", "投诉内容", "诉求"],
        "tip": "语言简练，直击重点，先说事实再说法条最后说诉求",
    },
    "消协": {
        "name": "消费者协会",
        "format": "详细叙述型",
        "max_length": "不限",
        "required_fields": ["投诉人信息", "被投诉方信息", "纠纷经过详细描述", "证据清单", "法律依据", "诉求"],
        "tip": "详细描述纠纷经过，附上全部证据材料复印件",
    },
    "法院": {
        "name": "人民法院起诉状",
        "format": "法律文书型",
        "max_length": "不限",
        "required_fields": ["原告信息", "被告信息", "诉讼请求", "事实与理由", "证据清单", "法律依据"],
        "tip": "需按照民事起诉状格式书写，诉讼请求需明确金额",
    },
    "电商平台": {
        "name": "电商平台投诉",
        "format": "订单格式型",
        "max_length": "300字以内",
        "required_fields": ["订单号", "商品名称", "问题描述", "期望处理方式"],
        "tip": "附上商品照片和订单截图，直接在订单页面发起投诉",
    },
}

# ============================================================
# 商家信誉模拟数据
# ============================================================
_MERCHANT_REPUTATION = {
    "永辉超市": {"complaints": 23, "resolved": 20, "rating": 3.5, "common_issues": ["过期食品", "价格欺诈"], "risk_level": "中"},
    "美团": {"complaints": 156, "resolved": 130, "rating": 3.2, "common_issues": ["退款纠纷", "配送问题"], "risk_level": "中"},
    "淘宝": {"complaints": 89, "resolved": 75, "rating": 3.8, "common_issues": ["假冒商品", "虚假宣传"], "risk_level": "低"},
    "京东": {"complaints": 45, "resolved": 42, "rating": 4.1, "common_issues": ["售后服务", "物流损坏"], "risk_level": "低"},
    "拼多多": {"complaints": 203, "resolved": 160, "rating": 2.9, "common_issues": ["假冒商品", "质量问题", "虚假宣传"], "risk_level": "高"},
    "默认": {"complaints": 0, "resolved": 0, "rating": 0, "common_issues": ["暂无数据"], "risk_level": "未知"},
}


# ============================================================
# 工具函数
# ============================================================

@tool
def estimate_compensation(dispute_type: str, purchase_amount: float, actual_loss: float = 0) -> str:
    """
    根据纠纷类型和购买金额智能预估赔偿金额

    参数:
        dispute_type: 纠纷类型，可选值: 食品安全、欺诈、人身损害、预付款、产品质量
        purchase_amount: 购买商品或服务的金额（元）
        actual_loss: 实际造成的损失金额（元），默认为0

    返回:
        赔偿预估结果，包含法律依据、计算公式和预估金额

    适用场景:
        用户询问能赔多少钱、想了解法定赔偿标准时调用
    """
    rule = _COMPENSATION_RULES.get(dispute_type)
    if not rule:
        available = "、".join(_COMPENSATION_RULES.keys())
        return f"暂不支持「{dispute_type}」类型的赔偿预估，目前支持的类型: {available}"

    # 计算赔偿金额
    amounts = []
    detail_lines = []

    if rule["multiplier"] > 0:
        penalty = purchase_amount * rule["multiplier"]
        amounts.append(penalty)
        detail_lines.append(f"价款{rule['multiplier']}倍赔偿: {purchase_amount} × {rule['multiplier']} = {penalty:.2f}元")

    if rule["loss_multiplier"] > 0 and actual_loss > 0:
        loss_penalty = actual_loss * rule["loss_multiplier"]
        amounts.append(loss_penalty)
        detail_lines.append(f"损失{rule['loss_multiplier']}倍赔偿: {actual_loss} × {rule['loss_multiplier']} = {loss_penalty:.2f}元")

    if actual_loss > 0 and rule["loss_multiplier"] == 0:
        amounts.append(actual_loss)
        detail_lines.append(f"实际损失赔偿: {actual_loss:.2f}元")

    if rule["minimum"] > 0:
        amounts.append(float(rule["minimum"]))
        detail_lines.append(f"法定最低赔偿: {rule['minimum']}元")

    final_amount = max(amounts) if amounts else 0

    result = f"""赔偿预估结果
══════════════════════════════════
纠纷类型: {dispute_type}
法律依据: {rule['law']}
适用情形: {rule['description']}

计算明细:
{chr(10).join(detail_lines)}

预估赔偿金额: {final_amount:.2f} 元
计算公式: {rule['formula']}

备注:
- 以上为法定最低赔偿标准，实际赔偿可能更高
- 如有人身损害，还可主张医疗费、误工费等
- 建议先与商家协商，协商不成可向12315投诉
══════════════════════════════════"""
    return result


@tool
def generate_evidence_checklist(dispute_type: str) -> str:
    """
    根据纠纷类型生成需要收集的证据清单

    参数:
        dispute_type: 纠纷类型，可选值: 食品安全、网购欺诈、格式条款、预付卡、服务质量

    返回:
        证据收集清单，包含每项证据的说明和收集建议

    适用场景:
        用户准备维权但不知道该收集哪些证据时调用
    """
    checklist = _EVIDENCE_CHECKLISTS.get(dispute_type)
    if not checklist:
        available = "、".join(_EVIDENCE_CHECKLISTS.keys())
        return f"暂不支持「{dispute_type}」类型的证据清单，目前支持: {available}"

    lines = [f"「{dispute_type}」纠纷证据收集清单", "=" * 40, ""]
    for i, item in enumerate(checklist, 1):
        lines.append(f"{i}. {item}")
    lines.append("")
    lines.append("证据收集建议:")
    lines.append("- 尽快拍照或截图，避免证据灭失")
    lines.append("- 保留原件，复印件备用")
    lines.append("- 电子证据注意保留原始载体（手机、电脑）")
    lines.append("- 证人证言尽量转化为书面形式")
    lines.append("- 所有沟通记录完整保留，不要删改")

    return "\n".join(lines)


@tool
def plan_rights_path(dispute_description: str) -> str:
    """
    根据纠纷描述规划维权路径，给出分步骤的行动建议

    参数:
        dispute_description: 纠纷情况描述，如"超市买到过期食品"、"网购假货商家不退款"

    返回:
        分步骤的维权路径规划，包含每步的操作方法、预计时间和成功率

    适用场景:
        用户不知道该怎么维权、想了解维权流程时调用
    """
    # 根据描述匹配路径类型
    desc = dispute_description.lower()
    if any(kw in desc for kw in ["食品", "过期", "变质", "吃坏", "异物"]):
        path_type = "食品安全"
    elif any(kw in desc for kw in ["网购", "快递", "退货", "七天", "电商"]):
        path_type = "网购"
    else:
        path_type = "通用"

    path = _RIGHTS_PATHS.get(path_type, _RIGHTS_PATHS["通用"])

    lines = [f"维权路径规划 — {path_type}类纠纷", "=" * 45, ""]
    lines.append(f"根据您描述的情况「{dispute_description}」，")
    lines.append("建议按以下步骤维权:")
    lines.append("")

    for step in path:
        lines.append(f"第{step['step']}步: {step['action']}")
        lines.append(f"  操作方法: {step['method']}")
        lines.append(f"  预计时间: {step['duration']}")
        lines.append(f"  成功率: {step['success_rate']}")
        lines.append(f"  提示: {step['tip']}")
        lines.append("")

    lines.append("=" * 45)
    lines.append("总结: 建议从第一步开始逐级升级，多数纠纷在第2-3步即可解决。")
    lines.append("如需法律援助，可拨打12348法律服务热线。")

    return "\n".join(lines)


@tool
def merchant_tactics_response(merchant_statement: str) -> str:
    """
    识别商家话术并给出法律反驳

    参数:
        merchant_statement: 商家说的话术，如"特价商品概不退换"、"最终解释权归本店所有"

    返回:
        商家话术的法律分析、反驳依据和应对策略

    适用场景:
        用户遇到商家推诿、不知道怎么反驳时调用
    """
    tactics_data = _load_knowledge("商家话术库.txt")
    if not tactics_data:
        return "话术库加载失败"

    # 按话术块解析
    blocks = tactics_data.split("【商家话术")
    matches = []

    for block in blocks[1:]:  # 跳过第一个空块
        block_text = "【商家话术" + block
        # 提取话术关键词
        quote_end = block.find("】")
        if quote_end > 0:
            quote = block[len("【商家话术"):quote_end]
            # 检查是否匹配
            quote_keywords = quote.replace("/", " ").replace("概不", "").replace("不", "")
            statement_lower = merchant_statement.lower()
            if any(kw in statement_lower for kw in [quote, quote.split("/")[0] if "/" in quote else quote]):
                matches.append(block_text.strip())

    if not matches:
        # 模糊匹配
        for block in blocks[1:]:
            block_text = "【商家话术" + block
            quote_end = block.find("】")
            if quote_end > 0:
                quote = block[len("【商家话术"):quote_end]
                # 取关键词
                keywords = [kw for kw in quote.replace("/", " ").replace("。", "").split() if len(kw) >= 2]
                if any(kw in merchant_statement for kw in keywords):
                    matches.append(block_text.strip())

    if not matches:
        result = f"""商家话术分析
══════════════════════════════════
商家说法: 「{merchant_statement}」

未在话术库中找到完全匹配的话术，但以下通用建议可供参考:

通用应对原则:
1. 商家不能以格式条款排除消费者法定权利
2. 根据《消费者权益保护法》第二十六条，不公平不合理的格式条款无效
3. 如商家拒绝协商，可拨打12315投诉
4. 保留所有沟通记录作为证据

建议您将商家的完整说法告诉我，我可以进一步分析。
══════════════════════════════════"""
        return result

    lines = [f"商家话术分析", "=" * 40, ""]
    for match in matches[:3]:  # 最多返回3条匹配
        lines.append(match)
        lines.append("")

    lines.append("=" * 40)
    lines.append("提示: 建议保留商家话术的截图或录音作为证据。")
    lines.append("如商家坚持不合理说法，可直接向12315投诉。")

    return "\n".join(lines)


@tool
def rights_deadline_reminder(dispute_type: str, purchase_date: str = "") -> str:
    """
    提醒维权的法定时效，计算剩余天数

    参数:
        dispute_type: 纠纷类型，可选值: 七天无理由退货、质量问题退货、质量保修、民事诉讼时效、12315投诉时效
        purchase_date: 购买日期，格式为 YYYY-MM-DD，未提供时使用今天

    返回:
        维权时效信息，包含法定期限、剩余天数和紧迫程度

    适用场景:
        用户想知道维权时效、是否还能退货或投诉时调用
    """
    deadline = _DEADLINES.get(dispute_type)
    if not deadline:
        available = "、".join(_DEADLINES.keys())
        return f"暂不支持「{dispute_type}」类型的时效查询，目前支持: {available}"

    # 解析购买日期
    if purchase_date:
        try:
            purchase_dt = datetime.strptime(purchase_date, "%Y-%m-%d")
        except ValueError:
            return f"日期格式错误，请使用 YYYY-MM-DD 格式，如 2026-07-05"
    else:
        purchase_dt = datetime.now()

    # 计算截止日期和剩余天数
    deadline_dt = purchase_dt + timedelta(days=deadline["days"])
    now = datetime.now()
    remaining_days = (deadline_dt - now).days

    # 紧迫程度
    if remaining_days < 0:
        urgency = "已过期"
        urgency_detail = f"该时效已于 {deadline_dt.strftime('%Y年%m月%d日')} 到期"
    elif remaining_days <= 3:
        urgency = "紧急"
        urgency_detail = f"仅剩 {remaining_days} 天，请立即行动！"
    elif remaining_days <= 7:
        urgency = "较紧急"
        urgency_detail = f"剩余 {remaining_days} 天，建议尽快处理"
    else:
        urgency = "充裕"
        urgency_detail = f"剩余 {remaining_days} 天，时间尚充裕"

    result = f"""维权时效提醒
══════════════════════════════════
时效类型: {dispute_type}
法律依据: {deadline['law']}
说明: {deadline['note']}

购买日期: {purchase_dt.strftime('%Y年%m月%d日')}
截止日期: {deadline_dt.strftime('%Y年%m月%d日')}
法定期限: {deadline['days']} 天
剩余天数: {remaining_days} 天
紧迫程度: 【{urgency}】{urgency_detail}

行动建议:
{"- 时效已过，但仍可尝试与商家协商解决" if remaining_days < 0 else "- 请在截止日期前采取行动，避免丧失权利"}
- 不同维权方式的时效可能不同，建议选择时效内的途径
- 如有疑问可拨打12315咨询
══════════════════════════════════"""
    return result


@tool
def multi_platform_complaint(
    dispute_info: str,
    platform: str,
    complainant_name: str = "",
    contact: str = "",
    merchant_name: str = "",
    demand: str = "",
) -> str:
    """
    根据纠纷信息和目标平台生成对应格式的投诉文书

    参数:
        dispute_info: 纠纷情况描述
        platform: 目标平台，可选值: 12315、消协、法院、电商平台
        complainant_name: 投诉人姓名
        contact: 联系电话
        merchant_name: 被投诉商家名称
        demand: 诉求描述

    返回:
        适配目标平台格式的投诉文书草稿

    适用场景:
        用户需要向不同平台投诉，需要对应格式的投诉文书时调用
    """
    template = _PLATFORM_TEMPLATES.get(platform)
    if not template:
        available = "、".join(_PLATFORM_TEMPLATES.keys())
        return f"暂不支持「{platform}」平台，目前支持: {available}"

    lines = [f"{template['name']}投诉文书草稿", "=" * 40, ""]
    lines.append(f"格式要求: {template['format']}")
    lines.append(f"字数限制: {template['max_length']}")
    lines.append(f"平台提示: {template['tip']}")
    lines.append("")

    if platform == "12315":
        lines.append(f"投诉人: {complainant_name or '（请填写姓名）'}")
        lines.append(f"联系电话: {contact or '（请填写电话）'}")
        lines.append(f"被投诉方: {merchant_name or '（请填写商家名称）'}")
        lines.append(f"被投诉方地址: （请填写商家地址）")
        lines.append("")
        lines.append("投诉内容:")
        lines.append(f"  {dispute_info}")
        lines.append("")
        lines.append(f"诉求: {demand or '请依法维护消费者合法权益'}")
        lines.append("")
        lines.append("投诉人签名: ___________")
        lines.append(f"日期: {datetime.now().strftime('%Y年%m月%d日')}")

    elif platform == "消协":
        lines.append("消费投诉书")
        lines.append("")
        lines.append(f"投诉人: {complainant_name or '（请填写姓名）'}")
        lines.append(f"联系电话: {contact or '（请填写电话）'}")
        lines.append(f"被投诉方: {merchant_name or '（请填写商家名称）'}")
        lines.append("")
        lines.append("一、纠纷经过")
        lines.append(f"  {dispute_info}")
        lines.append("")
        lines.append("二、证据清单")
        lines.append("  1. 购物凭证")
        lines.append("  2. 商品照片")
        lines.append("  3. 沟通记录")
        lines.append("  （请根据实际情况补充）")
        lines.append("")
        lines.append("三、法律依据")
        lines.append("  （请根据检索到的法条填写）")
        lines.append("")
        lines.append(f"四、诉求\n  {demand or '请依法调解，维护消费者合法权益'}")

    elif platform == "法院":
        lines.append("民事起诉状")
        lines.append("")
        lines.append(f"原告: {complainant_name or '（请填写姓名）'}")
        lines.append(f"联系电话: {contact or '（请填写电话）'}")
        lines.append(f"被告: {merchant_name or '（请填写商家名称）'}")
        lines.append("")
        lines.append("诉讼请求:")
        lines.append(f"  1. {demand or '请求被告赔偿损失'}")
        lines.append("  2. 本案诉讼费用由被告承担")
        lines.append("")
        lines.append("事实与理由:")
        lines.append(f"  {dispute_info}")
        lines.append("")
        lines.append("证据清单:")
        lines.append("  1. 购物凭证")
        lines.append("  2. 商品照片")
        lines.append("  3. 沟通记录")
        lines.append("")
        lines.append("此致")
        lines.append("__________人民法院")
        lines.append("")
        lines.append(f"具状人: {complainant_name or '___________'}")
        lines.append(f"日期: {datetime.now().strftime('%Y年%m月%d日')}")

    elif platform == "电商平台":
        lines.append(f"订单号: （请填写订单号）")
        lines.append(f"商品名称: {merchant_name or '（请填写商品名称）'}")
        lines.append("")
        lines.append("问题描述:")
        lines.append(f"  {dispute_info}")
        lines.append("")
        lines.append(f"期望处理方式: {demand or '退货退款'}")

    lines.append("")
    lines.append("=" * 40)
    lines.append("提示: 以上为草稿，请根据实际情况修改后提交。")

    return "\n".join(lines)


@tool
def package_evidence(
    complaint_filename: str = "",
    evidence_files: List[str] = None,
    output_name: str = "",
) -> str:
    """
    将投诉信、证据清单、法条引用等维权材料打包为一个 ZIP 文件

    参数:
        complaint_filename: 已生成的投诉信 Word 文件名（在 reports 目录下）
        evidence_files: 需要打包的证据文件名列表（在 workspace/files 目录下）
        output_name: 输出的 zip 文件名，未提供时自动生成

    返回:
        打包结果信息，包含文件路径和大小

    适用场景:
        用户准备去投诉或起诉，需要将所有维权材料打包时调用
    """
    try:
        if not output_name:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_name = f"维权材料包_{timestamp}.zip"

        if not output_name.endswith(".zip"):
            output_name += ".zip"

        output_path = os.path.join(WORD_REPORTS_DIR, output_name)

        file_count = 0
        total_size = 0

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # 打包投诉信
            if complaint_filename:
                complaint_path = os.path.join(WORD_REPORTS_DIR, complaint_filename)
                if not complaint_path.endswith(".docx"):
                    complaint_path += ".docx"
                if os.path.exists(complaint_path):
                    zf.write(complaint_path, arcname=f"投诉信/{complaint_filename}")
                    file_count += 1
                    total_size += os.path.getsize(complaint_path)

            # 打包证据文件
            if evidence_files:
                for ef in evidence_files:
                    ef_path = os.path.join(WORK_ROOT, ef)
                    if os.path.exists(ef_path):
                        if os.path.isfile(ef_path):
                            zf.write(ef_path, arcname=f"证据材料/{ef}")
                            file_count += 1
                            total_size += os.path.getsize(ef_path)
                        elif os.path.isdir(ef_path):
                            for root, dirs, files in os.walk(ef_path):
                                for f in files:
                                    full = os.path.join(root, f)
                                    arc = os.path.relpath(full, ef_path)
                                    zf.write(full, arcname=f"证据材料/{ef}/{arc}")
                                    file_count += 1
                                    total_size += os.path.getsize(full)

            # 自动生成证据清单
            checklist_content = "证据材料打包清单\n" + "=" * 40 + "\n\n"
            checklist_content += f"打包时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            checklist_content += f"文件数量: {file_count} 个\n"
            checklist_content += f"总大小: {total_size:,} 字节\n\n"
            checklist_content += "包含文件:\n"
            for info in zf.infolist():
                checklist_content += f"  - {info.filename} ({info.file_size:,} 字节)\n"
            zf.writestr("材料清单.txt", checklist_content.encode("utf-8"))
            file_count += 1

        zip_size = os.path.getsize(output_path)
        return f"维权材料包已生成: {output_name}\n包含 {file_count} 个文件，压缩后 {zip_size:,} 字节\n保存路径: {output_path}"

    except Exception as e:
        return f"打包失败: {str(e)}"


@tool
def trap_warning(industry: str) -> str:
    """
    查询指定行业的消费陷阱预警

    参数:
        industry: 行业类型，如 预付卡、电商、食品、教育、租房、医美、通信

    返回:
        该行业常见的消费陷阱、预警信号和防范建议

    适用场景:
        用户想了解某个行业有哪些消费陷阱时调用
        用户消费前想提前了解风险时调用
    """
    traps_data = _load_knowledge("消费陷阱库.txt")
    if not traps_data:
        return "陷阱库加载失败"

    # 按行业块解析
    industry_map = {
        "预付卡": "预付卡/会员卡",
        "会员卡": "预付卡/会员卡",
        "电商": "电商平台",
        "网购": "电商平台",
        "食品": "食品餐饮",
        "餐饮": "食品餐饮",
        "教育": "教育培训",
        "培训": "教育培训",
        "租房": "房屋租赁",
        "房屋": "房屋租赁",
        "医美": "医疗美容",
        "美容": "医疗美容",
        "通信": "通信服务",
        "运营商": "通信服务",
    }

    target = industry_map.get(industry, industry)

    # 按行业段落分割
    sections = traps_data.split("【行业:")
    matches = []

    for section in sections[1:]:
        section_full = "【行业:" + section
        section_end = section.find("】")
        if section_end > 0:
            section_name = section[:section_end]
            if target in section_name or section_name in target:
                matches.append(section_full.strip())

    if not matches:
        # 返回全部行业列表
        all_industries = []
        for section in sections[1:]:
            section_end = section.find("】")
            if section_end > 0:
                all_industries.append(section[:section_end])
        result = f"暂未找到「{industry}」行业的陷阱数据。\n\n目前收录的行业:\n"
        for ind in all_industries:
            result += f"  - {ind}\n"
        result += "\n请尝试以上关键词。"
        return result

    lines = [f"「{industry}」行业消费陷阱预警", "=" * 40, ""]
    for match in matches:
        lines.append(match)
        lines.append("")

    lines.append("=" * 40)
    lines.append("防范总则:")
    lines.append("- 消费前查验商家资质和信誉")
    lines.append("- 不轻信口头承诺，所有约定写入合同")
    lines.append("- 保留全部消费凭证和沟通记录")
    lines.append("- 发现问题及时维权，不要拖延")

    return "\n".join(lines)


@tool
def check_merchant_reputation(merchant_name: str) -> str:
    """
    查询商家的信誉信息和历史投诉记录

    先查本地数据库，本地无数据时自动联网搜索

    参数:
        merchant_name: 商家名称，如 永辉超市、美团、淘宝

    返回:
        商家信誉信息，包含投诉数量、解决率、评分、常见问题和风险等级

    适用场景:
        用户准备消费或投诉前想了解商家信誉时调用
    """
    # 模糊匹配本地数据
    matched = None
    for name, data in _MERCHANT_REPUTATION.items():
        if name in merchant_name or merchant_name in name:
            matched = (name, data)
            break

    if matched:
        name, data = matched
        resolved_rate = (data["resolved"] / data["complaints"] * 100) if data["complaints"] > 0 else 0

        result = f"""商家信誉查询
══════════════════════════════════
商家名称: {name}
数据来源: 本地信誉数据库
风险等级: 【{data["risk_level"]}】

投诉统计:
  历史投诉数: {data["complaints"]} 起
  已解决投诉: {data["resolved"]} 起
  解决率: {resolved_rate:.1f}%
  综合评分: {data["rating"]}/5.0

常见问题:
"""
        for issue in data["common_issues"]:
            result += f"  - {issue}\n"

        result += f"""
消费建议:
"""
        if data["risk_level"] == "高":
            result += "- 该商家投诉率较高，建议谨慎消费\n"
            result += "- 消费时保留全部凭证，以防维权需要\n"
            result += "- 建议优先选择其他信誉更好的商家\n"
        elif data["risk_level"] == "中":
            result += "- 该商家有一定投诉记录，消费时注意保留凭证\n"
            result += "- 遇到问题及时沟通，协商不成可投诉\n"
        else:
            result += "- 该商家信誉良好，正常消费即可\n"
            result += "- 仍建议保留消费凭证以备不时之需\n"

        result += "\n提示: 本地数据可能不是最新的，建议同时使用 search_merchant_info 联网搜索最新情况。"
        result += "\n══════════════════════════════════"
        return result

    # 本地无数据，自动联网搜索
    try:
        from tools.web_search_tools import search_merchant_info
        web_result = search_merchant_info.invoke({"merchant_name": merchant_name})

        result = f"""商家信誉查询
══════════════════════════════════
商家名称: {merchant_name}
数据来源: 联网搜索（实时）

本地信誉数据库未收录该商家，以下为联网搜索结果:

{web_result}

通用建议:
- 可在国家企业信用信息公示系统查询商家工商信息
- 可在天眼查、企查查等平台查询商家法律风险
- 可在全国12315平台查看商家历史投诉情况
- 消费前建议多方了解商家口碑
══════════════════════════════════"""
        return result
    except Exception as e:
        result = f"""商家信誉查询
══════════════════════════════════
查询商家: {merchant_name}

暂未收录该商家的信誉数据，联网搜索也未能获取信息。

已收录商家: {", ".join(k for k in _MERCHANT_REPUTATION.keys() if k != "默认")}

通用建议:
- 可在国家企业信用信息公示系统查询商家工商信息
- 可在天眼查、企查查等平台查询商家法律风险
- 可在全国12315平台查看商家历史投诉情况
- 消费前建议多方了解商家口碑
══════════════════════════════════"""
        return result
