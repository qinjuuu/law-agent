"""
消费维权智能助手 - 数据库管理器 (MySQL 版)

提供完整的数据库操作能力:
- 自动初始化 35 张表 + 4 个视图
- 从现有知识库文件自动填充种子数据
- 对话全流程数据记录（消息、工具调用、情绪、自反思等）
- 统计查询（工具调用排行、情绪分布、质量分布等）

设计原则:
- 单例模式，全局共享一个连接池
- 线程安全（threading.Lock）
- 非阻塞：所有写入操作 try-except，不影响主流程
- 懒加载：首次使用时自动初始化
"""
import json
import os
import threading
from datetime import datetime
from typing import Any, Optional

import pymysql
from pymysql.cursors import DictCursor

from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    _PROJECT_ROOT,
)

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


class DatabaseManager:
    """
    数据库管理器（单例）

    使用方式:
        from database import db
        db.log_message(conversation_id, "user", "你好")
        stats = db.get_stats()
    """

    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._connected = False
        return cls._instance

    def __init__(self):
        if self._connected:
            return
        self._write_lock = threading.Lock()
        self._connected = True
        self._init_db()
        self._seed_data()

    # ============================================================
    # 连接管理
    # ============================================================

    def _get_conn(self):
        """获取一个新的 MySQL 连接（每次操作用完即关，线程安全）"""
        return pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=True,
        )

    # ============================================================
    # 内部工具方法
    # ============================================================

    def _init_db(self):
        """读取 schema.sql 并执行，创建所有表和视图"""
        try:
            with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema_sql = f.read()
            conn = self._get_conn()
            # MySQL 不支持 executescript，按分号拆分逐条执行
            # 先去掉 SET 语句和注释行
            statements = []
            current_stmt = ""
            for line in schema_sql.split("\n"):
                stripped = line.strip()
                if stripped.startswith("--") or stripped.startswith("/*") or stripped.startswith("*/"):
                    continue
                current_stmt += line + "\n"
                if stripped.endswith(";"):
                    stmt = current_stmt.strip()
                    if stmt and not stmt.upper().startswith("SET "):
                        statements.append(stmt)
                    current_stmt = ""
            if current_stmt.strip():
                statements.append(current_stmt.strip())

            for stmt in statements:
                try:
                    cursor = conn.cursor()
                    cursor.execute(stmt)
                except Exception as ex:
                    # 忽略 "已存在" 类的错误
                    err_msg = str(ex).lower()
                    if "already exists" not in err_msg and "duplicate" not in err_msg:
                        print(f"[DB] SQL 警告: {ex}")
            conn.close()

            # 写入元信息
            self._execute_write(
                "INSERT INTO db_meta (`key`, `value`) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE `value` = VALUES(`value`), updated_at = NOW()",
                ("version", "1.0"),
            )
            self._execute_write(
                "INSERT INTO db_meta (`key`, `value`) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE `value` = VALUES(`value`), updated_at = NOW()",
                ("initialized_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            print(f"[DB] MySQL 数据库初始化完成: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
        except Exception as e:
            print(f"[DB] 初始化失败: {e}")

    def _seed_data(self):
        """从现有知识库文件填充种子数据"""
        try:
            row = self._execute_query(
                "SELECT `value` FROM db_meta WHERE `key` = %s", ("seeded",)
            )
            if row and row[0]["value"] == "1":
                return

            self._seed_compensation_rules()
            self._seed_evidence_templates()
            self._seed_rights_path_templates()
            self._seed_platform_templates()
            self._seed_rights_stages()
            self._seed_deadline_rules()
            self._seed_laws()
            self._seed_traps()

            self._execute_write(
                "INSERT INTO db_meta (`key`, `value`) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE `value` = VALUES(`value`)",
                ("seeded", "1"),
            )
            print("[DB] 种子数据填充完成")
        except Exception as e:
            print(f"[DB] 种子数据填充失败: {e}")

    def _seed_compensation_rules(self):
        """填充赔偿规则"""
        rules = [
            ("食品安全", "《食品安全法》第一百四十八条", 10, 3, 1000,
             "生产不符合食品安全标准的食品或经营明知不符合标准的食品",
             "赔偿金 = max(购买价款 x 10, 实际损失 x 3, 1000元)"),
            ("欺诈", "《消费者权益保护法》第五十五条", 3, 0, 500,
             "经营者提供商品或服务有欺诈行为",
             "赔偿金 = max(购买价款 x 3, 500元)"),
            ("人身损害", "《消费者权益保护法》第四十九条", 0, 1, 0,
             "经营者提供商品或服务造成消费者人身伤害",
             "赔偿金 = 医疗费 + 护理费 + 交通费 + 误工费"),
            ("预付款", "《消费者权益保护法》第五十三条", 0, 0, 0,
             "经营者以预收款方式提供服务，未按约定提供",
             "退回预付余额 + 预付款利息 + 已支付的合理费用"),
            ("产品质量", "《产品质量法》第四十四条", 0, 1, 0,
             "因产品缺陷造成人身或财产损害",
             "赔偿金 = 实际损失（人身伤害 + 财产损失）"),
        ]
        for r in rules:
            self._execute_write(
                "INSERT INTO compensation_rules "
                "(dispute_type, law, multiplier, loss_multiplier, minimum, description, formula) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE law=VALUES(law), multiplier=VALUES(multiplier), "
                "loss_multiplier=VALUES(loss_multiplier), minimum=VALUES(minimum), "
                "description=VALUES(description), formula=VALUES(formula)",
                r,
            )

    def _seed_evidence_templates(self):
        """填充证据模板"""
        templates = {
            "食品安全": [
                "购物小票或电子订单截图",
                "过期/问题食品实物照片",
                "食品问题部位特写照片",
                "食用后身体不适的就医记录",
                "医院收费单据和药品购买凭证",
                "与商家沟通的聊天记录或通话录音",
                "12315投诉受理截图",
            ],
            "网购欺诈": [
                "商品购买页面截图",
                "订单详情截图",
                "收到的实物照片",
                "与商家的聊天记录截图",
                "商品鉴定报告或检测报告",
                "物流信息截图",
                "支付凭证截图",
            ],
            "格式条款": [
                "合同或协议原件照片",
                "商家宣传材料截图",
                "签约过程的聊天记录或录音",
                "付款凭证",
                "已履行部分的证明材料",
                "因条款不公造成的损失证明",
            ],
            "预付卡": [
                "会员卡或预付卡实物照片",
                "充值凭证和消费记录",
                "合同或办卡协议",
                "商家承诺的材料",
                "剩余余额证明",
                "商家变更或关门的证据",
            ],
            "服务质量": [
                "服务合同或协议",
                "付款凭证",
                "服务过程的照片或视频",
                "服务效果不达标的证明材料",
                "与商家沟通的记录",
                "第三方评估或鉴定报告",
            ],
        }
        for dtype, items in templates.items():
            for i, item in enumerate(items, 1):
                self._execute_write(
                    "INSERT INTO evidence_templates (dispute_type, item_order, item_description) VALUES (%s,%s,%s)",
                    (dtype, i, item),
                )

    def _seed_rights_path_templates(self):
        """填充维权路径模板"""
        paths = {
            "通用": [
                (1, "与商家协商", "直接联系商家客服或负责人，明确表达诉求", "3个工作日", "约40%", "保留沟通记录，协商不成不要纠缠"),
                (2, "平台投诉", "通过电商平台、应用商店等渠道发起投诉", "7个工作日", "约60%", "提供完整的订单信息和证据截图"),
                (3, "12315投诉", "拨打12315或通过全国12315平台在线投诉", "15个工作日", "约75%", "12315是市场监管部门官方渠道，效果较好"),
                (4, "消费者协会调解", "向当地消费者协会申请调解", "30个工作日", "约70%", "消协调解不收费，但无强制执行力"),
                (5, "提起诉讼", "向人民法院提起民事诉讼（小额诉讼程序）", "3-6个月", "约85%", "小额诉讼程序简便快捷，诉讼费低"),
            ],
            "食品安全": [
                (1, "保留证据并联系商家", "拍照保留过期/问题食品和小票", "1-3天", "约50%", "食品安全问题商家通常愿意快速解决"),
                (2, "12315投诉", "向市场监管部门投诉，可主张退一赔十", "7-15天", "约80%", "明确引用《食品安全法》第148条"),
                (3, "举报违法行为", "向市场监管部门举报商家销售不合格食品", "30天", "约90%", "举报和投诉可以同时进行"),
                (4, "提起诉讼", "如有身体损害，可向法院提起损害赔偿诉讼", "3-6个月", "约85%", "保留就医记录，可主张医疗费和精神损害赔偿"),
            ],
            "网购": [
                (1, "申请七天无理由退货", "在平台直接申请退货", "7天", "约90%", "收到商品7天内均可申请，商品需完好"),
                (2, "平台介入", "商家拒绝退货时申请平台客服介入", "3-7天", "约80%", "提供商品照片和问题描述"),
                (3, "12315投诉", "平台处理不满意可向12315投诉", "15天", "约70%", "提供平台订单号和沟通记录"),
                (4, "提起诉讼", "向商家所在地或合同履行地法院起诉", "3-6个月", "约80%", "网购纠纷可选择小额诉讼程序"),
            ],
        }
        for ptype, steps in paths.items():
            for step in steps:
                self._execute_write(
                    "INSERT INTO rights_path_templates (path_type, step_number, action, method, duration, success_rate, tip) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (ptype, *step),
                )

    def _seed_platform_templates(self):
        """填充平台投诉格式模板"""
        platforms = [
            ("12315", "全国12315平台", "简洁叙述型", "500字以内",
             json.dumps(["投诉人姓名", "联系电话", "被投诉方名称", "被投诉方地址", "投诉内容", "诉求"]),
             "语言简练，直击重点"),
            ("消协", "消费者协会", "详细叙述型", "不限",
             json.dumps(["投诉人信息", "被投诉方信息", "纠纷经过详细描述", "证据清单", "法律依据", "诉求"]),
             "详细描述纠纷经过，附上全部证据材料复印件"),
            ("法院", "人民法院起诉状", "法律文书型", "不限",
             json.dumps(["原告信息", "被告信息", "诉讼请求", "事实与理由", "证据清单", "法律依据"]),
             "需按照民事起诉状格式书写"),
            ("电商平台", "电商平台投诉", "订单格式型", "300字以内",
             json.dumps(["订单号", "商品名称", "问题描述", "期望处理方式"]),
             "附上商品照片和订单截图"),
        ]
        for p in platforms:
            self._execute_write(
                "INSERT INTO platform_templates (platform_key, platform_name, format_type, max_length, required_fields, tip) "
                "VALUES (%s,%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE platform_name=VALUES(platform_name), format_type=VALUES(format_type), "
                "max_length=VALUES(max_length), required_fields=VALUES(required_fields), tip=VALUES(tip)",
                p,
            )

    def _seed_rights_stages(self):
        """填充维权阶段定义"""
        stages = [
            (1, "consult", "咨询了解", "了解自己的权益和维权方式",
             json.dumps(["咨询", "了解", "问一下", "怎么办", "能维权吗", "合法吗"])),
            (2, "negotiate", "与商家协商", "直接联系商家沟通解决",
             json.dumps(["协商", "沟通", "找了商家", "联系了", "商家说", "商家拒绝", "商家同意"])),
            (3, "platform", "平台投诉", "通过电商平台或12315投诉",
             json.dumps(["12315", "平台投诉", "已经投诉", "投诉了", "举报"])),
            (4, "mediate", "消协调解", "向消费者协会申请调解",
             json.dumps(["消协", "调解", "消费者协会"])),
            (5, "litigate", "提起诉讼", "向人民法院提起民事诉讼",
             json.dumps(["起诉", "法院", "诉讼", "打官司", "立案"])),
            (6, "resolved", "维权完成", "问题已解决",
             json.dumps(["解决了", "退了", "赔了", "已退款", "已赔偿", "搞定了"])),
        ]
        for s in stages:
            self._execute_write(
                "INSERT INTO rights_stages (stage_number, stage_key, stage_label, description, stage_keywords) "
                "VALUES (%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE stage_key=VALUES(stage_key), stage_label=VALUES(stage_label), "
                "description=VALUES(description), stage_keywords=VALUES(stage_keywords)",
                s,
            )

    def _seed_deadline_rules(self):
        """填充时效规则"""
        rules = [
            ("七天无理由退货", 7, "《消费者权益保护法》第二十五条", "自收到商品之日起7日内，网购商品默认享有无理由退货权"),
            ("质量问题退货", 15, "部分商品三包规定", "部分商品出现性能故障可要求换货或退货"),
            ("质量保修", 365, "商品三包规定", "多数商品保修期为1年"),
            ("民事诉讼时效", 1095, "《民法典》第一百八十八条", "向人民法院请求保护民事权利的诉讼时效期间为3年"),
            ("12315投诉时效", 730, "《消费者权益保护法》", "建议在纠纷发生后2年内投诉"),
        ]
        for r in rules:
            self._execute_write(
                "INSERT INTO deadline_rules (deadline_type, days, law, note) VALUES (%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE days=VALUES(days), law=VALUES(law), note=VALUES(note)",
                r,
            )

    def _seed_laws(self):
        """从法律文本文件填充法律条文"""
        laws_dir = os.path.join(_PROJECT_ROOT, "data", "laws")
        if not os.path.exists(laws_dir):
            return

        for filename in os.listdir(laws_dir):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(laws_dir, filename)
            law_name = filename.replace(".txt", "")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                import re
                articles = re.split(r"(第[一二三四五六七八九十百千零\d]+条)", content)
                current_article = ""
                current_number = ""

                for part in articles:
                    part = part.strip()
                    if re.match(r"^第[一二三四五六七八九十百千零\d]+条$", part):
                        if current_number and current_article:
                            self._execute_write(
                                "INSERT INTO laws (law_name, article_number, content, source_file) VALUES (%s,%s,%s,%s)",
                                (law_name, current_number, current_article[:2000], filename),
                            )
                        current_number = part
                        current_article = ""
                    else:
                        current_article += part

                if current_number and current_article:
                    self._execute_write(
                        "INSERT INTO laws (law_name, article_number, content, source_file) VALUES (%s,%s,%s,%s)",
                        (law_name, current_number, current_article[:2000], filename),
                    )
            except Exception:
                pass

    def _seed_traps(self):
        """从消费陷阱库文件填充陷阱数据"""
        traps_path = os.path.join(_PROJECT_ROOT, "data", "knowledge", "消费陷阱库.txt")
        if not os.path.exists(traps_path):
            return
        try:
            with open(traps_path, "r", encoding="utf-8") as f:
                content = f.read()

            import re
            sections = re.split(r"【行业:([^\】]+)】", content)
            for i in range(1, len(sections), 2):
                if i + 1 < len(sections):
                    industry = sections[i].strip()
                    trap_text = sections[i + 1].strip()[:2000]
                    self._execute_write(
                        "INSERT INTO trap_kb (industry, trap_description) VALUES (%s,%s)",
                        (industry, trap_text),
                    )
        except Exception:
            pass

    def _execute_write(self, sql: str, params: tuple = ()):
        """执行写操作（线程安全）"""
        try:
            with self._write_lock:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute(sql, params)
                last_id = cursor.lastrowid
                conn.close()
                return last_id
        except Exception as e:
            print(f"[DB] 写入失败: {e}")
            return None

    def _execute_query(self, sql: str, params: tuple = ()):
        """执行查询操作（线程安全）"""
        try:
            with self._write_lock:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                conn.close()
                return rows
        except Exception as e:
            print(f"[DB] 查询失败: {e}")
            return []

    def _json_dumps(self, obj) -> str:
        """安全序列化 JSON"""
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return "[]"

    # ============================================================
    # 用户与会话管理
    # ============================================================

    def get_or_create_user(self, session_id: str) -> int:
        """获取或创建用户，返回 user_id"""
        row = self._execute_query(
            "SELECT id FROM users WHERE session_id = %s", (session_id,)
        )
        if row:
            self._execute_write(
                "UPDATE users SET last_active = NOW(), total_messages = total_messages + 1 WHERE id = %s",
                (row[0]["id"],),
            )
            return row[0]["id"]

        user_id = self._execute_write(
            "INSERT INTO users (session_id) VALUES (%s)",
            (session_id,),
        )
        return user_id or 0

    def create_conversation(self, session_id: str, agent_type: str, title: str = "") -> int:
        """创建对话会话，返回 conversation_id"""
        user_id = self.get_or_create_user(session_id)
        conv_id = self._execute_write(
            "INSERT INTO conversations (user_id, session_id, agent_type, title) VALUES (%s,%s,%s,%s)",
            (user_id, session_id, agent_type, title),
        )
        return conv_id or 0

    def log_message(self, conversation_id: int, role: str, content: str,
                    agent_type: str = None, response_time_ms: int = None) -> int:
        """记录一条消息"""
        msg_id = self._execute_write(
            "INSERT INTO messages (conversation_id, role, content, content_length, agent_type, response_time_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (conversation_id, role, content, len(content) if content else 0,
             agent_type, response_time_ms),
        )
        if conversation_id:
            self._execute_write(
                "UPDATE conversations SET message_count = message_count + 1, updated_at = NOW() WHERE id = %s",
                (conversation_id,),
            )
        return msg_id or 0

    # ============================================================
    # Agent 处理流程记录
    # ============================================================

    def log_intent_route(self, conversation_id: int, message_id: int,
                         user_message: str, detected_intent: str,
                         routed_agent: str, confidence: float = None):
        """记录意图路由"""
        self._execute_write(
            "INSERT INTO intent_routes (conversation_id, message_id, user_message, detected_intent, routed_agent, confidence) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id, user_message[:500], detected_intent, routed_agent, confidence),
        )

    def log_user_profile(self, conversation_id: int, message_id: int, profile: dict):
        """记录用户画像"""
        self._execute_write(
            "INSERT INTO user_profiles (conversation_id, message_id, legal_level, urgency, user_type, style_hint) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id,
             profile.get("legal_level", ""), profile.get("urgency", ""),
             profile.get("user_type", ""), profile.get("style_hint", "")),
        )

    def log_emotion(self, conversation_id: int, message_id: int,
                    emotion: str, emotion_label: str = "",
                    keywords: list = None, soothing: str = "", action_hint: str = ""):
        """记录情绪检测"""
        self._execute_write(
            "INSERT INTO emotion_records (conversation_id, message_id, emotion_type, emotion_label, keywords_matched, soothing_text, action_hint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id, emotion, emotion_label,
             self._json_dumps(keywords or []), soothing, action_hint),
        )

    def log_completeness(self, conversation_id: int, message_id: int, completeness: dict):
        """记录信息完整性"""
        self._execute_write(
            "INSERT INTO completeness_records (conversation_id, message_id, agent_type, completeness_pct, provided_fields, missing_fields, should_ask, ask_hint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id, completeness.get("agent_type", ""),
             completeness.get("completeness", 0),
             self._json_dumps(completeness.get("provided_fields", [])),
             self._json_dumps(completeness.get("missing_fields", [])),
             1 if completeness.get("should_ask") else 0,
             completeness.get("ask_hint", "")),
        )

    def log_tool_call(self, conversation_id: int, message_id: int,
                      tool_name: str, tool_label: str = "",
                      tool_args: dict = None, tool_result: str = "",
                      duration_ms: int = None, success: bool = True,
                      error_message: str = ""):
        """记录工具调用"""
        self._execute_write(
            "INSERT INTO tool_calls (conversation_id, message_id, tool_name, tool_label, tool_args, tool_result, result_length, duration_ms, success, error_message) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id, tool_name, tool_label,
             self._json_dumps(tool_args or {}),
             tool_result[:2000] if tool_result else "", len(tool_result) if tool_result else 0,
             duration_ms, 1 if success else 0, error_message),
        )

    def log_reasoning_chain(self, conversation_id: int, message_id: int,
                            chain_text: str, tools_used: list,
                            emotion: str = "", step_count: int = 0):
        """记录思维链"""
        self._execute_write(
            "INSERT INTO reasoning_chains (conversation_id, message_id, chain_text, step_count, tools_used, emotion_detected) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id, chain_text, step_count,
             self._json_dumps(tools_used), emotion),
        )

    def log_reflection(self, conversation_id: int, message_id: int, reflection: dict):
        """记录自反思"""
        self._execute_write(
            "INSERT INTO self_reflections (conversation_id, message_id, score, passed, quality_label, issues, suggestion) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id, reflection.get("score", 0),
             1 if reflection.get("pass") else 0, reflection.get("quality_label", ""),
             self._json_dumps(reflection.get("issues", [])),
             reflection.get("suggestion", "")),
        )

    def log_confidence(self, conversation_id: int, message_id: int, confidence: dict):
        """记录置信度评估"""
        self._execute_write(
            "INSERT INTO confidence_assessments (conversation_id, message_id, confidence_level, confidence_label, confidence_emoji, reason, is_out_of_scope, boundary_warning, has_law_reference, law_signal_count) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id,
             confidence.get("confidence", ""), confidence.get("confidence_label", ""),
             confidence.get("confidence_emoji", ""), confidence.get("reason", ""),
             1 if confidence.get("is_out_of_scope") else 0,
             confidence.get("boundary_warning", ""),
             1 if confidence.get("has_law_ref") else 0,
             confidence.get("law_signal_count", 0)),
        )

    def log_progress(self, conversation_id: int, message_id: int, progress: dict):
        """记录维权进度"""
        self._execute_write(
            "INSERT INTO case_progress (conversation_id, message_id, current_stage, current_label, stages_completed, next_action, progress_bar, progress_pct) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id, progress.get("current_stage", 1),
             progress.get("current_label", ""),
             self._json_dumps(progress.get("stages_completed", [])),
             progress.get("next_action", ""), progress.get("progress_bar", ""),
             progress.get("progress_pct", 0)),
        )

    def log_handoff(self, conversation_id: int, message_id: int,
                    from_agent: str, to_agent: str, reason: str, context: str = ""):
        """记录 Agent 交接"""
        self._execute_write(
            "INSERT INTO agent_handoffs (conversation_id, message_id, from_agent, to_agent, reason, context_summary) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id, from_agent, to_agent, reason, context[:500]),
        )

    # ============================================================
    # 业务记录
    # ============================================================

    def log_compensation(self, conversation_id: int, dispute_type: str,
                         purchase_amount: float, actual_loss: float,
                         estimated_amount: float, legal_basis: str,
                         formula: str, detail: str = ""):
        """记录赔偿预估"""
        self._execute_write(
            "INSERT INTO compensation_estimates (conversation_id, dispute_type, purchase_amount, actual_loss, estimated_amount, legal_basis, formula, calculation_detail) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, dispute_type, purchase_amount, actual_loss,
             estimated_amount, legal_basis, formula, detail[:1000]),
        )

    def log_evidence_checklist(self, conversation_id: int, dispute_type: str, items: list):
        """记录证据清单"""
        self._execute_write(
            "INSERT INTO evidence_checklists (conversation_id, dispute_type, checklist_items, item_count) "
            "VALUES (%s,%s,%s,%s)",
            (conversation_id, dispute_type, self._json_dumps(items), len(items)),
        )

    def log_rights_path(self, conversation_id: int, path_type: str,
                        dispute_description: str, steps: list):
        """记录维权路径"""
        self._execute_write(
            "INSERT INTO rights_paths (conversation_id, path_type, dispute_description, steps, step_count) "
            "VALUES (%s,%s,%s,%s,%s)",
            (conversation_id, path_type, dispute_description[:500],
             self._json_dumps(steps), len(steps)),
        )

    def log_merchant_tactics(self, conversation_id: int, statement: str, refutation: str):
        """记录商家话术应对"""
        self._execute_write(
            "INSERT INTO merchant_tactics (conversation_id, merchant_statement, legal_refutation) "
            "VALUES (%s,%s,%s)",
            (conversation_id, statement[:500], refutation[:2000]),
        )

    def log_deadline(self, conversation_id: int, deadline_type: str,
                     purchase_date: str, deadline_date: str,
                     remaining_days: int, urgency: str, legal_basis: str):
        """记录时效提醒"""
        self._execute_write(
            "INSERT INTO deadline_reminders (conversation_id, deadline_type, purchase_date, deadline_date, remaining_days, urgency_level, legal_basis) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, deadline_type, purchase_date, deadline_date,
             remaining_days, urgency, legal_basis),
        )

    def log_merchant_reputation(self, merchant_name: str, data: dict):
        """记录/更新商家信誉查询"""
        existing = self._execute_query(
            "SELECT id FROM merchant_reputations WHERE merchant_name = %s", (merchant_name,)
        )
        if existing:
            self._execute_write(
                "UPDATE merchant_reputations SET query_count = query_count + 1, queried_at = NOW() WHERE id = %s",
                (existing[0]["id"],),
            )
        else:
            resolve_rate = 0
            if data.get("complaints", 0) > 0:
                resolve_rate = data.get("resolved", 0) / data["complaints"] * 100
            self._execute_write(
                "INSERT INTO merchant_reputations (merchant_name, complaints, resolved, resolve_rate, rating, common_issues, risk_level) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (merchant_name, data.get("complaints", 0), data.get("resolved", 0),
                 resolve_rate, data.get("rating", 0),
                 self._json_dumps(data.get("common_issues", [])),
                 data.get("risk_level", "")),
            )

    def log_trap_warning(self, conversation_id: int, industry: str, trap_data: str):
        """记录消费陷阱查询"""
        self._execute_write(
            "INSERT INTO trap_warnings (conversation_id, industry, trap_data) VALUES (%s,%s,%s)",
            (conversation_id, industry, trap_data[:2000]),
        )

    def log_document(self, conversation_id: int, doc_type: str,
                     filename: str, filepath: str, file_size: int = 0):
        """记录文档生成"""
        self._execute_write(
            "INSERT INTO documents (conversation_id, doc_type, filename, filepath, file_size) "
            "VALUES (%s,%s,%s,%s,%s)",
            (conversation_id, doc_type, filename, filepath, file_size),
        )

    def log_conversation_summary(self, conversation_id: int, summary_text: str,
                                 file_path: str = "", message_count: int = 0):
        """记录对话摘要"""
        self._execute_write(
            "INSERT INTO conversation_summaries (conversation_id, summary_text, file_path, message_count) "
            "VALUES (%s,%s,%s,%s)",
            (conversation_id, summary_text, file_path, message_count),
        )

    def log_complaint(self, conversation_id: int, data: dict, file_path: str = "", file_size: int = 0):
        """记录投诉信"""
        self._execute_write(
            "INSERT INTO complaints (conversation_id, complainant_name, contact, merchant_name, merchant_address, "
            "purchase_time, product_name, purchase_amount, dispute_detail, demand, legal_basis, file_path, file_size) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, data.get("complainant_name", ""), data.get("contact", ""),
             data.get("merchant_name", ""), data.get("merchant_address", ""),
             data.get("purchase_time", ""), data.get("product_name", ""),
             data.get("purchase_amount", 0), data.get("dispute_detail", ""),
             data.get("demand", ""), data.get("legal_basis", ""),
             file_path, file_size),
        )

    def log_email(self, conversation_id: int, message_id: int,
                  from_email: str, to_email: str, subject: str,
                  body_preview: str = "", attachments: list = None,
                  status: str = "sent", error_message: str = ""):
        """记录邮件发送日志"""
        self._execute_write(
            "INSERT INTO email_logs (conversation_id, message_id, from_email, to_email, subject, "
            "body_preview, attachments, attachment_count, status, error_message) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, message_id, from_email, to_email, subject,
             body_preview[:500] if body_preview else "",
             self._json_dumps(attachments or []), len(attachments or []),
             status, error_message[:500] if error_message else ""),
        )

    def log_clause_review(self, conversation_id: int, data: dict, file_path: str = "", file_size: int = 0):
        """记录条款审查"""
        self._execute_write(
            "INSERT INTO clause_reviews (conversation_id, contract_title, clause_content, risk_level, risk_score, "
            "risk_items, review_results, suggestions, legal_basis, file_path, file_size) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (conversation_id, data.get("contract_title", ""), data.get("clause_content", ""),
             data.get("risk_level", ""), data.get("risk_score", 0),
             self._json_dumps(data.get("risk_items", [])),
             data.get("review_results", ""), data.get("suggestions", ""),
             data.get("legal_basis", ""), file_path, file_size),
        )

    # ============================================================
    # 统计查询
    # ============================================================

    def get_table_count(self, table_name: str) -> int:
        """获取表行数"""
        rows = self._execute_query(f"SELECT COUNT(*) AS cnt FROM `{table_name}`")
        return rows[0]["cnt"] if rows else 0

    def get_stats(self) -> dict:
        """获取数据库全局统计"""
        tables = [
            "users", "conversations", "messages", "intent_routes",
            "user_profiles", "emotion_records", "completeness_records",
            "tool_calls", "reasoning_chains", "self_reflections",
            "confidence_assessments", "case_progress", "agent_handoffs",
            "complaints", "clause_reviews", "compensation_estimates",
            "evidence_checklists", "rights_paths", "merchant_tactics",
            "deadline_reminders", "merchant_reputations", "trap_warnings",
            "documents", "conversation_summaries", "email_logs",
            "laws", "case_precedents", "compensation_rules",
            "evidence_templates", "rights_path_templates", "platform_templates",
            "rights_stages", "deadline_rules", "merchant_tactics_kb", "trap_kb",
        ]
        counts = {}
        total = 0
        for t in tables:
            cnt = self.get_table_count(t)
            counts[t] = cnt
            total += cnt
        return {"tables": counts, "total_records": total, "table_count": len(tables)}

    def get_tool_call_stats(self) -> list:
        """获取工具调用统计"""
        rows = self._execute_query(
            "SELECT tool_name, tool_label, COUNT(*) AS call_count, "
            "SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) AS success_count, "
            "ROUND(AVG(duration_ms), 0) AS avg_ms, "
            "MAX(created_at) AS last_called "
            "FROM tool_calls GROUP BY tool_name, tool_label ORDER BY call_count DESC"
        )
        return [dict(r) for r in rows]

    def get_emotion_stats(self) -> list:
        """获取情绪分布统计"""
        rows = self._execute_query(
            "SELECT emotion_type, emotion_label, COUNT(*) AS count "
            "FROM emotion_records GROUP BY emotion_type, emotion_label ORDER BY count DESC"
        )
        return [dict(r) for r in rows]

    def get_reflection_stats(self) -> list:
        """获取自反思质量分布"""
        rows = self._execute_query(
            "SELECT quality_label, COUNT(*) AS count, ROUND(AVG(score), 1) AS avg_score "
            "FROM self_reflections GROUP BY quality_label ORDER BY avg_score DESC"
        )
        return [dict(r) for r in rows]

    def get_confidence_stats(self) -> list:
        """获取置信度分布"""
        rows = self._execute_query(
            "SELECT confidence_label, COUNT(*) AS count "
            "FROM confidence_assessments GROUP BY confidence_label ORDER BY count DESC"
        )
        return [dict(r) for r in rows]

    def get_recent_activity(self, limit: int = 20) -> list:
        """获取最近活动记录"""
        rows = self._execute_query(
            "SELECT role, LEFT(content, 80) AS content_preview, created_at "
            "FROM messages ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        return [dict(r) for r in rows]

    def get_conversation_stats(self) -> list:
        """获取对话统计"""
        rows = self._execute_query(
            "SELECT agent_type, COUNT(*) AS count, SUM(message_count) AS messages "
            "FROM conversations GROUP BY agent_type"
        )
        return [dict(r) for r in rows]

    def get_completeness_stats(self) -> list:
        """获取信息完整性统计"""
        rows = self._execute_query(
            "SELECT agent_type, ROUND(AVG(completeness_pct), 1) AS avg_completeness, "
            "SUM(CASE WHEN should_ask=1 THEN 1 ELSE 0 END) AS ask_count, "
            "COUNT(*) AS total "
            "FROM completeness_records GROUP BY agent_type"
        )
        return [dict(r) for r in rows]

    def get_intent_route_stats(self) -> list:
        """获取意图路由统计"""
        rows = self._execute_query(
            "SELECT detected_intent, routed_agent, COUNT(*) AS count "
            "FROM intent_routes GROUP BY detected_intent, routed_agent ORDER BY count DESC"
        )
        return [dict(r) for r in rows]

    def get_document_stats(self) -> list:
        """获取文档生成统计"""
        rows = self._execute_query(
            "SELECT doc_type, COUNT(*) AS count, SUM(file_size) AS total_size "
            "FROM documents GROUP BY doc_type ORDER BY count DESC"
        )
        return [dict(r) for r in rows]

    def get_handoff_stats(self) -> list:
        """获取Agent交接统计"""
        rows = self._execute_query(
            "SELECT from_agent, to_agent, COUNT(*) AS count "
            "FROM agent_handoffs GROUP BY from_agent, to_agent ORDER BY count DESC"
        )
        return [dict(r) for r in rows]

    def get_progress_stats(self) -> list:
        """获取维权进度统计"""
        rows = self._execute_query(
            "SELECT current_label, COUNT(*) AS count, ROUND(AVG(progress_pct), 1) AS avg_pct "
            "FROM case_progress GROUP BY current_label ORDER BY count DESC"
        )
        return [dict(r) for r in rows]

    def get_compensation_stats(self) -> list:
        """获取赔偿预估统计"""
        rows = self._execute_query(
            "SELECT dispute_type, COUNT(*) AS count, "
            "ROUND(AVG(estimated_amount), 2) AS avg_amount, "
            "ROUND(MAX(estimated_amount), 2) AS max_amount "
            "FROM compensation_estimates GROUP BY dispute_type ORDER BY count DESC"
        )
        return [dict(r) for r in rows]

    def get_deadline_stats(self) -> list:
        """获取时效提醒统计"""
        rows = self._execute_query(
            "SELECT deadline_type, COUNT(*) AS count, "
            "SUM(CASE WHEN urgency_level='紧急' THEN 1 ELSE 0 END) AS urgent_count, "
            "SUM(CASE WHEN urgency_level='已过期' THEN 1 ELSE 0 END) AS expired_count "
            "FROM deadline_reminders GROUP BY deadline_type"
        )
        return [dict(r) for r in rows]

    def get_user_profile_stats(self) -> list:
        """获取用户画像统计"""
        rows = self._execute_query(
            "SELECT legal_level, urgency, user_type, COUNT(*) AS count "
            "FROM user_profiles GROUP BY legal_level, urgency, user_type ORDER BY count DESC"
        )
        return [dict(r) for r in rows]

    def get_email_stats(self) -> list:
        """获取邮件发送统计"""
        rows = self._execute_query(
            "SELECT status, COUNT(*) AS count, "
            "SUM(CASE WHEN status='sent' THEN 1 ELSE 0 END) AS sent_count, "
            "SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_count "
            "FROM email_logs GROUP BY status ORDER BY count DESC"
        )
        return [dict(r) for r in rows]

    def get_db_size(self) -> int:
        """获取数据库大小（字节）"""
        try:
            rows = self._execute_query(
                "SELECT SUM(data_length + index_length) AS size_bytes "
                "FROM information_schema.tables WHERE table_schema = %s",
                (MYSQL_DATABASE,),
            )
            return int(rows[0]["size_bytes"]) if rows and rows[0]["size_bytes"] else 0
        except Exception:
            return 0


# 全局单例
db = DatabaseManager()
