-- ================================================================
-- 消费维权智能助手 - 完整数据库 Schema
-- 35 张表，覆盖系统运行的全维度数据
-- MySQL 8.0+
-- ================================================================

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ================================================================
-- 一、系统元数据 (1张表)
-- ================================================================

-- 1. 数据库元信息
CREATE TABLE IF NOT EXISTS db_meta (
    `key`         VARCHAR(100) PRIMARY KEY,
    `value`       TEXT,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ================================================================
-- 二、用户与对话 (3张表)
-- ================================================================

-- 2. 用户表（基于会话）
CREATE TABLE IF NOT EXISTS users (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    session_id      VARCHAR(255) UNIQUE NOT NULL,
    first_seen      DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active     DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_messages  INT DEFAULT 0,
    legal_level     VARCHAR(20) DEFAULT 'novice',
    urgency         VARCHAR(20) DEFAULT 'normal',
    user_type       VARCHAR(20) DEFAULT 'consumer'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. 对话会话表
CREATE TABLE IF NOT EXISTS conversations (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT,
    session_id      VARCHAR(255) NOT NULL,
    agent_type      VARCHAR(50) NOT NULL,
    title           VARCHAR(500) DEFAULT '',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    message_count   INT DEFAULT 0,
    status          VARCHAR(20) DEFAULT 'active',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_conv_session ON conversations(session_id);
CREATE INDEX idx_conv_agent ON conversations(agent_type);

-- 4. 消息记录表
CREATE TABLE IF NOT EXISTS messages (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    role            VARCHAR(20) NOT NULL,
    content         LONGTEXT,
    content_length  INT DEFAULT 0,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INT,
    agent_type      VARCHAR(50),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_msg_conv ON messages(conversation_id);
CREATE INDEX idx_msg_role ON messages(role);

-- ================================================================
-- 三、Agent 处理流程 (7张表)
-- ================================================================

-- 5. 意图路由记录
CREATE TABLE IF NOT EXISTS intent_routes (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    user_message    TEXT,
    detected_intent VARCHAR(50) NOT NULL,
    routed_agent    VARCHAR(50) NOT NULL,
    confidence      DOUBLE,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. 用户画像检测记录
CREATE TABLE IF NOT EXISTS user_profiles (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    legal_level     VARCHAR(20),
    urgency         VARCHAR(20),
    user_type       VARCHAR(20),
    style_hint      TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7. 情绪检测记录
CREATE TABLE IF NOT EXISTS emotion_records (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    emotion_type    VARCHAR(30) NOT NULL,
    emotion_label   VARCHAR(50),
    keywords_matched TEXT,
    soothing_text   TEXT,
    action_hint     TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_emotion_type ON emotion_records(emotion_type);

-- 8. 信息完整性追踪记录
CREATE TABLE IF NOT EXISTS completeness_records (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    agent_type      VARCHAR(50),
    completeness_pct INT,
    provided_fields TEXT,
    missing_fields  TEXT,
    should_ask      TINYINT DEFAULT 0,
    ask_hint        TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 9. 工具调用记录
CREATE TABLE IF NOT EXISTS tool_calls (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    tool_name       VARCHAR(100) NOT NULL,
    tool_label      VARCHAR(200),
    tool_args       TEXT,
    tool_result     TEXT,
    result_length   INT DEFAULT 0,
    duration_ms     INT,
    success         TINYINT DEFAULT 1,
    error_message   TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_tool_name ON tool_calls(tool_name);
CREATE INDEX idx_tool_conv ON tool_calls(conversation_id);

-- 10. 思维链记录
CREATE TABLE IF NOT EXISTS reasoning_chains (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    chain_text      LONGTEXT,
    step_count      INT DEFAULT 0,
    tools_used      TEXT,
    emotion_detected VARCHAR(30),
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 11. Agent 自反思记录
CREATE TABLE IF NOT EXISTS self_reflections (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    score           INT,
    passed          TINYINT DEFAULT 1,
    quality_label   VARCHAR(20),
    issues          TEXT,
    suggestion      TEXT,
    legal_accuracy  INT,
    completeness    INT,
    tone            INT,
    actionability   INT,
    risk_notice     INT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 12. 置信度评估记录
CREATE TABLE IF NOT EXISTS confidence_assessments (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    confidence_level VARCHAR(20),
    confidence_label VARCHAR(20),
    confidence_emoji VARCHAR(10),
    reason          TEXT,
    is_out_of_scope TINYINT DEFAULT 0,
    boundary_warning TEXT,
    has_law_reference TINYINT DEFAULT 0,
    law_signal_count INT DEFAULT 0,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 13. 维权进度记录
CREATE TABLE IF NOT EXISTS case_progress (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    current_stage   INT,
    current_label   VARCHAR(100),
    stages_completed TEXT,
    next_action     TEXT,
    progress_bar    VARCHAR(200),
    progress_pct    INT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ================================================================
-- 四、多 Agent 协作 (1张表)
-- ================================================================

-- 14. Agent 交接记录
CREATE TABLE IF NOT EXISTS agent_handoffs (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    from_agent      VARCHAR(50),
    to_agent        VARCHAR(50) NOT NULL,
    reason          TEXT,
    context_summary TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ================================================================
-- 五、业务记录 (11张表)
-- ================================================================

-- 15. 投诉信记录
CREATE TABLE IF NOT EXISTS complaints (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    complainant_name VARCHAR(100),
    contact         VARCHAR(200),
    merchant_name   VARCHAR(200),
    merchant_address VARCHAR(500),
    purchase_time   VARCHAR(100),
    product_name    VARCHAR(200),
    purchase_amount DOUBLE,
    dispute_detail  LONGTEXT,
    demand          TEXT,
    legal_basis     TEXT,
    file_path       VARCHAR(500),
    file_size       INT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_comp_merchant ON complaints(merchant_name);

-- 16. 条款审查记录
CREATE TABLE IF NOT EXISTS clause_reviews (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    contract_title  VARCHAR(300),
    clause_content  LONGTEXT,
    risk_level      VARCHAR(20),
    risk_score      INT,
    risk_items      TEXT,
    review_results  LONGTEXT,
    suggestions     LONGTEXT,
    legal_basis     TEXT,
    file_path       VARCHAR(500),
    file_size       INT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_review_risk ON clause_reviews(risk_level);

-- 17. 赔偿预估记录
CREATE TABLE IF NOT EXISTS compensation_estimates (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    dispute_type    VARCHAR(50) NOT NULL,
    purchase_amount DOUBLE,
    actual_loss     DOUBLE DEFAULT 0,
    estimated_amount DOUBLE,
    legal_basis     VARCHAR(500),
    formula         VARCHAR(500),
    calculation_detail TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_comp_type ON compensation_estimates(dispute_type);

-- 18. 证据清单记录
CREATE TABLE IF NOT EXISTS evidence_checklists (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    dispute_type    VARCHAR(50) NOT NULL,
    checklist_items TEXT,
    item_count      INT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 19. 维权路径规划记录
CREATE TABLE IF NOT EXISTS rights_paths (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    path_type       VARCHAR(50),
    dispute_description TEXT,
    steps           TEXT,
    step_count      INT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 20. 商家话术应对记录
CREATE TABLE IF NOT EXISTS merchant_tactics (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    merchant_statement TEXT,
    legal_refutation TEXT,
    matched_tactics  TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 21. 维权时效提醒记录
CREATE TABLE IF NOT EXISTS deadline_reminders (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    deadline_type   VARCHAR(100) NOT NULL,
    purchase_date   VARCHAR(50),
    deadline_date   VARCHAR(50),
    remaining_days  INT,
    urgency_level   VARCHAR(20),
    legal_basis     VARCHAR(500),
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 22. 商家信誉查询缓存
CREATE TABLE IF NOT EXISTS merchant_reputations (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    merchant_name   VARCHAR(200) NOT NULL,
    complaints      INT DEFAULT 0,
    resolved        INT DEFAULT 0,
    resolve_rate    DOUBLE,
    rating          DOUBLE,
    common_issues   TEXT,
    risk_level      VARCHAR(20),
    queried_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    query_count     INT DEFAULT 1
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_rep_name ON merchant_reputations(merchant_name);

-- 23. 消费陷阱查询记录
CREATE TABLE IF NOT EXISTS trap_warnings (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    industry        VARCHAR(50) NOT NULL,
    trap_data       TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 24. 文档生成记录
CREATE TABLE IF NOT EXISTS documents (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    doc_type        VARCHAR(50) NOT NULL,
    filename        VARCHAR(300),
    filepath        VARCHAR(500),
    file_size       INT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_doc_type ON documents(doc_type);

-- 25. 对话摘要记录
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    summary_text    LONGTEXT,
    file_path       VARCHAR(500),
    message_count   INT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ================================================================
-- 六、知识库 (10张表)
-- ================================================================

-- 26. 法律条文库
CREATE TABLE IF NOT EXISTS laws (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    law_name        VARCHAR(100) NOT NULL,
    article_number  VARCHAR(50),
    article_title   VARCHAR(300),
    content         LONGTEXT,
    category        VARCHAR(100),
    tags            TEXT,
    source_file     VARCHAR(200),
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_law_name ON laws(law_name);
CREATE INDEX idx_law_article ON laws(article_number);

-- 27. 案例库
CREATE TABLE IF NOT EXISTS case_precedents (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    case_title      VARCHAR(500),
    case_type       VARCHAR(100),
    court           VARCHAR(200),
    ruling_date     VARCHAR(50),
    summary         LONGTEXT,
    legal_basis     TEXT,
    ruling_result   TEXT,
    tags            TEXT,
    source_file     VARCHAR(200),
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_case_type ON case_precedents(case_type);

-- 28. 赔偿规则库
CREATE TABLE IF NOT EXISTS compensation_rules (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    dispute_type    VARCHAR(50) UNIQUE NOT NULL,
    law             VARCHAR(500),
    multiplier      DOUBLE DEFAULT 0,
    loss_multiplier DOUBLE DEFAULT 0,
    minimum         DOUBLE DEFAULT 0,
    description     TEXT,
    formula         VARCHAR(500),
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 29. 证据模板库
CREATE TABLE IF NOT EXISTS evidence_templates (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    dispute_type    VARCHAR(50) NOT NULL,
    item_order      INT,
    item_description TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_ev_template_type ON evidence_templates(dispute_type);

-- 30. 维权路径模板库
CREATE TABLE IF NOT EXISTS rights_path_templates (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    path_type       VARCHAR(50) NOT NULL,
    step_number     INT,
    action          VARCHAR(200),
    method          TEXT,
    duration        VARCHAR(100),
    success_rate    VARCHAR(50),
    tip             TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_path_type ON rights_path_templates(path_type);

-- 31. 平台投诉格式模板库
CREATE TABLE IF NOT EXISTS platform_templates (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    platform_key    VARCHAR(50) UNIQUE NOT NULL,
    platform_name   VARCHAR(100),
    format_type     VARCHAR(50),
    max_length      VARCHAR(50),
    required_fields TEXT,
    tip             TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 32. 维权阶段定义
CREATE TABLE IF NOT EXISTS rights_stages (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    stage_number    INT UNIQUE NOT NULL,
    stage_key       VARCHAR(50) NOT NULL,
    stage_label     VARCHAR(100) NOT NULL,
    description     TEXT,
    stage_keywords  TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 33. 时效规则库
CREATE TABLE IF NOT EXISTS deadline_rules (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    deadline_type   VARCHAR(100) UNIQUE NOT NULL,
    days            INT NOT NULL,
    law             VARCHAR(500),
    note            TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 34. 商家话术知识库
CREATE TABLE IF NOT EXISTS merchant_tactics_kb (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    tactic_text     TEXT,
    tactic_keywords TEXT,
    legal_refutation TEXT,
    law_reference   VARCHAR(500),
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 35. 消费陷阱知识库
CREATE TABLE IF NOT EXISTS trap_kb (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    industry        VARCHAR(50) NOT NULL,
    trap_name       VARCHAR(200),
    trap_description LONGTEXT,
    warning_signs   TEXT,
    prevention_tips TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_trap_industry ON trap_kb(industry);

-- 36. 邮件发送日志
CREATE TABLE IF NOT EXISTS email_logs (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    message_id      INT,
    from_email      VARCHAR(200),
    to_email        VARCHAR(200) NOT NULL,
    subject         VARCHAR(500),
    body_preview    TEXT,
    attachments     TEXT,
    attachment_count INT DEFAULT 0,
    status          VARCHAR(20) DEFAULT 'pending',
    error_message   TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE INDEX idx_email_to ON email_logs(to_email);
CREATE INDEX idx_email_status ON email_logs(status);

SET FOREIGN_KEY_CHECKS = 1;

-- ================================================================
-- 视图：常用统计查询
-- ================================================================

-- 工具调用统计视图
CREATE OR REPLACE VIEW v_tool_stats AS
SELECT
    tool_name,
    tool_label,
    COUNT(*) AS call_count,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS success_count,
    ROUND(AVG(duration_ms), 0) AS avg_duration_ms,
    MAX(created_at) AS last_called
FROM tool_calls
GROUP BY tool_name, tool_label
ORDER BY call_count DESC;

-- 情绪分布视图
CREATE OR REPLACE VIEW v_emotion_stats AS
SELECT
    emotion_type,
    emotion_label,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM emotion_records), 1) AS percentage
FROM emotion_records
GROUP BY emotion_type, emotion_label
ORDER BY count DESC;

-- 对话统计视图
CREATE OR REPLACE VIEW v_conversation_stats AS
SELECT
    agent_type,
    COUNT(*) AS conversation_count,
    SUM(message_count) AS total_messages,
    MAX(updated_at) AS last_active
FROM conversations
GROUP BY agent_type;

-- 自反思质量分布视图
CREATE OR REPLACE VIEW v_reflection_stats AS
SELECT
    quality_label,
    COUNT(*) AS count,
    ROUND(AVG(score), 1) AS avg_score,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM self_reflections), 1) AS percentage
FROM self_reflections
GROUP BY quality_label
ORDER BY avg_score DESC;
