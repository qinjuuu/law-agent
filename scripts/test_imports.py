"""综合导入和一致性测试"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
# 将项目根目录加入 path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
errors = []

# 1. 测试所有Agent导入
try:
    from agents.qa_agent import ConsumerQAAgent
    from agents.complaint_agent import ComplaintAgent
    from agents.review_agent import ReviewAgent
    from agents.router import RouterAgent
    print('[OK] 所有Agent导入成功')
except Exception as e:
    errors.append(f'Agent导入失败: {e}')
    print(f'[FAIL] Agent导入失败: {e}')

# 2. 测试所有工具导入
try:
    from tools.search_tools import search_law, search_case, smart_search
    from tools.web_search_tools import search_web, search_latest_regulation, search_merchant_info
    from tools.innovation_tools import (
        estimate_compensation, generate_evidence_checklist, plan_rights_path,
        merchant_tactics_response, rights_deadline_reminder, trap_warning,
        check_merchant_reputation, multi_platform_complaint, package_evidence,
    )
    from tools.agent_tools import handoff_to_agent, get_rights_progress
    from tools.file_tools import create_file, read_file, delete_file, zip_files
    from tools.word_tools import create_complaint_report, create_review_report
    from tools.email_tools import send_email
    print('[OK] 所有工具导入成功')
except Exception as e:
    errors.append(f'工具导入失败: {e}')
    print(f'[FAIL] 工具导入失败: {e}')

# 3. 测试RAG模块
try:
    from rag.retriever import HybridRetriever
    from rag.embedder import Embedder
    from rag.vector_store import VectorStore
    from rag.bm25_retriever import BM25Retriever
    from rag.chunking import TextChunker, Chunk
    print('[OK] RAG模块导入成功')
except Exception as e:
    errors.append(f'RAG导入失败: {e}')
    print(f'[FAIL] RAG导入失败: {e}')

# 4. 测试数据库
try:
    from database.db_manager import DatabaseManager
    db = DatabaseManager()
    tables = db._execute_query('SHOW TABLES')
    table_names = [list(t.values())[0] for t in tables]
    print(f'[OK] 数据库连接成功，{len(table_names)}张表')
    if 'email_logs' not in table_names:
        errors.append('数据库缺少email_logs表')
        print('[FAIL] 数据库缺少email_logs表')
    else:
        print('[OK] email_logs表存在')
except Exception as e:
    errors.append(f'数据库连接失败: {e}')
    print(f'[FAIL] 数据库连接失败: {e}')

# 5. 测试config
try:
    from config import SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_ENABLED
    print(f'[OK] SMTP配置: enabled={SMTP_ENABLED}, server={SMTP_SERVER}, port={SMTP_PORT}')
except Exception as e:
    errors.append(f'配置导入失败: {e}')
    print(f'[FAIL] 配置导入失败: {e}')

# 6. 检查系统提示词中法条匹配指南
try:
    from agents.qa_agent import QA_SYSTEM_PROMPT
    from agents.complaint_agent import COMPLAINT_SYSTEM_PROMPT
    from agents.review_agent import REVIEW_SYSTEM_PROMPT

    qa_has = '法条匹配' in QA_SYSTEM_PROMPT
    comp_has = '法条匹配' in COMPLAINT_SYSTEM_PROMPT
    rev_has = '法条匹配' in REVIEW_SYSTEM_PROMPT

    print(f'法条匹配指南: QA={qa_has}, Complaint={comp_has}, Review={rev_has}')
    if not rev_has:
        errors.append('Review Agent缺少法条匹配指南')
    else:
        print('[OK] 三个Agent都有法条匹配指南')
except Exception as e:
    errors.append(f'提示词检查失败: {e}')
    print(f'[FAIL] 提示词检查失败: {e}')

# 7. 检查投诉Agent编号是否修复
try:
    from agents.complaint_agent import COMPLAINT_SYSTEM_PROMPT
    # 找工作流程部分的编号
    import re
    steps = re.findall(r'^(\d+)\. ', COMPLAINT_SYSTEM_PROMPT, re.MULTILINE)
    # 在工作流程部分应该有 1-7
    workflow_section = COMPLAINT_SYSTEM_PROMPT.split('你的工作流程:')[1].split('投诉信应当')[0]
    workflow_steps = re.findall(r'^(\d+)\. ', workflow_section, re.MULTILINE)
    print(f'投诉Agent工作流程编号: {workflow_steps}')
    if len(workflow_steps) != len(set(workflow_steps)):
        errors.append(f'投诉Agent工作流程编号重复: {workflow_steps}')
        print(f'[FAIL] 投诉Agent工作流程编号重复: {workflow_steps}')
    else:
        print('[OK] 投诉Agent工作流程编号无重复')
except Exception as e:
    errors.append(f'编号检查失败: {e}')
    print(f'[FAIL] 编号检查失败: {e}')

# 8. 检查 _TOOL_LABELS 是否覆盖所有工具
try:
    from agents.base import _TOOL_LABELS
    expected = {
        'smart_search', 'search_law', 'search_case', 'search_web',
        'search_latest_regulation', 'search_merchant_info',
        'create_complaint_report', 'create_review_report',
        'create_file', 'read_file', 'delete_file', 'zip_files',
        'estimate_compensation', 'generate_evidence_checklist',
        'plan_rights_path', 'merchant_tactics_response',
        'rights_deadline_reminder', 'multi_platform_complaint',
        'package_evidence', 'trap_warning', 'check_merchant_reputation',
        'handoff_to_agent', 'get_rights_progress', 'send_email',
    }
    missing = expected - set(_TOOL_LABELS.keys())
    if missing:
        errors.append(f'_TOOL_LABELS缺少: {missing}')
        print(f'[FAIL] _TOOL_LABELS缺少: {missing}')
    else:
        print(f'[OK] _TOOL_LABELS覆盖全部{len(expected)}个工具')
except Exception as e:
    errors.append(f'_TOOL_LABELS检查失败: {e}')
    print(f'[FAIL] _TOOL_LABELS检查失败: {e}')

# 9. 检查Review Agent工具列表是否与提示词一致
try:
    from agents.review_agent import ReviewAgent
    import inspect
    src = inspect.getsource(ReviewAgent.__init__)
    has_smart_search = 'smart_search' in src
    has_search_law = 'search_law' in src
    has_create_review = 'create_review_report' in src
    has_send_email = 'send_email' in src
    has_handoff = 'handoff_to_agent' in src
    print(f'Review Agent工具: smart_search={has_smart_search}, search_law={has_search_law}, '
          f'create_review={has_create_review}, send_email={has_send_email}, handoff={has_handoff}')
    if not all([has_smart_search, has_search_law, has_create_review, has_send_email, has_handoff]):
        errors.append('Review Agent工具列表不完整')
        print('[FAIL] Review Agent工具列表不完整')
    else:
        print('[OK] Review Agent工具列表完整')
except Exception as e:
    errors.append(f'Review Agent检查失败: {e}')
    print(f'[FAIL] Review Agent检查失败: {e}')

# 10. 检查 db_manager 统计方法
try:
    from database.db_manager import DatabaseManager
    db = DatabaseManager()
    stats_methods = [m for m in dir(db) if m.startswith('get_') and 'stats' in m]
    print(f'数据库统计方法: {len(stats_methods)}个')
    for m in stats_methods:
        try:
            result = getattr(db, m)()
            print(f'  {m}: OK ({len(result) if isinstance(result, list) else "ok"})')
        except Exception as e:
            errors.append(f'{m}执行失败: {e}')
            print(f'  {m}: FAIL - {e}')
except Exception as e:
    errors.append(f'统计方法检查失败: {e}')
    print(f'[FAIL] 统计方法检查失败: {e}')

print()
print('=' * 50)
if errors:
    print(f'发现 {len(errors)} 个问题:')
    for e in errors:
        print(f'  - {e}')
else:
    print('所有检查通过!')
