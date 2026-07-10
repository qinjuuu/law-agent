"""
联网搜索工具模块
让智能体在本地知识库覆盖不了时自动联网搜索最新法律法规和案例
使用 Bing 搜索 API（通过 requests 直接抓取），无需 API Key
"""
import sys
import re
import json
import urllib.parse
import urllib.request
from langchain.tools import tool

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _bing_search(query: str, max_results: int = 5) -> list:
    """
    通过 Bing 搜索获取结果（无需 API Key）

    返回:
        [{title, body, href}, ...]
    """
    results = []
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.bing.com/search?q={encoded_query}&count={max_results}&setlang=zh-CN"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # 按 b_algo 分块
        algo_blocks = re.split(r'<li class="b_algo"', html)

        for block in algo_blocks[1:]:  # 跳过第一段（b_algo 之前的内容）
            # 找第一个有效的 https 链接和文本
            link_match = re.search(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
            # 找摘要段落
            cap_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)

            if link_match:
                href = link_match.group(1)
                # 清理标题文本，去掉域名前缀和 HTML 标签
                title_raw = re.sub(r"<[^>]+>", "", link_match.group(2)).strip()
                # Bing 有时把域名放在标题前面，如 "baidu.comhttps://baike.baidu.com..."
                # 去掉域名前缀
                title = re.sub(r"^[a-zA-Z0-9._-]+\.(com|cn|org|net|gov|edu)[^\s]*", "", title_raw).strip()
                if not title:
                    title = title_raw  # 如果清理后为空，用原始值

                body = ""
                if cap_match:
                    body = re.sub(r"<[^>]+>", "", cap_match.group(1)).strip()
                    # 清理 HTML 实体
                    body = body.replace("&ensp;", " ").replace("&#0183;", "·").replace("&amp;", "&")

                if title and len(title) > 3:
                    results.append({"title": title, "body": body, "href": href})

            if len(results) >= max_results:
                break

    except Exception as e:
        print(f"[WebSearch] Bing搜索失败: {e}")

    return results


def _ddgs_search(query: str, max_results: int = 5) -> list:
    """
    通过 DuckDuckGo 搜索获取结果（备用方案）
    """
    results = []
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                })
    except Exception as e:
        print(f"[WebSearch] DuckDuckGo搜索失败: {e}")

    return results


def _do_search(query: str, max_results: int = 5) -> list:
    """
    执行搜索，优先使用 Bing，失败时回退到 DuckDuckGo
    """
    # 先尝试 Bing
    results = _bing_search(query, max_results)
    if results:
        return results

    # 回退到 DuckDuckGo
    results = _ddgs_search(query, max_results)
    return results


@tool
def search_web(query: str) -> str:
    """
    联网搜索最新的法律法规、消费维权案例和商家信息

    当本地知识库检索不到相关内容、用户询问最新法规、或需要实时信息时调用此工具

    参数:
        query: 搜索关键词，如"2024年消费者权益保护法实施条例"、"某某商家投诉"

    返回:
        搜索结果摘要，包含标题、摘要和来源链接
    """
    results_list = _do_search(query, max_results=5)

    if not results_list:
        return f"联网搜索「{query}」未找到相关结果，建议换个关键词试试"

    lines = [f"联网搜索「{query}」结果如下:\n"]
    for i, r in enumerate(results_list, 1):
        lines.append(f"--- 搜索结果{i} ---")
        lines.append(f"标题: {r.get('title', '')}")
        lines.append(f"摘要: {r.get('body', '')}")
        lines.append(f"来源: {r.get('href', '')}\n")

    lines.append("提示: 以上信息来自互联网，请注意甄别准确性。法律法规以官方发布为准。")
    return "\n".join(lines)


@tool
def search_latest_regulation(topic: str) -> str:
    """
    搜索最新的法律法规和政策动态

    专门用于查询最新的消费维权相关法规，如新出台的实施条例、部门规章等

    参数:
        topic: 法规主题，如"预付卡管理办法"、"直播带货监管规定"、"消费者权益保护法实施条例"

    返回:
        最新法规搜索结果，包含法规名称、发布时间和核心内容摘要
    """
    search_query = f"{topic} 最新法规 2024 2025"
    results_list = _do_search(search_query, max_results=5)

    if not results_list:
        return f"未搜索到「{topic}」相关最新法规，建议访问中国政府网或司法部官网查询"

    lines = [f"「{topic}」最新法规搜索结果:\n"]
    for i, r in enumerate(results_list, 1):
        lines.append(f"--- 法规资讯{i} ---")
        lines.append(f"标题: {r.get('title', '')}")
        lines.append(f"摘要: {r.get('body', '')}")
        lines.append(f"来源: {r.get('href', '')}\n")

    lines.append("注意: 法规条文以全国人大官网或国务院官网发布版本为准。")
    lines.append("如需引用具体条文，建议到官方法源核实。")
    return "\n".join(lines)


@tool
def search_merchant_info(merchant_name: str) -> str:
    """
    联网搜索商家的实时信誉信息、投诉情况和新闻动态

    当本地商家信誉数据库中没有收录该商家时调用此工具

    参数:
        merchant_name: 商家名称，如"某某超市"、"某某电商平台"

    返回:
        商家实时信息，包含投诉新闻、监管处罚、用户评价等
    """
    queries = [
        f"{merchant_name} 投诉 举报",
        f"{merchant_name} 处罚 市场监管",
    ]

    all_results = []
    for q in queries:
        results = _do_search(q, max_results=3)
        for r in results:
            r["query"] = q
            all_results.append(r)

    if not all_results:
        return f"联网搜索「{merchant_name}」未找到相关投诉或处罚信息，该商家可能信誉良好或信息较少"

    # 去重
    seen_titles = set()
    unique_results = []
    for r in all_results:
        if r.get("title", "") not in seen_titles:
            seen_titles.add(r.get("title", ""))
            unique_results.append(r)

    lines = [f"「{merchant_name}」联网搜索结果:\n"]
    for i, r in enumerate(unique_results[:6], 1):
        lines.append(f"--- 资讯{i} ---")
        lines.append(f"标题: {r.get('title', '')}")
        lines.append(f"摘要: {r.get('body', '')}")
        lines.append(f"来源: {r.get('href', '')}")
        lines.append(f"搜索角度: {r.get('query', '')}\n")

    lines.append("提示: 以上信息来自互联网公开报道，仅供参考。")
    lines.append("建议同时到全国12315平台查看该商家的官方投诉统计数据。")
    return "\n".join(lines)
