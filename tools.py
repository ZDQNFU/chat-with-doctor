"""
医疗问诊机器人 —— 工具集

对外暴露三个 @tool 供 LangGraph Agent 调用：
  - retrival_func   → 寻医问药网内部文档检索 (Chroma)
  - graph_func      → 医疗知识图谱查询 (Neo4j)
  - search_func     → 互联网搜索 (博世 API)
"""

from __future__ import annotations

import os
from typing import Optional

import requests
from langchain_chroma import Chroma
from langchain_core.tools import tool

from config import GRAPH_TEMPLATE
from logger_config import get_logger
from utils import get_embedding_model, get_neo4j_conn, replace_token_in_string

logger = get_logger(__name__)

# ── 可配置常量 ─────────────────────────────────────────
_RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
_RETRIEVAL_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))
_SEARCH_TIMEOUT: float = float(os.getenv("SEARCH_TIMEOUT", "10"))
_SEARCH_COUNT: int = int(os.getenv("SEARCH_COUNT", "10"))


# ═══════════════════════════════════════════════════════
#  懒加载 单例工厂（避免 import 时初始化重型资源）
# ═══════════════════════════════════════════════════════

_vdb: Optional[Chroma] = None
_vdb_init_error: Optional[str] = None


def _get_vdb() -> Optional[Chroma]:
    """获取 Chroma 向量库（懒加载，失败时返回 None 而非崩溃）"""
    global _vdb, _vdb_init_error

    if _vdb is not None:
        return _vdb
    if _vdb_init_error is not None:
        logger.warning("Chroma 已初始化失败，跳过: %s", _vdb_init_error)
        return None

    try:
        persist_dir = os.path.join(os.path.dirname(__file__), "data", "db")
        embedding = get_embedding_model()
        if embedding is None:
            raise RuntimeError("get_embedding_model() 返回 None")

        _vdb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
        )
        logger.info("Chroma 向量库初始化成功 (persist_dir=%s)", persist_dir)
    except Exception as exc:
        _vdb_init_error = str(exc)
        logger.error("Chroma 向量库初始化失败: %s", exc, exc_info=True)

    return _vdb


# ═══════════════════════════════════════════════════════
#  Tool 1 — 寻医问药网平台文档检索
# ═══════════════════════════════════════════════════════

@tool
def retrival_func(query: str) -> str:
    """仅用于查询"寻医问药网"平台本身的信息，例如：
    - 寻医问药网的成立时间、发展历程、联系方式、功能服务等。

    绝对不能用于回答医学/疾病/药品等专业医疗问题。
    如果用户的问题与寻医问药网平台无关，就不要调用这个工具。

    参数 query: 一句完整、明确的搜索查询语句。
    返回: 最相关的若干文档内容。未找到时告知用户即可。
    """
    # ── 输入校验 ──
    if not query or not query.strip():
        logger.warning("retrival_func 收到空查询")
        return "查询内容为空，请提供有效的搜索词。"

    # ── 懒加载向量库 ──
    vector_db = _get_vdb()
    if vector_db is None:
        return "文档检索服务暂不可用，请联系管理员。"

    # ── 检索 ──
    try:
        logger.info("[retrival_func] 检索 | query=%s", query[:80])
        documents = vector_db.similarity_search_with_relevance_scores(
            query.strip(), k=_RETRIEVAL_K
        )
    except Exception as exc:
        logger.error("Chroma 检索失败: %s", exc, exc_info=True)
        return "文档检索过程中发生错误，请稍后重试。"

    if not documents:
        return "未找到相关文档。"

    relevant = [doc for doc, score in documents if score > _RETRIEVAL_THRESHOLD]
    if not relevant:
        return "未找到相关文档。"

    # ── 格式化结果 ──
    parts = ["检索到以下相关信息：\n"]
    for i, doc in enumerate(relevant, 1):
        clean = doc.page_content.replace("\n", " ").strip()
        parts.append(f"{i}. {clean[:500]}...\n")
    return "".join(parts)


# ═══════════════════════════════════════════════════════
#  Tool 2 — 医疗知识图谱查询
# ═══════════════════════════════════════════════════════

@tool
def graph_func(
    disease: Optional[list[str]] = None,
    symptom: Optional[list[str]] = None,
    drug: Optional[list[str]] = None,
) -> str:
    """用于回答用户关于疾病、药物、症状的相关提问。

    参数:
      disease  疾病实体列表
      symptom  症状实体列表
      drug     药物实体列表

    适用场景（调用本工具前先从用户问题中提取实体）：
      - 疾病怎么治 / 疾病的症状 / 药物能治什么病
      - 疾病并发症 / 预防 / 检查 / 饮食禁忌 / 就诊科室

    查询无结果时请如实告知用户。
    """
    disease = disease or []
    symptom = symptom or []
    drug = drug or []

    # ── 实体→模板槽位 映射 ──
    slot_map: dict[str, list[str]] = {
        "disease": disease,
        "symptom": symptom,
        "drug": drug,
    }

    # ── 构建查询模板 ──
    graph_templates: list[dict[str, str]] = []
    for template in GRAPH_TEMPLATE.values():
        slot = template["slots"][0]
        for value in slot_map.get(slot, []):
            graph_templates.append(
                {
                    "question": replace_token_in_string(template["question"], [[slot, value]]),
                    "cypher": replace_token_in_string(template["cypher"], [[slot, value]]),
                    "answer": replace_token_in_string(template["answer"], [[slot, value]]),
                }
            )

    if not graph_templates:
        return "未找到相关医疗信息。"

    # ── 执行 Neo4j 查询 ──
    try:
        neo4j_conn = get_neo4j_conn()
    except Exception as exc:
        logger.error("Neo4j 连接失败: %s", exc, exc_info=True)
        return "知识图谱服务暂不可用，请稍后重试。"

    results: list[str] = []
    for item in graph_templates:
        try:
            rows = neo4j_conn.run(item["cypher"]).data()
            if not rows:
                continue

            first = rows[0]
            if not any(first.values()):
                continue

            answer = replace_token_in_string(item["answer"], list(first.items()))
            results.append(f"问题：{item['question']}\n答案：{answer}")

        except Exception as exc:
            logger.error("Neo4j 查询异常 (cypher=%s): %s", item.get("cypher"), exc, exc_info=True)
            # 单条失败不中断其余查询

    if not results:
        return "知识图谱未找到匹配信息。"

    return "\n\n".join(results)


# ═══════════════════════════════════════════════════════
#  Tool 3 — 互联网搜索
# ═══════════════════════════════════════════════════════

@tool
def search_func(query: str) -> str:
    """网络搜索引擎，用于获取最新的、公开的、非医疗专业知识的信息。

    适用场景：
      - 实时新闻、政策变动、药品价格、医院排名等动态信息。
      - 用户明确要求"在网上搜一下"或需要最新数据时。

    注意：
      - **不要**查询疾病症状/治疗方案/药物说明书（优先使用 graph_func）。
      - **不要**查询寻医问药网平台自身信息（优先使用 retrival_func）。
      - 其他工具已返回足够信息时不要重复搜索。
    """
    # ── 前置检查 ──
    search_url = os.getenv("SEARCH_URL")
    search_api_key = os.getenv("SEARCH_API_KEY")

    if not search_url:
        logger.error("SEARCH_URL 未配置")
        return "搜索功能未配置，请联系管理员。"
    if not search_api_key:
        logger.error("SEARCH_API_KEY 未配置")
        return "搜索功能未配置，请联系管理员。"
    if not query or not query.strip():
        logger.warning("search_func 收到空查询")
        return "搜索内容为空，请提供有效的关键词。"

    # ── 发起请求 ──
    try:
        response = requests.post(
            search_url,
            headers={
                "Authorization": f"Bearer {search_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query.strip(),
                "summary": True,
                "count": _SEARCH_COUNT,
            },
            timeout=_SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
    except requests.Timeout:
        logger.error("搜索请求超时 (url=%s)", search_url)
        return "搜索请求超时，请稍后重试。"
    except requests.HTTPError as exc:
        logger.error("搜索 HTTP 错误 (status=%s): %s", exc.response.status_code if exc.response else "?", exc)
        return "搜索服务暂时不可用，请稍后重试。"
    except requests.RequestException as exc:
        logger.error("搜索网络错误: %s", exc)
        return "搜索服务暂时不可用，请稍后重试。"
    except Exception as exc:
        logger.error("搜索解析错误: %s", exc, exc_info=True)
        return "搜索服务暂时不可用，请稍后重试。"

    # ── 解析结果 ──
    if body.get("code") != 200:
        logger.warning("搜索 API 返回非 200 code: %s", body.get("code"))
        return "未找到相关信息。"

    web_pages = body.get("data", {}).get("webPages")
    if not web_pages or not web_pages.get("value"):
        return "未找到相关信息。"

    parts: list[str] = []
    for item in web_pages["value"]:
        name = item.get("name", "")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        if not any([name, snippet, url]):
            continue
        parts.append(f"标题：{name}\n摘要：{snippet}\n链接：{url}\n")

    if not parts:
        return "未找到相关信息。"

    return "".join(parts)
