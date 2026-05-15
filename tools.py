from langchain_community.vectorstores import Chroma, FAISS
from config import GRAPH_TEMPLATE
from utils import replace_token_in_string
import os
import requests
from typing import List, Optional
from langchain_core.tools import tool
from utils import get_embedding_model, get_neo4j_conn

vdb  = Chroma(
    persist_directory=os.path.join(os.path.dirname(__file__), './data/db'),
    embedding_function=get_embedding_model()
)

@tool
def retrival_func(query):
    """
    仅用于查询“寻医问药网”平台本身的信息，例如：
    - 寻医问药网的成立时间、发展历程、联系方式、功能服务等。

    绝对不能用于回答医学/疾病/药品等专业医疗问题。
    如果用户的问题与寻医问药网平台无关，就不要调用这个工具。

    输入：应为一句完整、明确的搜索查询语句。
    返回：最相关的若干文档内容，并附带来源信息。
    当这个方法返回未找到相关文档，你回答不知道即可。
    """
    documents = vdb.similarity_search_with_relevance_scores(query, k=5)

    if not documents:
        return "未找到相关文档。"

    # 过滤并整理结果
    relevant_docs = [doc for doc, score in documents if score > 0.5]

    if not relevant_docs:
        return "未找到相关文档。"

    # 返回格式化的字符串，而不是列表
    formatted_result = "检索到以下相关信息：\n\n"
    for i, doc in enumerate(relevant_docs, 1):
        # 清理文档内容中的换行符
        clean_content = doc.page_content.replace('\n', ' ').strip()
        formatted_result += f"{i}. {clean_content[:500]}...\n\n"

    return formatted_result


@tool
def graph_func(
    disease: Optional[List[str]] = None,
    symptom: Optional[List[str]] = None,
    drug: Optional[List[str]] = None
) -> str:
    """
    用于回答用户关于疾病、药物、症状的相关提问。
     参数:
     - disease: 疾病实体列表
     - symptom: 症状实体列表
     - drug: 药物实体列表
     当用户询问：
     - 疾病怎么治
     - 疾病的症状
     - 药物可以治疗什么疾病
     - 疾病治疗
     - 疾病的并发症
     时使用该工具。
     调用前请先从用户问题中提取相关实体。

    返回知识图谱中的医疗信息。
    当查询无结果时，请如实告知用户。
    """

    disease = disease or []
    symptom = symptom or []
    drug = drug or []

    graph_templates = []

    # 构造查询模板
    for _, template in GRAPH_TEMPLATE.items():

        slot = template["slots"][0]

        if slot == "disease":
            slot_values = disease

        elif slot == "symptom":
            slot_values = symptom

        elif slot == "drug":
            slot_values = drug

        else:
            slot_values = []

        for value in slot_values:

            graph_templates.append({
                "question": replace_token_in_string(
                    template["question"],
                    [[slot, value]]
                ),

                "cypher": replace_token_in_string(
                    template["cypher"],
                    [[slot, value]]
                ),

                "answer": replace_token_in_string(
                    template["answer"],
                    [[slot, value]]
                ),
            })

    if not graph_templates:
        return "未找到相关医疗信息。"

    # 执行 Neo4j 查询
    neo4j_conn = get_neo4j_conn()

    query_results = []

    for item in graph_templates:

        try:

            result = neo4j_conn.run(
                item["cypher"]
            ).data()

            if not result:
                continue

            first_record = result[0]

            if not any(first_record.values()):
                continue

            answer = replace_token_in_string(
                item["answer"],
                list(first_record.items())
            )

            query_results.append(
                f"问题：{item['question']}\n答案：{answer}"
            )

        except Exception as e:
            print(f"Neo4j查询错误: {e}")

    # 返回结果
    if not query_results:
        return "知识图谱未找到匹配信息。"

    return "\n\n".join(query_results)


@tool
def search_func(query):
    """
    网络搜索引擎，用于获取最新的、公开的、非医疗专业知识的信息。

    适用场景：
    - 查询实时新闻、政策变动、药品价格、医院排名等动态信息。
    - 用户明确要求“在网上搜一下”或需要最新数据时。
    - 知识图谱和内部文档无法覆盖的泛生活问题（如天气、交通、蔡徐坤是谁等）。

    注意：
    - **不要**用来查询疾病症状、治疗方案、药物说明书等医学专业知识（应优先使用 graph_func）。
    - **不要**用来查询寻医问药网平台自身的信息（如成立时间、客服电话等，应使用 retrival_func）。
    - 如果其他工具已经返回了足够的信息，就不要重复搜索。
    """
    search_url = os.getenv('SEARCH_URL')
    search_api_key = os.getenv('SEARCH_API_KEY')

    payload = {
        "query": query,
        "summary": True,
        "count": 10
    }
    headers = {
        'Authorization': 'Bearer ' + search_api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(search_url, headers=headers, json=payload, timeout=10)
        search_result = response.json()

        # 提取搜索结果文本
        query_result = ""
        if search_result.get('code') == 200:
            data = search_result.get('data', {})
            if data:
                web_pages = data.get('webPages', {})
                if web_pages and 'value' in web_pages:
                    for item in web_pages['value']:
                        query_result += f"标题：{item.get('name', '')}\n"
                        query_result += f"摘要：{item.get('snippet', '')}\n"
                        query_result += f"链接：{item.get('url', '')}\n\n"

        if not query_result.strip():
            return "未找到相关信息"

        return query_result
    except Exception as e:
        print(f"搜索错误: {e}")
        return "搜索功能暂时不可用"

