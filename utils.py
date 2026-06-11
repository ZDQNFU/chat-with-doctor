import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from py2neo import Graph
from langchain_community.embeddings import HuggingFaceEmbeddings, DashScopeEmbeddings

from logger_config import get_logger

#加载环境变量
load_dotenv()

logger = get_logger(__name__)


def get_embedding_model():
    """获取词嵌入模型
    支持两种模型：
    - qwen: 千问云端 embedding（DashScope 原生 API，需配置 QWEN_API_KEY / QWEN_EMBEDDING_MODEL）
    - bge:  本地 HuggingFace 模型（需配置 BGE_MODEL_PATH 或 BGE_EMBEDDING_MODEL）
    通过 .env 中的 EMBEDDING_MODEL 切换（默认 bge）
    """
    model_type = os.getenv('EMBEDDING_MODEL', 'bge')

    if model_type == 'qwen':
        model_name = os.getenv('QWEN_EMBEDDING_MODEL', 'text-embedding-v4')
        api_key = os.getenv('QWEN_API_KEY')
        logger.info("初始化 Qwen embedding 模型 (DashScope) | model=%s", model_name)
        if not api_key:
            logger.warning("QWEN_API_KEY 未设置，Qwen embedding 将回退到 BGE")
            # fall through to bge below
        else:
            return DashScopeEmbeddings(
                model=model_name,
                dashscope_api_key=api_key,
            )

    # ── 默认：bge 本地模型 ──
    model_path = os.getenv('BGE_MODEL_PATH')
    if model_path:
        model_name = model_path
    else:
        model_name = os.getenv('BGE_EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')

    logger.info("初始化 BGE embedding 模型 | model=%s device=%s",
                model_name, os.getenv('EMBEDDING_DEVICE', 'cpu'))
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            'device': os.getenv('EMBEDDING_DEVICE')
        },
        encode_kwargs={
            'normalize_embeddings': True
        }
    )


def get_llm_model():
    """获取大语言模型"""
    model_type = os.getenv('LLM_MODEL', 'deepseek')
    model_name = os.getenv('DEEPSEEK_LLM_MODEL', 'unknown')
    temperature = os.getenv('TEMPERATURE', '0')
    max_tokens = os.getenv('MAX_TOKENS', '1000')

    logger.info("初始化 LLM | type=%s model=%s temperature=%s max_tokens=%s",
                model_type, model_name, temperature, max_tokens)

    model_map = {
        'deepseek': ChatOpenAI(
            model=model_name,
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=os.getenv('DEEPSEEK_BASE_URL'),
            temperature=temperature,
            max_tokens=int(max_tokens),
            extra_body={"thinking": {"type": "disabled"}}
        )
    }
    return model_map.get(model_type)


def chat_with_agent(agent, user_input, thread_id="default-thread"):
    """简化的聊天接口"""
    if isinstance(user_input, str):
        messages = [HumanMessage(content=user_input)]
    else:
        messages = user_input

    config = {"configurable": {"thread_id": thread_id}}

    t_start = time.perf_counter()
    logger.info("[%s] Agent 开始处理 | query_len=%d", thread_id, len(str(user_input)))
    try:
        result = agent.invoke({"messages": messages}, config=config)
        elapsed = time.perf_counter() - t_start
        logger.info("[%s] Agent 处理完成 | latency=%.2fs", thread_id, elapsed)
        return result
    except Exception:
        elapsed = time.perf_counter() - t_start
        logger.exception("[%s] Agent 处理失败 | latency=%.2fs", thread_id, elapsed)
        raise


def stream_chat_with_agent(agent, user_input, thread_id="default-thread"):
    """流式聊天接口"""
    config = {"configurable": {"thread_id": thread_id}}

    t_start = time.perf_counter()
    token_count = 0
    logger.info("[%s] Agent 开始流式处理 | query_len=%d", thread_id, len(str(user_input)))
    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                message, _metadata = chunk
                # 只输出AI生成的内容，跳过工具消息等
                if isinstance(message, AIMessageChunk) and message.content:
                    if isinstance(message.content, str):
                        token_count += 1
                        yield message.content
                    elif isinstance(message.content, list):
                        for item in message.content:
                            if isinstance(item, dict) and 'text' in item:
                                token_count += 1
                                yield item['text']
                            elif isinstance(item, str):
                                token_count += 1
                                yield item
        elapsed = time.perf_counter() - t_start
        logger.info("[%s] Agent 流式处理完成 | latency=%.2fs chunks=%d",
                    thread_id, elapsed, token_count)
    except Exception:
        elapsed = time.perf_counter() - t_start
        logger.exception("[%s] Agent 流式处理失败 | latency=%.2fs chunks=%d",
                         thread_id, elapsed, token_count)
        raise


def extract_ai_response(result):
    """从结果中提取 AI 的回复文本"""
    messages = result.get("messages", [])
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
    return ai_messages[-1] if ai_messages else None

#填充CQL中词，用于查询知识图谱
def replace_token_in_string(string, slots):
    for key, value in slots:
        string = string.replace('%'+key+'%', value)
    return string

#连接Neo4j
def get_neo4j_conn():
    uri = os.getenv('NEO4J_URI', '?')
    logger.debug("连接 Neo4j | uri=%s", uri)
    return Graph(
        uri,
        auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )
