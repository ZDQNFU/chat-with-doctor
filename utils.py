import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from py2neo import Graph
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

#加载环境变量
load_dotenv()

def get_embedding_model():
    """获取词嵌入模型"""
    model_path = os.getenv('BGE_MODEL_PATH')
    # 如果 .env 里有路径就用，否则回退到在线模型名
    if model_path:
        model_name = model_path
    else:
        model_name = os.getenv('BGE_EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')

    model_map = {
        'bge': HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs={
                'device': os.getenv('EMBEDDING_DEVICE')  # GPU:'cuda'
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
    }
    model_type = os.getenv('EMBEDDING_MODEL', 'bge')
    return model_map.get(model_type)


def get_llm_model():
    """获取大语言模型"""
    model_map = {
        'deepseek': ChatOpenAI(
            model=os.getenv('DEEPSEEK_LLM_MODEL'),
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=os.getenv('DEEPSEEK_BASE_URL'),
            temperature=os.getenv('TEMPERATURE'),
            max_tokens=int(os.getenv('MAX_TOKENS')),
            extra_body={"thinking": {"type": "disabled"}}
        )
    }
    model_type = os.getenv('LLM_MODEL', 'deepseek')
    return model_map.get(model_type)

def chat_with_agent(agent, user_input, thread_id="default-thread"):
    """简化的聊天接口"""
    # 统一处理输入格式
    if isinstance(user_input, str):
        messages = [HumanMessage(content=user_input)]
    else:
        messages = user_input

    config = {"configurable": {"thread_id": thread_id}}
    return agent.invoke({"messages": messages}, config=config)


def stream_chat_with_agent(agent, user_input, thread_id="default-thread"):
    config = {"configurable": {"thread_id": thread_id}}
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages"
    ):
        if isinstance(chunk, tuple) and len(chunk) == 2:
            message, metadata = chunk
            # 只输出AI生成的内容，跳过工具消息等
            if isinstance(message, AIMessageChunk) and message.content:
                if isinstance(message.content, str):
                    yield message.content
                elif isinstance(message.content, list):
                    for item in message.content:
                        if isinstance(item, dict) and 'text' in item:
                            yield item['text']
                        elif isinstance(item, str):
                            yield item

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
    return Graph(
        os.getenv('NEO4J_URI'),
        auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )
