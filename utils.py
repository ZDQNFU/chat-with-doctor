import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from py2neo import Graph
from langchain_community.embeddings import HuggingFaceEmbeddings
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
            temperature=float(os.getenv('TEMPERATURE', 0)),
            max_tokens=int(os.getenv('MAX_TOKENS', 1000)),
            openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
            openai_api_base=os.getenv('DEEPSEEK_BASE_URL'),
        )
    }
    model_type = os.getenv('LLM_MODEL', 'deepseek')
    return model_map.get(model_type)

#json输出格式化函数
def structured_output_parser(response_schemas):
    text = """
    请从以下文本中，抽取出实体信息，并按json格式输出。
    直接输出纯净JSON，不要包含'''json标记、换行符或任何其他文本。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段:\n
    """
    for schema in response_schemas:
        text += schema.name + ' 字段，表示: ' + schema.description + ', 类型为: ' + schema.type + '\n'
    return text

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
