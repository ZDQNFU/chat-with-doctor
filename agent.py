import prompt
from utils import *
from config import *
from prompt import *
import requests
import json
import os
from langchain.chains import LLMChain, LLMRequestsChain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain import hub

class Agent():
    def __init__(self):
        #加载文档
        self.vdb = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__),'./data/db'),
            embedding_function = get_embedding_model()
        )

    #通用对话，大模型自身知识
    def generic_func(self, query):
        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
        llm_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        return llm_chain.run(query)

    #公司相关内容，Chroma 向量数据库
    def retrival_func(self, query):
        documents = self.vdb.similarity_search_with_relevance_scores(query, k=5)
        query_result = [doc[0].page_content for doc in documents if doc[1]>0.5]
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        #填充提示词并总结答案
        retrival_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) > 0 else '没有查到'
        }
        return retrival_chain.run(inputs)

    #疾病相关内容，查询知识图谱Neo4j
    def graph_func(self, query):
        # 命名实体识别
        response_schemas = [
            ResponseSchema(type='list', name='disease', description='疾病名称实体'),
            ResponseSchema(type='list', name='symptom', description='疾病症状实体'),
            ResponseSchema(type='list', name='drug', description='药品名称实体'),
        ]

        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        format_instructions = structured_output_parser(response_schemas)

        ner_prompt = PromptTemplate(
            template=NER_PROMPT_TPL,
            partial_variables={'format_instructions': format_instructions},
            input_variables=['query']
        )

        ner_chain = LLMChain(
            llm=get_llm_model(),
            prompt=ner_prompt,
            verbose=os.getenv('VERBOSE')
        )

        # result = ner_chain.invoke({
        #     'query': query
        # })['text']
        #
        # ner_result = output_parser.parse(result)
        result = ner_chain.invoke({
            'query': query
        })['text']

        # 打印 LLM 原始返回，方便调试
        print(f"NER原始返回: {result}")

        # 容错解析
        try:
            ner_result = output_parser.parse(result)
        except Exception as e:
            ner_result = {"disease": [], "symptom": [], "drug": []}

        # 命名实体识别结果，填充模板
        graph_templates = []
        for key, template in GRAPH_TEMPLATE.items():
            slot = template['slots'][0]
            slot_values = ner_result[slot]
            for value in slot_values:
                graph_templates.append({
                    'question': replace_token_in_string(template['question'], [[slot, value]]),
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                })
        if not graph_templates:
            return

            # 计算问题相似度，筛选最相关问题
        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates
        ]
        db = FAISS.from_documents(graph_documents, get_embedding_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)
        # print(graph_documents_filter)

        # 执行CQL，拿到结果
        query_result = []
        neo4j_conn = get_neo4j_conn()
        for document in graph_documents_filter:
            question = document[0].page_content
            cypher = document[0].metadata['cypher']
            answer = document[0].metadata['answer']
            try:
                result = neo4j_conn.run(cypher).data()
                if result and any(value for value in result[0].values()):
                    answer_str = replace_token_in_string(answer, list(result[0].items()))
                    query_result.append(f'问题：{question}\n答案：{answer_str}')
            except:
                pass
        # print(query_result)

        # 总结答案
        prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
        graph_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
        }
        return graph_chain.invoke(inputs)['text']

    #以上方法都没有答案时，调用搜索引擎查询答案
    def search_func(self, query):
        search_url = os.getenv('SEARCH_URL')
        search_api_key = os.getenv('SEARCH_API_KEY')
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )

        payload = {
            "query": query,
            "summary": True,
            "count": 10
        }
        headers = {
            'Authorization': 'Bearer ' + search_api_key,
            'Content-Type': 'application/json'
        }

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

        inputs = {
            'query': query,
            'query_result': query_result
        }

        return llm_chain.invoke(inputs)['text']


    def query(self, query):
        tools = [
            Tool.from_function(
                name='generic_func',
                func=lambda x: self.generic_func(query),
                description='可以解答通用领域的知识，例如打招呼，问你是谁等问题',
            ),
            Tool.from_function(
                name='retrival_func',
                func=lambda x: self.retrival_func(query),
                description='用于回答寻医问药网相关问题',
            ),
            Tool(
                name='graph_func',
                func=lambda x: self.graph_func(query),
                description='用于回答疾病、症状、药物等医疗相关问题',
            ),
            Tool(
                name='search_func',
                func=self.search_func,
                description='其他工具没有正确答案时，通过搜索引擎，回答通用类问题，例如现在几点了？蔡徐坤是谁',
            ),
        ]

        # prefix = """请用中文，尽你所能回答以下问题。您可以使用以下工具："""
        # suffix = """Begin!

        # History: {chat_history}
        # Question: {input}
        # Thought:{agent_scratchpad}"""

        # agent_prompt = ZeroShotAgent.create_prompt(
        #     tools=tools,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=['input', 'agent_scratchpad', 'chat_history']
        # )
        # llm_chain = LLMChain(llm=get_llm_model(), prompt=agent_prompt)
        # agent = ZeroShotAgent(llm_chain=llm_chain)

        # memory = ConversationBufferMemory(memory_key='chat_history')
        # agent_chain = AgentExecutor.from_agent_and_tools(
        #     agent = agent,
        #     tools = tools,
        #     memory = memory,
        #     verbose = os.getenv('VERBOSE')
        # )
        # return agent_chain.run({'input': query})
        #拉取提示词模版
        prompt = hub.pull('hwchase17/react-chat')
        prompt.template = '''你是一个医疗问诊机器人。
        重要规则：
        1. Final Answer 必须原封不动使用 Observation 的内容，禁止自己重新回答。
        2. 如果 Observation 中包含"我是ZDQNFU打造的医疗问诊机器人"，Final Answer 也必须说这句话。
        3. 请用中文回答。
        ''' + prompt.template
        agent = create_react_agent(llm=get_llm_model(), tools=tools, prompt=prompt)
        memory = ConversationBufferMemory(memory_key='chat_history')
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            handle_parsing_errors=True,
            verbose=os.getenv('VERBOSE')
        )
        return agent_executor.invoke({"input": query})['output']