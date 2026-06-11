from utils import *
from prompt import *
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from tools import retrival_func, graph_func, search_func

from logger_config import get_logger

logger = get_logger(__name__)


class Agent:
    def __init__(self):
        self.checkpointer = InMemorySaver()
        logger.info("Agent 管理器初始化完成")

    def get_agent(self):
        agent = create_agent(
            model = get_llm_model(),
            tools = [
                retrival_func,
                graph_func,
                search_func,
            ],
            system_prompt = SYSTEM_PROMPT,
            # middleware=[
            #     SummarizationMiddleware(
            #         model=get_llm_model(),
            #         trigger=("tokens", 4000),  # 达到4000 tokens时触发总结
            #         keep=("messages", 10),  # 保留最近10条消息
            #     )
            # ],
            checkpointer=self.checkpointer,
        )
        logger.info("Agent 实例创建完成 | tools=%d", 3)
        return agent
