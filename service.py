from agent import *
import os
from logger_config import get_logger

os.environ['CHROMA_TELEMETRY_IMPL'] = 'none'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

logger = get_logger(__name__)


class Service:
    def __init__(self):
        self.thread_id = "user_ZDQNFU"
        logger.info("Service 初始化 | thread_id=%s", self.thread_id)
        self.agent_manager = Agent()
        self.agent = self.agent_manager.get_agent()

    def stream_answer(self, query):
        logger.info("[%s] 收到用户请求 | query=%s", self.thread_id, query[:80])
        for token in stream_chat_with_agent(
            self.agent,
            query,
            self.thread_id
        ):
            yield token
