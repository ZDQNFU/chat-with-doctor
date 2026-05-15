from agent import *
import os
os.environ['CHROMA_TELEMETRY_IMPL'] = 'none'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

class Service:
    def __init__(self):
        #测试用例
        self.thread_id = "user_ZDQNFU"
        # Agent Manager
        self.agent_manager = Agent()
        # 单例 Agent
        self.agent = self.agent_manager.get_agent()

    def stream_answer(self, query):
        for token in stream_chat_with_agent(
            self.agent,
            query,
            self.thread_id
        ):
            yield token

# if __name__ == '__main__':
#     service = Service()
#     while True:
#         user_input = input("\n你: ")
#         if user_input.lower() in ['quit', 'exit', '退出']:
#             print("机器人: 拜拜！")
#             break
#         if not user_input.strip():
#             continue
#
#         print("机器人: ", end="", flush=True)
#         service.stream_answer(user_input)
#         print()
