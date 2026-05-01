from prompt import *
from utils import *
from agent import *

from langchain.chains import LLMChain
from langchain.prompts import Prompt

class Service():
    def __init__(self):
        self.agent = Agent()

    def get_summary_message(self, message, history):
        llm = get_llm_model()
        prompt = Prompt.from_template(SUMMARY_PROMPT_TPL)
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=os.getenv('VERBOSE'))
        chat_history = ''

        # 兼容新旧两种 history 格式
        for item in history[-2:]:
            if isinstance(item, dict):
                # 新格式: {"role": "user", "content": "..."}
                q = item.get('content', '') if item.get('role') == 'user' else ''
                a = item.get('content', '') if item.get('role') == 'assistant' else ''
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                # 旧格式: (question, answer)
                q, a = item
            else:
                continue
            chat_history += f'问题:{q}, 答案:{a}\n'

        return llm_chain.invoke({'query': message, 'chat_history': chat_history})['text']

    def answer(self, message, history):
        if history:
            message = self.get_summary_message(message, history)
        return self.agent.query(message)

# if __name__ == '__main__':
#     service = Service()
#     service.answer("肾病综合征有什么症状", "")
