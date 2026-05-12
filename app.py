import gradio as gr
from service import Service
from file_watcher import start_file_watcher
import threading

# 在后台线程中启动监控
watcher_thread = threading.Thread(target=start_file_watcher, daemon=True)
watcher_thread.start()

# 创建全局 Service 实例，避免每次都重新加载模型
service = Service()

def doctor_bot(message, history):
    # 生成器函数，支持流式输出
    response = service.answer(message, history)
    yield response

demo = gr.ChatInterface(
    fn=doctor_bot,
    title='医疗问诊机器人',
    description='基于知识图谱的医疗问答助手',
    examples=['鼻炎是一种什么病？', '一般会有哪些症状？'],
    concurrency_limit=5,  # 限制并发数
)

if __name__ == '__main__':
    demo.queue()  # 启用队列
    demo.launch(share=False)