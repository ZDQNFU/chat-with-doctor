import gradio as gr
from service import Service
from file_watcher import start_file_watcher
import threading

# 后台监控文件改动（可选）
watcher_thread = threading.Thread(target=start_file_watcher, daemon=True)
watcher_thread.start()

service = Service()

def doctor_bot(message, history):
    """生成器函数：逐 token 输出，支持流式显示"""
    response = service.stream_answer(message)
    partial_text = ""
    for token in response:
        partial_text += token
        yield partial_text

demo = gr.ChatInterface(
    fn=doctor_bot,
    title='医疗问诊机器人',
    description='基于知识图谱的医疗问答助手',
    examples=['鼻炎是一种什么病？', '一般会有哪些症状？'],
    concurrency_limit=5,
)

if __name__ == '__main__':
    demo.queue()
    demo.launch(share=False)