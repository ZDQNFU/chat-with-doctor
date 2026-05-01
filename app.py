import gradio as gr
from service import Service

def doctor_bot(message, history):
    service = Service()
    return service.answer(message, history)

demo = gr.ChatInterface(
    fn=doctor_bot,
    title='医疗问诊机器人',
    description='基于知识图谱的医疗问答助手，可以回答疾病、症状、药物等相关问题',
    examples=['鼻炎是一种什么病？', '一般会有哪些症状？', '吃什么药好得快？', '高血压用什么药治疗？'],
)

if __name__ == '__main__':
    demo.launch(share=True)