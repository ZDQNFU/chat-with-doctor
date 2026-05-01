from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from service import Service

app = FastAPI(title="医疗问诊机器人 API", description="基于知识图谱的医疗问答服务")

# 允许跨域（如果前端页面需要调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局单例
service = Service()

# 请求体
class AskRequest(BaseModel):
    message: str

# 响应体
class AskResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """医疗问答接口"""
    try:
        answer = service.answer(request.message, history=None)
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)