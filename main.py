import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from service import Service
from file_watcher import start_file_watcher
from logger_config import setup_logging, get_logger
import threading

# ── 初始化日志系统 ──
setup_logging()
logger = get_logger(__name__)

# 在后台线程中启动监控
watcher_thread = threading.Thread(target=start_file_watcher, daemon=True)
watcher_thread.start()

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


# ── 请求日志中间件 ──
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个 HTTP 请求的耗时与状态码"""
    t_start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t_start
    logger.info(
        "%s %s → %d | latency=%.3fs",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


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
        logger.exception("POST /ask 处理失败 | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI 服务启动 | host=127.0.0.1 port=8000")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
