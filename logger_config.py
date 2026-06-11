"""
统一日志配置模块

用法（只需在入口文件调用一次 setup_logging，其余模块用 get_logger）：

    # main.py / app.py
    from logger_config import setup_logging
    setup_logging()

    # 任意模块
    from logger_config import get_logger
    logger = get_logger(__name__)
    logger.info("...")

日志输出：
  - 控制台：实时可读（WARNING 以上带高亮）
  - 文件：logger/YYYY-MM-DD.log，按天轮转，保留 30 天
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone, timedelta

# ── 时区：北京时间 ──
_TZ = timezone(timedelta(hours=8))


class _BeijingFormatter(logging.Formatter):
    """北京时间 + 毫秒"""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f".{int(record.msecs):03d}"


_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
)

# ── 单次初始化标记 ──
_initialized = False


def setup_logging(
    log_dir: str | None = None,
    level: str | None = None,
    *,
    console: bool = True,
) -> None:
    """
    初始化全局日志系统。

    参数:
        log_dir: 日志目录，默认为项目根目录下的 logger/
        level:   日志级别（DEBUG/INFO/WARNING/ERROR），默认从 LOG_LEVEL 环境变量读取，
                 环境变量未设时用 INFO
        console: 是否同时输出到控制台
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # ── 确定日志目录 ──
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "logger")
    os.makedirs(log_dir, exist_ok=True)

    # ── 确定日志级别 ──
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level, logging.INFO)

    # ── 根 Logger ──
    root = logging.getLogger()
    root.setLevel(log_level)
    # 清空已有的 handler（避免重复添加）
    root.handlers.clear()

    # ── 控制台 Handler ──
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(_BeijingFormatter(_FORMAT))
        root.addHandler(console_handler)

    # ── 按天轮转文件 Handler ──
    log_file = os.path.join(log_dir, "default.log")  # 哨兵文件名，实际会被 TimedRotating 重命名
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(_BeijingFormatter(_FORMAT))
    # 自定义 namer：让轮转后的文件名变成 YYYY-MM-DD.log
    file_handler.namer = lambda default_name: _daily_namer(default_name, log_dir)
    root.addHandler(file_handler)

    # ── 降低第三方库日志噪音 ──
    for lib in ("httpx", "httpcore", "urllib3", "openai", "chromadb", "chromadb.telemetry", "watchdog"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # ── 启动提示 ──
    root.info("日志系统初始化完成 | level=%s dir=%s", level, log_dir)


def _daily_namer(default_name: str, log_dir: str) -> str:
    """从默认轮转文件名提取日期，重命名为 YYYY-MM-DD.log"""
    base = os.path.basename(default_name)  # e.g. "default.log.2026-06-11"
    parts = base.rsplit(".", 1)
    if len(parts) == 2:
        date_str = parts[1]  # "2026-06-11"
        return os.path.join(log_dir, f"{date_str}.log")
    return os.path.join(log_dir, base)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger（确保 setup_logging 已调用过，否则会自动补调）"""
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)
