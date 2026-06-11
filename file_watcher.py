import os
import time
import hashlib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from data_process import doc2vec
from logger_config import get_logger

logger = get_logger(__name__)


class FileChangeHandler(FileSystemEventHandler):
    """文件变化事件处理器"""

    def __init__(self, watch_dir, debounce_seconds=2):
        """
        初始化文件变化处理器

        参数:
            watch_dir: 监控的目录路径
            debounce_seconds: 防抖时间（秒），避免频繁触发
        """
        super().__init__()
        self.watch_dir = watch_dir
        self.debounce_seconds = debounce_seconds
        self.last_event_time = 0
        self.file_hashes = {}

        # 初始化时记录所有文件的哈希值
        self._initialize_file_hashes()

    def _initialize_file_hashes(self):
        """初始化时计算所有文件的哈希值"""
        self.file_hashes = {}
        inputs_path = Path(self.watch_dir)
        if inputs_path.exists():
            for file_path in inputs_path.iterdir():
                if file_path.is_file():
                    file_hash = self._calculate_file_hash(str(file_path))
                    self.file_hashes[str(file_path)] = file_hash
        logger.info("文件监控初始化完成 | files=%d dir=%s",
                    len(self.file_hashes), self.watch_dir)

    def _calculate_file_hash(self, file_path):
        """计算文件的MD5哈希值"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning("计算文件哈希失败 | file=%s error=%s", file_path, e)
            return None

    def _has_file_changed(self, file_path):
        """检查文件是否真的发生了变化"""
        current_hash = self._calculate_file_hash(file_path)
        old_hash = self.file_hashes.get(file_path)

        if current_hash != old_hash:
            self.file_hashes[file_path] = current_hash
            return True
        return False

    def _rebuild_vector_db(self):
        """重建向量数据库"""
        logger.info("检测到文件变化，开始重建向量数据库...")

        try:
            # 使用混合方法进行文档分割和向量化
            vdb = doc2vec(method='hybrid', chunk_size=300, chunk_overlap=50)

            if vdb:
                logger.info("向量数据库重建成功")
            else:
                logger.warning("向量数据库重建失败：没有生成任何文档chunks")
        except Exception as e:
            logger.exception("向量数据库重建失败 | error=%s", e)

    def on_created(self, event):
        """处理文件创建事件"""
        if not event.is_directory:
            current_time = time.time()
            if current_time - self.last_event_time > self.debounce_seconds:
                self.last_event_time = current_time

                file_path = event.src_path
                logger.info("[新建文件] %s", os.path.basename(file_path))

                # 延迟执行，避免文件还在写入中
                time.sleep(self.debounce_seconds)

                if self._has_file_changed(file_path):
                    self._rebuild_vector_db()

    def on_modified(self, event):
        """处理文件修改事件"""
        if not event.is_directory:
            current_time = time.time()
            if current_time - self.last_event_time > self.debounce_seconds:
                self.last_event_time = current_time

                file_path = event.src_path
                logger.info("[修改文件] %s", os.path.basename(file_path))

                # 延迟执行，避免文件还在写入中
                time.sleep(self.debounce_seconds)

                if self._has_file_changed(file_path):
                    self._rebuild_vector_db()

    def on_deleted(self, event):
        """处理文件删除事件"""
        if not event.is_directory:
            current_time = time.time()
            if current_time - self.last_event_time > self.debounce_seconds:
                self.last_event_time = current_time

                file_path = event.src_path
                logger.info("[删除文件] %s", os.path.basename(file_path))

                # 从哈希字典中移除
                if file_path in self.file_hashes:
                    del self.file_hashes[file_path]

                # 重建向量数据库
                self._rebuild_vector_db()

    def on_moved(self, event):
        """处理文件移动/重命名事件"""
        if not event.is_directory:
            current_time = time.time()
            if current_time - self.last_event_time > self.debounce_seconds:
                self.last_event_time = current_time

                old_path = event.src_path
                new_path = event.dest_path
                logger.info("[重命名文件] %s -> %s",
                           os.path.basename(old_path), os.path.basename(new_path))

                # 更新哈希字典
                if old_path in self.file_hashes:
                    del self.file_hashes[old_path]

                time.sleep(self.debounce_seconds)
                new_hash = self._calculate_file_hash(new_path)
                if new_hash:
                    self.file_hashes[new_path] = new_hash

                # 重建向量数据库
                self._rebuild_vector_db()


def start_file_watcher(watch_dir=None, debounce_seconds=2):
    """
    启动文件监控器

    参数:
        watch_dir: 要监控的目录路径，默认为 data/inputs
        debounce_seconds: 防抖时间（秒）
    """
    if watch_dir is None:
        watch_dir = Path(__file__).parent / 'data' / 'inputs'

    watch_dir = str(watch_dir)

    # 确保监控目录存在
    if not os.path.exists(watch_dir):
        logger.warning("监控目录不存在，正在创建: %s", watch_dir)
        os.makedirs(watch_dir, exist_ok=True)

    logger.info("文件监控启动 | dir=%s debounce=%ds", watch_dir, debounce_seconds)

    # 创建事件处理器
    event_handler = FileChangeHandler(watch_dir, debounce_seconds)

    # 创建观察者
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=False)

    # 启动监控
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("文件监控已手动停止")
        observer.stop()

    observer.join()
    logger.info("文件监控已退出")


if __name__ == '__main__':
    from logger_config import setup_logging
    setup_logging()

    # 启动文件监控
    start_file_watcher()
