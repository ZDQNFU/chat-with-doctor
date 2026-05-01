import os
from glob import glob
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from utils import *

from langchain_community.vectorstores import Chroma

#构建一个chroma文档向量数据库，用于RAG（检索增强生成）
def doc2vec():
    # 定义文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    #读取分割文件
    dir_path = Path(__file__).parent / 'data' / 'inputs'
    documents = []

    for file_path in glob(os.path.join(dir_path, '*.*')):
        loader = None
        if '.csv' in file_path:
            loader = CSVLoader(file_path, encoding='utf-8')
        if '.txt' in file_path:
            loader = TextLoader(file_path, encoding='utf-8')
        if '.pdf' in file_path:
            loader = PyMuPDFLoader(file_path)
        if loader:
            #分割
            documents+=loader.load_and_split(text_splitter)
    print(documents)
    #向量化并存储
    if documents:
        vdb = Chroma.from_documents(
            documents,
            embedding = get_embedding_model(),
            persist_directory = os.path.join(os.path.dirname(__file__), './data/db'),
        )
        # vdb.persist()

if __name__ == '__main__':
    doc2vec()