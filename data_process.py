import os
from glob import glob
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from utils import *

from langchain_community.vectorstores import Chroma


def get_text_splitter(method='recursive', chunk_size=300, chunk_overlap=50):
    """
    获取文本分割器

    参数:
        method: 分割方法
            - 'recursive': 递归字符分割（原有方法）
            - 'semantic': 语义分割（基于嵌入模型）
            - 'hybrid': 混合分割（结合结构和语义）
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
    """

    if method == 'recursive':
        # 方法1: 递归字符分割（原有方法）
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

    elif method == 'semantic':
        # 方法2: 语义分割（基于嵌入模型的语义相似度）
        from langchain_text_splitters import SemanticChunker
        return SemanticChunker(
            embeddings=get_embedding_model(),
            buffer_size=1,
            add_start_index=True,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=95,
        )

    elif method == 'hybrid':
        # 方法3: 混合分割（先按结构分割，再合并语义相似的块）
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 第一步：使用较大的递归分割
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 2,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True
        )
        return base_splitter

    else:
        raise ValueError(f"Unsupported method: {method}. Use 'recursive', 'semantic', or 'hybrid'.")


def merge_semantic_chunks(documents, min_chunk_size=200, max_chunk_size=500):
    """
    后处理：合并过小的语义块，确保语义完整性
    """
    merged_docs = []
    current_doc = None

    for doc in documents:
        if current_doc is None:
            current_doc = doc
        else:
            # 如果当前块和前一个块都很小，且来自同一来源，尝试合并
            if (len(current_doc.page_content) + len(doc.page_content) < max_chunk_size and
                current_doc.metadata.get('source') == doc.metadata.get('source')):
                from langchain_core.documents import Document
                merged_content = current_doc.page_content + "\n" + doc.page_content
                current_doc = Document(
                    page_content=merged_content,
                    metadata=current_doc.metadata
                )
            else:
                merged_docs.append(current_doc)
                current_doc = doc

    if current_doc:
        merged_docs.append(current_doc)

    return merged_docs


def doc2vec(method='hybrid', chunk_size=300, chunk_overlap=50):
    """
    构建chroma文档向量数据库，用于RAG（检索增强生成）

    参数:
        method: 分割方法 ('recursive', 'semantic', 'hybrid')
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
    """
    # 获取文本分割器
    text_splitter = get_text_splitter(method, chunk_size, chunk_overlap)

    # 读取分割文件
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
            try:
                # 加载文档
                raw_docs = loader.load()

                # 分割文档
                split_docs = text_splitter.split_documents(raw_docs)

                # 如果是混合方法，进行后处理合并
                if method == 'hybrid':
                    split_docs = merge_semantic_chunks(split_docs)

                documents.extend(split_docs)
                print(f"处理文件: {os.path.basename(file_path)}, 生成 {len(split_docs)} 个chunks")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")

    print(f"\n总共生成 {len(documents)} 个文档chunks")

    # 向量化并存储
    if documents:
        vdb = Chroma.from_documents(
            documents,
            embedding = get_embedding_model(),
            persist_directory = os.path.join(os.path.dirname(__file__), './data/db'),
        )
        print("向量数据库构建完成！")
        return vdb
    else:
        print("警告: 没有生成任何文档chunks")
        return None


if __name__ == '__main__':
    # 可以选择不同的分割方法:
    # method='recursive'  - 传统的递归字符分割
    # method='semantic'   - 基于语义的分割（langchain-experimental）
    # method='hybrid'     - 混合分割（推荐）

    doc2vec(method='hybrid', chunk_size=300, chunk_overlap=50)