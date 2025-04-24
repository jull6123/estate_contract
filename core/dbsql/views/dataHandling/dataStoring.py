import json
from zipfile import Path

from langchain_community.document_loaders import (
    JSONLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# import os
# import faiss
# from zhipuai import ZhipuAI
# import numpy as np
# # from split import split_pdf, storage_pdf
#
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# client = ZhipuAI(api_key="eee3ab03c49ebe869b2bd6bd49c29c37.n6SyFHxFLgWrgjlG")

# # 生成向量
# def generate_vector(text):
#     response = client.embeddings.create(
#         model="embedding-2",  # 填写需要调用的模型名称
#         input=text,
#     )
#     vector = response.data[0].embedding
#     return np.array(vector, dtype=np.float32)

# 1. 初始化嵌入模型
embedding = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'}  # GPU加速
)

# 2. 文本分块配置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "；"]
)


def load_cases(json_path):
    # 自定义元数据提取函数
    def metadata_func(record: dict, metadata: dict):
        metadata["source"] = record.get("CaseId")
        metadata["case_type"] = record.get("Category", {}).get("cat_2")
        metadata["title"] = record.get("Case")
        return metadata

    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".[]",
        content_key="JudgeReason",  # 主要使用裁判理由
        metadata_func=metadata_func
    )
    docs = loader.load()

    # 添加其他字段内容
    for doc in docs:
        record = json.loads(doc.page_content)
        doc.page_content = f"""
        案件名称：{record.get('Case')}
        案由：{record.get('JudgeAccusation')}
        裁判理由：{record.get('JudgeReason')}
        裁判结果：{record.get('JudgeResult')}
        """
    return text_splitter.split_documents(docs)


def load_laws(word_dir):
    loader = DirectoryLoader(
        word_dir,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True
    )
    docs = loader.load()

    # 添加文档类型元数据
    for doc in docs:
        doc.metadata["doc_type"] = "law"
        # 提取法规名称（从文件名或内容）
        doc.metadata["law_name"] = Path(doc.metadata["source"]).stem
    return text_splitter.split_documents(docs)


def load_contracts(word_dir):
    loader = DirectoryLoader(
        word_dir,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True
    )
    docs = loader.load()

    # 合同特定处理
    for doc in docs:
        doc.metadata["doc_type"] = "contract"
        # 从文件名提取地区信息（示例：南京_房屋租赁.docx）
        filename = Path(doc.metadata["source"]).stem
        doc.metadata["region"] = filename.split("_")[0]
        doc.metadata["contract_type"] = filename.split("_")[1]
    return text_splitter.split_documents(docs)


def build_knowledge_base():
    # 加载所有数据
    case_docs = load_cases("data/cases.json")
    law_docs = load_laws("data/laws/")
    contract_docs = load_contracts("data/contracts/")

    # 分别创建向量库（便于单独检索）
    case_db = FAISS.from_documents(case_docs, embedding)
    law_db = FAISS.from_documents(law_docs, embedding)
    contract_db = FAISS.from_documents(contract_docs, embedding)

    # 保存本地
    case_db.save_local("vector_stores/case_db")
    law_db.save_local("vector_stores/law_db")
    contract_db.save_local("vector_stores/contract_db")


import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
from langchain.schema import Document
from config import EMBEDDING_MODEL
from langchain.embeddings import HuggingFaceEmbeddings


class KeywordFAISS:
    def __init__(self, embedding_model=None):
        self.index = None
        self.doc_store = []
        self.embedder = embedding_model or HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    def add_documents(self, docs: List[Document]):
        """添加文档到向量库"""
        if not docs:
            return

        # 初始化索引（首次添加时）
        if self.index is None:
            dim = len(self.embedder.embed_query("test"))
            self.index = faiss.IndexFlatIP(dim)

        # 提取关键词并生成嵌入
        embeddings = []
        for doc in docs:
            keyword_text = " ".join(doc.metadata["keywords"])
            emb = self.embedder.embed_query(keyword_text)
            embeddings.append(emb)
            self.doc_store.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        # 添加到索引
        self.index.add(np.array(embeddings).astype('float32'))

    def save_local(self, path: str):
        """保存到本地"""
        if not self.index:
            raise ValueError("Index not initialized")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存FAISS索引
        faiss.write_index(self.index, str(path / "index.faiss"))

        # 保存文档存储
        with open(path / "docs.json", 'w', encoding='utf-8') as f:
            json.dump(self.doc_store, f, ensure_ascii=False)

    @classmethod
    def load_local(cls, path: str, embedding_model=None):
        """从本地加载"""
        path = Path(path)
        instance = cls(embedding_model)

        # 加载FAISS索引
        instance.index = faiss.read_index(str(path / "index.faiss"))

        # 加载文档存储
        with open(path / "docs.json", 'r', encoding='utf-8') as f:
            instance.doc_store = json.load(f)

        return instance

    def similarity_search(self, query_keywords: List[str], k: int = 4):
        """基于关键词搜索"""
        if not self.index:
            return []

        # 生成查询嵌入
        query_embed = self.embedder.embed_query(" ".join(query_keywords))

        # 执行搜索
        distances, indices = self.index.search(
            np.array([query_embed]).astype('float32'),
            k
        )

        # 返回结果
        return [
            {
                "text": self.doc_store[i]["text"],
                "metadata": self.doc_store[i]["metadata"],
                "score": float(distances[0][j])
            }
            for j, i in enumerate(indices[0])
            if i >= 0
        ]