import json

import contractReptilia
import caseDataCleaning
import lawDataProcesse
from pathlib import Path
import os
import faiss
from zhipuai import ZhipuAI
import numpy as np
import json
from typing import List, Dict
import spacy
from langchain.schema import Document

nlp = spacy.load("zh_core_web_lg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
client = ZhipuAI(api_key="eee3ab03c49ebe869b2bd6bd49c29c37.n6SyFHxFLgWrgjlG")


def extract_legal_keywords(text: str) -> List[str]:
    if not text.strip():
        return []
    doc = nlp(text)
    keywords = set()

    for ent in doc.ents:
        if ent.label_ in ["LAW", "ORG", "LOC"]:
            keywords.add(ent.text)
    for token in doc:
        if (token.pos_ in ["NOUN", "VERB"] and
                token.dep_ in ["nsubj", "dobj", "ROOT"]):
            keywords.add(token.lemma_)

    return [k for k in keywords if len(k) > 1]


class CaseProcessor:
    @staticmethod
    def process_file(dir_path: str) -> List[Document]:
        # 获取该地址下所有json文件
        json_files = list(Path(dir_path).glob('*.json'))
        docs = []
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:
                cases = json.load(f)

            for case in cases:
                content = f"""
                案件名称：{case.get('Case')}
                案由：{case.get('JudgeAccusation', '')}
                裁判理由：{case.get('JudgeReason', '')}
                裁判结果：{case.get('JudgeResult', '')}
                """

                # 提取关键词
                keywords = extract_legal_keywords(content)

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "doc_id": case.get("CaseId"),
                        "type": "case",
                        "category": case.get("Category", {}).get("cat_2"),
                        "keywords": keywords
                    }
                ))
        return docs


class LawProcessor:
    @staticmethod
    def process_file(file_path: str) -> List[Document]:
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])

        # 按条款分割
        clauses = [t for t in text.split("第") if t.strip()][1:]  # 跳过开头

        docs = []
        for i, clause in enumerate(clauses, 1):
            clause_text = f"第{i}条 {clause.strip()}"
            docs.append(Document(
                page_content=clause_text,
                metadata={
                    "doc_id": f"{file_path.stem}_clause_{i}",
                    "type": "law",
                    "keywords": extract_legal_keywords(clause_text),
                    "source": file_path.name
                }
            ))
        return docs



# 生成向量
def generate_vector(text):
    response = client.embeddings.create(
        model="embedding-2",  # 填写需要调用的模型名称
        input=text,
    )
    vector = response.data[0].embedding
    return np.array(vector, dtype=np.float32)


# 获取摘要
# def get_summary(paragraphs):
#     prompt = f"""
#         提取这段文本的主题，主题需要包含所有实体。不超过300字，只需要输出内容
#
#        原文： {paragraphs}
#        """
#     # ## Goals
#     # 提取这段文本的主题，主题需要包含所有实体。
#     #
#     # ## Constrains
#     # 答案必须基于提供的文本，不得添加编造成分，使用中文回答，不超过300字，只需要输出内容。
#     messages = [{"role": "user", "content": prompt}]
#     response = client.chat.completions.create(
#         model="glm-4",
#         messages=messages,
#     )
#     return response.choices[0].message.content


# # 根据检索结果得到相近文件名列表
# def get_filename(filenames, results):
#     sources_rag = [source[0] for _, source, _ in results]
#     sources_rag = list(set(sources_rag))
#     references = []
#     for index, filename in enumerate(filenames):
#         if filename[0] in sources_rag:
#             references.append(filename[1])
#     print(f"references: {references}")
#     return references


# 根据检索结果减少indexs
def decrease_index(results, indexs, texts, sources):
    # 检索结果的到文件的序号
    sources_rag = [source[0] for _, source, _ in results]
    sources_rag = list(set(sources_rag))
    vectors = indexs.reconstruct_n(0, indexs.ntotal)

    sources_file = []
    texts_file = []
    vectors_file = []
    for source, text, vector in zip(sources, texts, vectors):
        if source[0] in sources_rag:
            sources_file.append(source)
            texts_file.append(text)
            vectors_file.append(vector)
    print("\n检索内容减少...")
    print(f"sources_file: {sources_file}")

    vectors_file = np.array(vectors_file)
    if len(vectors_file) > 0:
        indexs_file = faiss.IndexFlatL2(vectors_file.shape[1])
        indexs_file.add(vectors_file)
    return sources_file, texts_file, indexs_file


# # 得到目录下的pdf文件路径列表
# def get_all_pdfs(dir_path):
#     pdf_paths = []
#     for root, dirs, files in os.walk(dir_path):
#         print(f'正在访问目录: {root}')
#         for file in files:
#             if not (file.startswith('._') or os.path.join(root, file).startswith('__MACOSX')):
#                 if file.lower().endswith('.pdf'):
#                     pdf_path = os.path.join(root, file)
#                     print(f'找到PDF文件: {pdf_path}')
#                     pdf_paths.append(pdf_path)
#     return pdf_paths

# 得到目录下的pdf文件路径列表
def get_all_files(dir_path):
    pdf_paths = []
    for root, dirs, files in os.walk(dir_path):
        print(f'正在访问目录: {root}')
        for file in files:
            if not (file.startswith('._') or os.path.join(root, file).startswith('__MACOSX')):
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    print(f'找到PDF文件: {pdf_path}')
                    pdf_paths.append(pdf_path)
    return pdf_paths


# 数据集的首次存储
def create_vector_store_from_directory(directory_path):
    print("从目录中解析所有文件...")
    vectors = []
    texts = []
    i = 0
    files = get_all_files(directory_path)
    print(f'\n找到 {len(pdf_files)} 个PDF文件:')
    if len(pdf_files) == 0:
        return None, None, None, None, None, None, None, 0
    else:
        for pdf in pdf_files:
            file_path = pdf
            filenames.append((i, file_path))
            print(f"\nProcessing file: {file_path}")

            min_paragraphs = split_pdf.process_uploaded_pdf(file_path, i, len(pdf_files))
            summary = get_summary(min_paragraphs)
            summaries.append(summary)
            sources_summary.append((i, i))
            vector_s = generate_vector(summary)
            vectors_summary.append(vector_s)
            for index, text in enumerate(min_paragraphs):
                if text.strip():  # 忽略空段落
                    texts.append(text)
                    sources.append((i, index))
                    vector = generate_vector(text)
                    vectors.append(vector)
            i += 1

        vectors = np.array(vectors)
        vectors_summary = np.array(vectors_summary)
        if len(vectors) > 0 and len(vectors_summary) > 0:
            indexs = faiss.IndexFlatL2(vectors.shape[1])
            indexs.add(vectors)
            indexs_summary = faiss.IndexFlatL2(vectors_summary.shape[1])
            indexs_summary.add(vectors_summary)

        return indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames, 1


# 更新：处理目录中的未添加过的PDF文件
def create_vector_store_from_directory_update(directory_path, filenames):
    print("从目录中解析未添加过的pdf文件...")
    vectors = []
    vectors_summary = []
    texts = []
    sources = []
    sources_summary = []
    summaries = []
    filenames_add = []
    filenames_now = [filename for index, filename in filenames]
    i = len(filenames_now)

    pdf_files = get_all_pdfs(directory_path)
    print(f'pdf文件进度：{len(filenames_now)} / {len(pdf_files)}')
    if len(pdf_files) <= len(filenames_now):
        return None, None, None, None, None, None, None, 0
    else:
        for pdf in pdf_files:
            if pdf not in filenames_now:
                file_path = pdf
                filenames_add.append((i, file_path))
                print(f"\nProcessing file: {file_path}")

                min_paragraphs = split_pdf.process_uploaded_pdf(file_path, i, len(pdf_files))
                summary = get_summary(min_paragraphs)
                summaries.append(summary)
                sources_summary.append((i, i))
                vector_s = generate_vector(summary)
                vectors_summary.append(vector_s)
                for index, text in enumerate(min_paragraphs):
                    if text.strip():  # 忽略空段落
                        texts.append(text)
                        sources.append((i, index))
                        vector = generate_vector(text)
                        vectors.append(vector)
                i += 1

        vectors = np.array(vectors)
        vectors_summary = np.array(vectors_summary)
        if len(vectors) > 0 and len(vectors_summary) > 0:
            indexs = faiss.IndexFlatL2(vectors.shape[1])
            indexs.add(vectors)
            indexs_summary = faiss.IndexFlatL2(vectors_summary.shape[1])
            indexs_summary.add(vectors_summary)
        print("文件摘要...")
        for filename, summary in zip(filenames_add, summaries):
            print(f"filenames_add: {filenames_add}\nsummary: {summary}\n\n")
        return indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames_add, 1


# 初始化，得到数据
def get_init(directory_path):
    case_vector_path = 'D:\program_files\estate_data\case_data_vector.pkl'
    contract_vector_path = 'D:\program_files\estate_data\contarct_data_vector.pkl'
    law_vector_path = 'D:\program_files\estate_data\law_data_vector.pkl'
    # indexs_summary_path = 'storage_data/vectors_summary.pkl'
    # summaries_file_path = 'storage_data/summaries.pkl'
    # text_file_path = 'storage_data/texts.pkl'
    # text_index_path = 'storage_data/sources.pkl'
    # summary_index_path = 'storage_data/sources_summary.pkl'
    # filename_file_path = 'storage_data/filenames.pkl'

    if (os.path.exists(case_vector_path) and os.path.exists(contract_vector_path)
        and os.path.exists(law_vector_path)) :
        print("正在加载预处理数据...")
        indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames, flag = create_vector_store_from_directory_update(
            directory_path, storage_pdf.get_filenames_from_pkl(filename_file_path))
        if flag == 1:
            print(f"更新知识库...")
            storage_pdf.save_data(indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames,
                                  indexs_vector_path, indexs_summary_path, text_file_path, summaries_file_path,
                                  text_index_path, summary_index_path, filename_file_path)
        indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames = storage_pdf.load_data(
            indexs_vector_path, indexs_summary_path,
            text_file_path, summaries_file_path,
            text_index_path, summary_index_path,
            filename_file_path)
        print("预处理数据加载完成。")
    else:
        print("解析文件数据并完成初次存储...")
        indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames, flag = create_vector_store_from_directory(
            directory_path)
        if flag == 1:
            storage_pdf.save_data(indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames,
                                  indexs_vector_path, indexs_summary_path, text_file_path, summaries_file_path,
                                  text_index_path, summary_index_path, filename_file_path)
            print("文件数据处理完成。")
        else:
            return indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames, 0
    return indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames, 1


# 向量检索摘要
def vector_search_by_summary(index, texts, sources, query, k=5, threshold=0.9):
    print("\n摘要向量检索...")
    query_vector = generate_vector(query)
    D, I = index.search(np.array([query_vector]), k)
    results = [(texts[i], sources[i], D[0][j]) for j, i in enumerate(I[0])]
    print(f"all...")
    for result in results:
        print(f"result: {result}")
    filtered_results = [result for result in results if result[2] < threshold]
    print(f"\nthreshold = {threshold}...")
    for resultf in filtered_results:
        print(f"result: {resultf}")
    return filtered_results


# 文件向量检索
def vector_search_by_text_decrease(results, indexs, texts, sources, query, k=5, threshold=0.7):
    query_vector = generate_vector(query)
    sources, texts, indexs = decrease_index(results, indexs, texts, sources)
    D, I = indexs.search(np.array([query_vector]), k)
    results = [(texts[i], sources[i], D[0][j]) for j, i in enumerate(I[0])]
    print("\n\n文件向量检索...")
    print(f"all...")
    for result in results:
        print(f"result: {result}")
    filtered_results = [result for result in results if result[2] < threshold]
    print(f"\nthreshold = {threshold}...")
    for resultf in filtered_results:
        print(f"result: {resultf}")
    return filtered_results


def vector_search_by_text(indexs, texts, sources, query, k=5, threshold=0.7):
    query_vector = generate_vector(query)
    D, I = indexs.search(np.array([query_vector]), k)
    results = [(texts[i], sources[i], D[0][j]) for j, i in enumerate(I[0])]
    print("\n\n文件向量检索...")
    print(f"all...")
    for result in results:
        print(f"result: {result}")
    filtered_results = [result for result in results if result[2] < threshold]
    print(f"\nthreshold = {threshold}...")
    for resultf in filtered_results:
        print(f"result: {resultf}")
    return filtered_results


# 生成回答 降低temperature，减少回答的创造性
def get_completion(query, context):
    prompt = f"""
    #  Role: 信息检索 

    ## Goals
    根据提供的知识库、上下文信息回答用户的问题。 

    ## Constrains
    答案必须基于提供的知识库与上下文信息，不得添加编造成分，使用中文回答，只需要回复消息。

    ## Skills
    理解并应用知识库内容，简洁专业地回答问题。

    ## Output Format
    简洁明了的中文回答，若无法回答则告知“根据已知信息无法回答该问题”或“没有提供足够的相关信息”。

    请遵循上述要求，完成如下问题：
    上下文信息： {context}
    问题： {query}

    """

    messages = [{"role": "user", "content": query, "context": prompt}]
    response = client.chat.completions.create(
        model='glm-4',
        messages=messages,
        max_tokens=300,  # 增加最大token数
        temperature=0,
        stop=None  # 确保不会提前停止
    )
    return response.choices[0].message.content


# 两次检索，RAG问答
def rag_qa_with_external_validation_by_sum(query, directory_path, k=5, threshold_summary=0.95, threshold=0.75):
    indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames, flag = get_init(directory_path)
    if flag == 1:
        # indexs_summary, summaries, texts, sources_summary, sources, filenames = get_init(directory_path)
        result_summary = vector_search_by_summary(indexs_summary, summaries, sources_summary, query, k,
                                                  threshold_summary)
        if result_summary:
            get_filename(filenames, result_summary)
            # results = vector_search_by_text_decrease(result_summary, texts, sources, query, k=5, threshold=0.7)
            results = vector_search_by_text_decrease(result_summary, indexs, texts, sources, query, k, threshold)
            if results:
                context = "  _   ".join([content for content, source, distance in results])
                references = get_filename(filenames, results)
                refer = "\n".join([reference.replace('\\', '/').split('/')[-1] for reference in references])
                response = get_completion(query, context)
                return response, context, refer
        return "很抱歉，我们没有关于此主题的足够信息。你能换个说法吗？", "", []
    else:
        return "知识库为空，请先上传PDF文件", "", []


def rag_qa_with_external_validation_by_one(query, directory_path, k=5, threshold_summary=0.95, threshold=0.75):
    indexs, indexs_summary, texts, summaries, sources, sources_summary, filenames, flag = get_init(directory_path)
    if flag == 1:
        results = vector_search_by_text(indexs, texts, sources, query, k, threshold)
        if results:
            context = "  _   ".join([content for content, source, distance in results])
            references = get_filename(filenames, results)
            refer = "\n".join([reference.replace('\\', '/').split('/')[-1] for reference in references])
            response = get_completion(query, context)
            return response, context, refer
        return "很抱歉，我们没有关于此主题的足够信息。你能换个说法吗？", "", []
    else:
        return "知识库为空，请先上传PDF文件", "", []
