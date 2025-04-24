# import jieba
# import jieba.posseg as pseg
# from pyhanlp import HanLP
# import re
# from zhipuai import ZhipuAI
#
# #
# # class QuestionProcessor:
# #     def __init__(self):
# #         # 加载法律领域词典
# #         jieba.load_userdict('legal_terms.txt')
# #         self.client = ZhipuAI(api_key="your_api_key")
# #
# #         # 法律领域关键词库
# #         self.legal_domains = {
# #             '房地产': ['购房', '租赁', '物业', '产权', '违约金', '首付款'],
# #             '合同': ['条款', '签约', '解除', '违约责任', '补充协议']
# #         }
# #     # 核心关键词提取方法
# #     def extract_keywords(self, question):
# #         cleaned_text = re.sub(r'[^\w\s]', '', question.strip())
# #         # 领域判断
# #         if not self._is_legal_domain(cleaned_text):
# #             return None, False
# #         keywords = set()
# #
# #         # 1. 基于词性标注的名词提取
# #         words = pseg.cut(cleaned_text)
# #         for word, flag in words:
# #             if flag.startswith('n') and len(word) > 1:  # 名词且长度>1
# #                 keywords.add(word)
# #
# #         # 2. 依存句法分析提取核心成分
# #         dep_parse = HanLP.parseDependency(cleaned_text)
# #         for word in dep_parse.iterator():
# #             if word.DEPREL in ['主谓关系', '动宾关系', '核心关系']:
# #                 keywords.add(word.LEMMA)
# #
# #         # 3. 法律术语强化
# #         for term in self.legal_domains['房地产'] + self.legal_domains['合同']:
# #             if term in cleaned_text:
# #                 keywords.add(term)
# #
# #         return list(keywords), True
# #
# #     def _is_legal_domain(self, text):
# #         """判断问题是否属于法律领域"""
# #         domain_keywords = sum(self.legal_domains.values(), [])
# #         return any(keyword in text for keyword in domain_keywords)
# #     # 生成会话标题
# #     def generate_session_title(self, question, keywords):
# #         prompt = f"""
# #         请将以下法律问题概括为10字以内的标题：
# #         问题：{question}
# #         关键词：{','.join(keywords)}
# #         要求：
# #         1. 包含核心法律行为（如"违约"、"解除"）
# #         2. 包含标的物类型（如"商品房"、"商铺"）
# #         示例：
# #         - "二手房买卖定金纠纷"
# #         - "物业费缴纳争议咨询"
# #         标题："""
# #
# #         response = self.client.chat.completions.create(
# #             model="glm-4-plus",
# #             messages=[{"role": "user", "content": prompt}],
# #             temperature=0.3,
# #             max_tokens=15
# #         )
# #         return response.choices[0].message.content.strip('"')
# #     # 保证问答的有界性
# #     def process_question(self, question):
# #         """完整的处理流水线"""
# #         keywords, is_legal = self.extract_keywords(question)
# #         if not is_legal:
# #             return {
# #                 'status': 'error',
# #                 'message': '该问题不在房地产合同法律领域内，请重新提问！'
# #             }
# #
# #         title = self.generate_session_title(question, keywords)
# #
# #         return {
# #             'status': 'success',
# #             'keywords': keywords,
# #             'session_title': title
# #         }
# #
# #
# # # 使用示例
# # if __name__ == "__main__":
# #     processor = QuestionProcessor()
# #
# #     # 测试案例1：有效问题
# #     result1 = processor.process_question("租房合同到期后押金不退怎么办？")
# #     print(f"结果1：{result1}")
# #     # 输出示例：
# #     # {'status': 'success', 'keywords': ['租房', '合同', '押金'], 'session_title': '租房押金退还纠纷'}
# #
# #     # 测试案例2：非法律问题
# #     result2 = processor.process_question("如何装修我的客厅？")
# #     print(f"结果2：{result2}")
# #     # 输出：{'status': 'error', 'message': '该问题不在房地产合同法律领域内，请重新提问！'}
# #
# #
# #
#
#
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from elasticsearch import Elasticsearch
# from sentence_transformers import SentenceTransformer
#
# client = ZhipuAI(api_key="your_api_key")
# class HybridRetriever:
#     def __init__(self):
#         # 初始化向量编码模型
#         self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#
#         # 连接Elasticsearch
#         self.es = Elasticsearch(
#             hosts=["http://localhost:9200"],
#             http_auth=('elastic', 'password')
#         )
#
#         # 加载向量数据库连接
#         self.vector_db_law = ChromaClient("law_vectors")
#         self.vector_db_case = ChromaClient("case_vectors")
#
#         # 检索配置
#         self.top_k = 5
#         self.alpha = 0.6  # 混合权重（向量检索占比）
#
#     def vector_search(self, query, db_type):
#         """向量相似度检索"""
#         query_vec = self.encoder.encode(query)
#         if db_type == "law":
#             results = self.vector_db_law.query(
#                 query_embeddings=query_vec,
#                 n_results=self.top_k
#             )
#         else:
#             results = self.vector_db_case.query(
#                 query_embeddings=query_vec,
#                 n_results=self.top_k
#             )
#         return results['documents'], results['distances']
#
#     def text_search(self, keywords, db_type):
#         """基于Elasticsearch的文本检索"""
#         query = {
#             "query": {
#                 "bool": {
#                     "should": [
#                         {"match": {"content": kw}} for kw in keywords
#                     ],
#                     "minimum_should_match": 1
#                 }
#             },
#             "size": self.top_k
#         }
#         if db_type == "law":
#             index = "laws"
#         else:
#             index = "cases"
#
#         response = self.es.search(index=index, body=query)
#         return [hit["_source"] for hit in response['hits']['hits']]
#
#     def hybrid_search(self, keywords):
#         """混合检索主方法"""
#         # 构造查询语句
#         query_text = ' '.join(keywords)
#
#         # 并行执行四种检索
#         law_vec, law_vec_scores = self.vector_search(query_text, "law")
#         case_vec, case_vec_scores = self.vector_search(query_text, "case")
#         law_text = self.text_search(keywords, "law")
#         case_text = self.text_search(keywords, "case")
#
#         # 结果融合与重排序
#         def fuse_results(vec_results, text_results, scores):
#             fused = []
#             # 向量结果加权
#             for doc, score in zip(vec_results, scores):
#                 fused.append({
#                     "content": doc,
#                     "score": self.alpha * (1 - score)  # 距离转相似度
#                 })
#             # 文本结果基础分
#             for doc in text_results:
#                 fused.append({
#                     "content": doc['content'],
#                     "score": (1 - self.alpha) * doc.get('_score', 0.5) / 10
#                 })
#             # 去重后排序
#             unique_contents = {}
#             for item in fused:
#                 content = item["content"]
#                 if content not in unique_contents or item["score"] > unique_contents[content]["score"]:
#                     unique_contents[content] = item
#             return sorted(unique_contents.values(), key=lambda x: -x["score"])[:self.top_k]
#
#         # 获取最终结果
#         final_laws = fuse_results(law_vec, law_text, law_vec_scores)
#         final_cases = fuse_results(case_vec, case_text, case_vec_scores)
#
#         return {
#             "laws": self._format_results(final_laws, "law"),
#             "cases": self._format_results(final_cases, "case")
#         }
#
#     def _format_results(self, items, doc_type):
#         """格式化检索结果"""
#         formatted = []
#         for item in items:
#             if doc_type == "law":
#                 formatted.append({
#                     "text": item["content"][:500] + "...",
#                     "source": item["content"].get("law_name", "未知法规"),
#                     "article": item["content"].get("article_num", ""),
#                     "score": round(item["score"], 4)
#                 })
#             else:
#                 formatted.append({
#                     "text": item["content"][:500] + "...",
#                     "case_id": item["content"].get("case_id", ""),
#                     "court": item["content"].get("court", "未知法院"),
#                     "score": round(item["score"], 4)
#                 })
#         return formatted
#
#
# # 使用示例
# if __name__ == "__main__":
#     retriever = HybridRetriever()
#
#     # 从问题处理模块获取的关键词
#     keywords = ['租房', '押金', '退还']
#
#     # 执行混合检索
#     results = retriever.hybrid_search(keywords)
#
#     print("法律法规结果：")
#     for law in results['laws']:
#         print(f"{law['source']} 第{law['article']}条 (相关度:{law['score']}): {law['text']}")
#
#     print("\n纠纷案例结果：")
#     for case in results['cases']:
#         print(f"{case['case_id']} {case['court']} (相关度:{case['score']}): {case['text']}")
#
#
# keywords = []
def generate_legal_answer(question, case_ids, retrieved_laws):
#     """
#     生成法律问答的完整回答
#
#     参数:
#         question: 用户问题
#         case_ids: 混合检索得到的案例ID列表
#         retrieved_laws: 检索得到的法律法规列表
#     """
#
#     # 1. 根据案例ID从源数据库获取完整案例信息
    full_case_texts = []
    for case_id in case_ids:
        case_info = get_full_case_from_db(case_id)  # 从数据库获取完整案例
        full_case_texts.append(format_case(case_info))  # 格式化案例信息

    # 2. 构建上下文素材
    # context = build_context(full_case_texts, retrieved_laws)
context = {}
question = ''
keywords = []
#     # 3. 构建完整prompt

prompt = f"""
    你是一个专业的房地产法律顾问，请根据以下法律依据和参考案例，专业、准确地回答用户问题。

    【相关法律法规】
    {context['laws']}

    【参考纠纷案例】
    {context['cases']}

    【用户问题】
    {question}

    【用户问题关键词列表】
    {keywords}

    请按照以下要求回答：
    1. 首先明确指出适用的主要法律法规
    2. 然后结合参考案例进行分析
    3. 最后给出专业建议
    4. 使用法律专业术语，保持严谨性
    5. 要求返回格式要求为：
        建议：
        相关法条：
            1. 《民法典》第712条：承租人应妥善保管租赁物...
            2. 《房屋租赁司法解释》第5条：正常使用损耗不得扣除押金...
        参考案例：
            1. (2023)京02民终456号：判决房东退还押金并赔偿...
    """
#
    # 4. 调用GLM-4-Plus模型生成回答
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=15
    )

    return response
#
#
# def build_context(case_texts, law_texts):
#     """构建问答上下文"""
#     context = {
#         'laws': "\n".join([f"- {law}" for law in law_texts]),
#         'cases': "\n".join([f"- 案例摘要：{text}" for text in case_texts])
#     }
#     return context
#
#
# def format_case(case_info):
#     """格式化案例信息"""
#     return f"{case_info['title']}\n" \
#            f"案号：{case_info['case_number']}\n" \
#            f"法院：{case_info['court']}\n" \
#            f"裁判要点：{case_info['summary']}\n" \
#            f"判决结果：{case_info['result']}"
#
#
#
# import jieba
# import jieba.analyse
# from gensim import corpora, models
# import PyPDF2  # 用于PDF解析
# import docx  # 用于Word解析
#
# class ContractAnalyzer:
#     def __init__(self, stopwords_path='stopwords.txt'):
#         self.stopwords = set()
#         self.lda_model = None
#         self.dictionary = None
#         # 加载停用词表
#         with open(stopwords_path, 'r', encoding='utf-8') as f:
#             self.stopwords = set([line.strip() for line in f])
#         # 加载法律领域自定义词典
#         jieba.load_userdict('legal_terms.txt')
#
#     def parse_contract_file(self, file_path):
#         """解析上传的合同文件"""
#         text = ""
#         if file_path.endswith('.pdf'):
#             with open(file_path, 'rb') as f:
#                 reader = PyPDF2.PdfReader(f)
#                 for page in reader.pages:
#                     text += page.extract_text()
#         elif file_path.endswith('.docx'):
#             doc = docx.Document(file_path)
#             text = '\n'.join([para.text for para in doc.paragraphs])
#         else:
#             raise ValueError("Unsupported file format")
#         return self.preprocess_text(text)
#
#     def preprocess_text(self, text):
#         """文本预处理"""
#         # 去除特殊字符和数字
#         text = re.sub(r'[^\w\u4e00-\u9fa5]+', '', text)
#         # 分词处理
#         words = jieba.lcut(text)
#         # 去除停用词和单字
#         words = [word for word in words if word not in self.stopwords and len(word) > 1]
#         return words
#
#     def train_lda_model(self, documents, num_topics=5):
#         """训练LDA模型"""
#         # 创建字典和语料库
#         self.dictionary = corpora.Dictionary(documents)
#         corpus = [self.dictionary.doc2bow(doc) for doc in documents]
#         # 训练LDA模型
#         self.lda_model = models.LdaModel(
#             corpus=corpus,
#             id2word=self.dictionary,
#             num_topics=num_topics,
#             passes=10,
#             alpha='auto'
#         )
#         return self.lda_model
#
#     def extract_keywords_from_contract(self, contract_text, topn=10):
#         """从合同文本提取关键词"""
#         # 预处理文本
#         words = self.preprocess_text(contract_text)
#         # 转换为词袋表示
#         bow = self.dictionary.doc2bow(words)
#         # 获取文档主题分布
#         topic_dist = self.lda_model.get_document_topics(bow)
#         # 获取最重要的主题
#         main_topic = sorted(topic_dist, key=lambda x: x[1], reverse=True)[0][0]
#         # 获取该主题的关键词
#         topic_keywords = self.lda_model.show_topic(main_topic, topn=topn)
#         return [word for word, prob in topic_keywords]
#
#     def analyze_contract(self, file_path):
#         """完整分析流程"""
#         # 1. 解析合同文本
#         raw_text = self.parse_contract_file(file_path)
#         # 2. 预处理
#         words = self.preprocess_text(raw_text)
#         # 3. 提取关键词 (使用TF-IDF作为备用方法)
#         tfidf_keywords = jieba.analyse.extract_tags(
#             raw_text,
#             topK=15,
#             withWeight=False,
#             allowPOS=('n', 'vn', 'ns')
#         )
#         # 4. LDA主题关键词
#         if self.lda_model:
#             lda_keywords = self.extract_keywords_from_contract(raw_text)
#             keywords = list(set(tfidf_keywords + lda_keywords))[:20]  # 合并并去重
#         else:
#             keywords = tfidf_keywords
#         return {
#             'raw_text': raw_text,
#             'keywords': keywords,
#             'topic_distribution': self.lda_model.get_document_topics(
#                 self.dictionary.doc2bow(words)) if self.lda_model else None
#         }
#
#
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import difflib
# import re
#
#
# class ContractComparator:
#     def __init__(self, standard_template_path):
#         # 加载预训练语义模型
#         self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#         # 加载标准合同模板
#         self.standard_clauses = self.load_template(standard_template_path)
#         # 必备条款清单
#         self.required_clauses = [
#             "合同主体", "标的物描述", "价款及支付方式",
#             "违约责任", "争议解决", "合同终止"
#         ]
#         # 金额相关条款正则
#         self.amount_patterns = [
#             r'金额.*?([\d,，.]+元)',
#             r'总计.*?([\d,，.]+元)',
#             r'价款.*?([\d,，.]+元)'
#         ]
#
#     def load_template(self, file_path):
#         """加载标准合同模板并分条款"""
#         with open(file_path, 'r', encoding='utf-8') as f:
#             text = f.read()
#         # 按条款分割（实际可根据合同结构优化）
#         clauses = re.split(r'\n第[一二三四五六七八九十]+条\s', text)
#         return {self.get_clause_title(clause): clause for clause in clauses if clause.strip()}
#
#     def get_clause_title(self, clause_text):
#         """提取条款标题"""
#         first_line = clause_text.split('\n')[0]
#         return re.sub(r'[^\u4e00-\u9fa5]', '', first_line)[:20]
#
#     def extract_clauses(self, contract_text):
#         """从用户合同提取条款"""
#         clauses = re.split(r'\n第[一二三四五六七八九十]+条\s', contract_text)
#         return {self.get_clause_title(clause): clause for clause in clauses if clause.strip()}
#
#     def calculate_semantic_similarity(self, text1, text2):
#         """计算文本语义相似度"""
#         emb1 = self.model.encode(text1)
#         emb2 = self.model.encode(text2)
#         return cosine_similarity([emb1], [emb2])[0][0]
#
#     def detect_amount_discrepancies(self, std_text, user_text):
#         """检测金额差异"""
#         discrepancies = []
#         for pattern in self.amount_patterns:
#             std_amounts = re.findall(pattern, std_text)
#             user_amounts = re.findall(pattern, user_text)
#             if std_amounts and user_amounts:
#                 std_val = float(re.sub(r'[^\d.]', '', std_amounts[0]))
#                 user_val = float(re.sub(r'[^\d.]', '', user_amounts[0]))
#                 if abs(std_val - user_val) / std_val > 0.1:  # 偏差超过10%
#                     discrepancies.append({
#                         'type': '金额偏差',
#                         'std_value': std_amounts[0],
#                         'user_value': user_amounts[0],
#                         'deviation': f"{(user_val - std_val) / std_val * 100:.2f}%"
#                     })
#         return discrepancies
#
#     def generate_diff_report(self, std_clause, user_clause):
#         """生成文本差异报告"""
#         diff = difflib.ndiff(
#             std_clause.splitlines(keepends=True),
#             user_clause.splitlines(keepends=True)
#         )
#         changes = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]
#         return ''.join(changes[:10])  # 返回前10处差异
#
#     def compare_contracts(self, user_contract_path):
#         """完整合同比对流程"""
#         # 1. 解析用户合同
#         with open(user_contract_path, 'r', encoding='utf-8') as f:
#             user_text = f.read()
#         user_clauses = self.extract_clauses(user_text)
#
#         report = {
#             'missing_clauses': [],
#             'high_risk_changes': [],
#             'amount_discrepancies': [],
#             'clause_comparisons': []
#         }
#
#         # 2. 必备条款检查
#         for req_clause in self.required_clauses:
#             if req_clause not in user_clauses:
#                 report['missing_clauses'].append(req_clause)
#
#         # 3. 条款级比对
#         for title, std_clause in self.standard_clauses.items():
#             if title in user_clauses:
#                 similarity = self.calculate_semantic_similarity(
#                     std_clause, user_clauses[title])
#
#                 # 语义相似度低于阈值视为高风险修改
#                 if similarity < 0.6:
#                     diff = self.generate_diff_report(std_clause, user_clauses[title])
#                     report['high_risk_changes'].append({
#                         'clause': title,
#                         'similarity': f"{similarity:.2f}",
#                         'diff': diff
#                     })
#
#                 # 金额差异检测
#                 amount_issues = self.detect_amount_discrepancies(
#                     std_clause, user_clauses[title])
#                 report['amount_discrepancies'].extend(amount_issues)
#
#                 # 记录条款比对结果
#                 report['clause_comparisons'].append({
#                     'clause': title,
#                     'similarity': similarity,
#                     'status': '匹配' if similarity > 0.8 else '部分匹配'
#                 })
#             else:
#                 report['clause_comparisons'].append({
#                     'clause': title,
#                     'similarity': 0.0,
#                     'status': '缺失'
#                 })
#
#         return report
#
#
# # 使用示例
# if __name__ == "__main__":
#     comparator = ContractComparator("standard_contract_template.txt")
#     report = comparator.compare_contracts("user_contract.docx")
#
#     # 输出报告摘要
#     print(f"缺失必备条款：{report['missing_clauses']}")
#     print(f"高风险修改条款：{[item['clause'] for item in report['high_risk_changes']}")
#     print(f"金额差异：{report['amount_discrepancies']}")
#
#     import re
#     from docx import Document
#     import numpy as np
#     from sentence_transformers import SentenceTransformer
#     import pandas as pd
#     from typing import List, Dict
#
#
# class LegalDocumentProcessor:
#     def __init__(self, embedding_model_name='GanymedeNil/text2vec-large-chinese'):
#         """
#         初始化法律文档处理器
#         :param embedding_model_name: 文本向量化模型名称
#         """
#         self.embedding_model = SentenceTransformer(embedding_model_name)
#         self.clause_pattern = re.compile(r'^第[一二三四五六七八九十百零]+条\s*([^\n]*)')  # 匹配"第X条"标题
#
#     def parse_word_document(self, file_path: str) -> List[Dict]:
#         """
#         解析Word格式法律文档
#         :param file_path: Word文档路径
#         :return: 结构化条文列表
#         """
#         doc = Document(file_path)
#         clauses = []
#         current_clause = {"title": "", "content": []}
#
#         for paragraph in doc.paragraphs:
#             text = paragraph.text.strip()
#             if not text:
#                 continue
#
#             # 检查是否是条款标题
#             match = self.clause_pattern.match(text)
#             if match:
#                 if current_clause["title"]:  # 保存上一个条款
#                     current_clause["content"] = "\n".join(current_clause["content"])
#                     clauses.append(current_clause)
#
#                 # 开始新条款
#                 current_clause = {
#                     "title": text,
#                     "content": [match.group(1)] if match.group(1) else []
#                 }
#             else:
#                 current_clause["content"].append(text)
#
#         # 添加最后一个条款
#         if current_clause["title"]:
#             current_clause["content"] = "\n".join(current_clause["content"])
#             clauses.append(current_clause)
#
#         return clauses
#
#     def generate_embeddings(self, clauses: List[Dict]) -> pd.DataFrame:
#         """
#         生成法律条文向量
#         :param clauses: 结构化条文列表
#         :return: 包含向量和元数据的DataFrame
#         """
#         texts = [f"{item['title']}\n{item['content']}" for item in clauses]
#         embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
#
#         # 构建结果DataFrame
#         results = []
#         for idx, (clause, embedding) in enumerate(zip(clauses, embeddings)):
#             results.append({
#                 "clause_id": f"clause_{idx + 1}",
#                 "title": clause["title"],
#                 "content": clause["content"],
#                 "embedding": embedding,
#                 "text_length": len(clause["content"])
#             })
#
#         return pd.DataFrame(results)
#
#     def save_to_vector_db(self, df: pd.DataFrame, output_path: str):
#         """
#         将向量数据保存到本地文件
#         :param df: 包含向量的DataFrame
#         :param output_path: 输出文件路径
#         """
#         # 保存为parquet格式（保留向量数据）
#         df.to_parquet(output_path, index=False)
#         print(f"成功保存{len(df)}条法律条文向量到{output_path}")
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 1. 初始化处理器
#     processor = LegalDocumentProcessor()
#
#     # 2. 解析Word文档
#     clauses = processor.parse_word_document("civil_law.docx")
#     print(f"共解析出{len(clauses)}个法律条款")
#
#     # 3. 生成向量
#     vector_df = processor.generate_embeddings(clauses)
#
#     # 4. 保存向量数据
#     processor.save_to_vector_db(vector_df, "law_clause_vectors.parquet")
#
#     # 打印样例
#     print("\n样例条款向量化结果：")
#     sample = vector_df.iloc[0]
#     print(f"条款标题：{sample['title']}")
#     print(f"内容长度：{sample['text_length']}字符")
#     print(f"向量维度：{len(sample['embedding'])}")

caseIdList = []
caseId = ''
if caseIdList.__contains__(caseId):
    caseIdList.append(caseId)