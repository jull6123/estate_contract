'''
整体采用zhipuapi的向量化与相似度检索等
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
client = ZhipuAI(api_key="eee3ab03c49ebe869b2bd6bd49c29c37.n6SyFHxFLgWrgjlG")

# 生成向量
def generate_vector(text):
    response = client.embeddings.create(
        model="embedding-2",  # 填写需要调用的模型名称
        input=text,
    )
    vector = response.data[0].embedding
    return np.array(vector, dtype=np.float32)
1. 读取目录下的所有json文件，json文件格式为[{"Case":"","CaseId":""},{},{}]
2. 对应的判决书信息拼装（Category[cat_2]具体分类+Keywords[]格式需要转化+\n+CaseId(便于后续根据caseId得到全部的判决书相关信息）+\n+Case案例名+\n+CaseRecord案件记录+\n+CaseAccusation案由)--案例相关文本
3. 判决书信息向量化存储知识库中
4. 给出问题query，向量化后进行相似度检索得到topK个案例信息
5. 根据topK个案例信息的CaseId得到全部的案例信息（遍历json文件得到）并展示
'''

import os
import json
import numpy as np
from typing import List, Dict
from zhipuai import ZhipuAI
from sklearn.metrics.pairwise import cosine_similarity

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
client = ZhipuAI(api_key="31b49f664ea5410986eb1f76f9ff85be.jBD9TuKDwrvVAIxD")


class CaseSearchSystem:
    def __init__(self):
        self.case_data = {}  # 存储所有案例原始数据 {case_id: case_info}
        self.case_texts = []  # 存储案例文本
        self.case_vectors = []  # 存储案例向量
        self.case_ids = []  # 存储对应的case_id

    # 生成向量
    def generate_vector(self, text: str) -> np.ndarray:
        response = client.embeddings.create(
            model="embedding-2",
            input=text,
        )
        vector = response.data[0].embedding
        return np.array(vector, dtype=np.float32).reshape(1, -1)

    # 读取目录下的所有JSON文件
    def load_json_files(self, directory: str):
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    cases = json.load(f)
                    for case in cases:
                        self._process_case(case)

    # 处理单个案例数据
    def _process_case(self, case: Dict):
        case_id = case.get('CaseId', '')
        if not case_id or case_id in self.case_data:
            return

        # 拼装案例文本
        case_text = self._construct_case_text(case)

        # 存储数据
        self.case_data[case_id] = case
        self.case_texts.append(case_text)
        self.case_ids.append(case_id)

    # 构造案例文本
    def _construct_case_text(self, case: Dict) -> str:
        category = case.get('Category', {}).get('cat_2', '')
        keywords = ', '.join(case.get('Keywords', []))
        case_name = case.get('Case', '')
        case_record = case.get('CaseRecord', '')
        case_accusation = case.get('JudgeAccusation', '')

        return (
            f"分类: {category}\n"
            f"关键词: {keywords}\n"
            f"案例ID: {case.get('CaseId', '')}\n"
            f"案例名: {case_name}\n"
            f"案件记录: {case_record}\n"
            f"案由: {case_accusation}"
        )

    # 向量化所有案例文本
    def vectorize_cases(self):
        print("开始向量化案例文本...")
        for text in self.case_texts:
            vector = self.generate_vector(text)
            self.case_vectors.append(vector)
        print(f"已完成 {len(self.case_vectors)} 个案例的向量化")

    # 相似度检索
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        # 向量化查询
        query_vector = self.generate_vector(query)

        # 计算相似度
        similarities = []
        for case_vector in self.case_vectors:
            sim = cosine_similarity(query_vector, case_vector)[0][0]
            similarities.append(sim)

        # 获取top_k结果
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # 返回完整案例信息
        results = []
        for idx in top_indices:
            case_id = self.case_ids[idx]
            results.append({
                'case_id': case_id,
                'similarity': similarities[idx],
                'text': self.case_texts[idx],
                'full_info': self.case_data[case_id]
            })

        return results

    # 展示结果
    def display_results(self, results: List[Dict]):
        for i, result in enumerate(results, 1):
            print(f"\n=== 结果 {i} ===")
            print(f"相似度: {result['similarity']:.4f}")
            print(f"案例ID: {result['case_id']}")
            print("\n摘要信息:")
            print(result['text'])
            print("\n完整信息:")
            print(json.dumps(result['full_info'], ensure_ascii=False, indent=2))



if __name__ == "__main__":
    system = CaseSearchSystem()

    # 1. 加载JSON文件
    json_directory = "D:\program_files\estate_data\case_data\estate_case_dataset\\test"  # 替换为你的JSON文件目录
    system.load_json_files(json_directory)

    # 2. 向量化案例
    system.vectorize_cases()

    # 3. 查询示例
    query = "关于交通事故赔偿的案例"
    print(f"\n查询: {query}")
    results = system.search(query, top_k=3)

    # 4. 展示结果
    system.display_results(results)