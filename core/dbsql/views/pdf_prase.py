import requests
import fitz
from io import BytesIO

# 1. 获取PDF数据
url = 'https://htsfwb.samr.gov.cn/api/File/DownTemplate?id=43e369b7-dce2-4da2-aad8-89fc15e54220&type=2'
response = requests.get(url)
pdf_data = response.content  # 获取二进制数据

# print(pdf_data)

# 2. 验证PDF有效性
def is_valid_pdf(pdf_bytes):
    return pdf_bytes[:8] == b'%PDF-1.7'  # 检查文件头

if not is_valid_pdf(pdf_data):
    raise ValueError("不是有效的PDF文件")

# 3. 解析内容
doc = fitz.open(stream=pdf_data, filetype="pdf")

# print(doc)

# # 4. 结构化提取（示例：提取所有加粗文本）
bold_texts = []
for page in doc:
    for block in page.get_text("dict")["blocks"]:
        # print(block)
        if "lines" in block:
            # print()
            for line in block["lines"]:
                for span in line["spans"]:
                    print(span["text"])
                    # if span["text"] & 2**4:  # 加粗标志
        #                 bold_texts.append(span["text"])

# print("文档中的加粗内容：", bold_texts)