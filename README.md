# estate_contract

##1. 文件目录作用解析：
###项目根目录文件
- **manage.py**：这是Django项目的命令行工具，用于执行与项目管理和开发相关的任务，如运行服务器、运行测试、创建数据库模式等
###项目配置目录--core
- **[&init.py&]**：这是一个空文件，用于告诉Python解释器该目录是一个Python包
- **settings.py**：包含所有的项目配置，如数据库配置、应用、中间件、模板、缓存等
- **urls.py**：定义项目的URL路由，将用户请求的URL映射到相应的视图函数或类---api层
- **asgi.py**：用于配置ASGI应用，支持异步编程
- **wsgi.py**：用于配置WSGI应用，使Web服务器能够与Django应用进行交互
###数据库
- **views/admin.py**：相当于serviseImpl
- **modules.py**：表定义


#2. 数据爬虫获取
### 前期准备：模块分类（根据案件分类）
- 土地纠纷
- 房产纠纷
- 房屋买卖
- 物业纠纷

##1. 法律法规
- [cfa532/CHLAWS|法律法规数据集](https://hf-mirror.com/datasets/cfa532/CHLAWS)
  - 采用现有数据集：`pip install datasets`
  - 问题：地址无法下载
- [Pile-FreeLaw|法律文本数据集|自然语言处理数据集](https://opendatalab.org.cn/OpenDataLab/Pile-FreeLaw)
  - 英文版
- [威科先行](https://law.wkinfo.com.cn/)
  - 裁判文书筛选条件（1千数据）
    - 房地产行业 民事-合同法 现行有效、
    - 下载权限问题


**方案一：数据合并清洗**
- [CnOpenData中国法律法规数据](https://www.cnopendata.com/data/law.html)
  - 数据量较少（可能数据不全）
- [法律法规-law-book](https://aistudio.baidu.com/datasetdetail/300709)
- [20240501法律法规全文-法律法规](https://aistudio.baidu.com/datasetdetail/271155)

##2. 裁判文书网--案件
- [裁判文书网](https://wenshu.court.gov.cn/)
  - 爬虫有风险，存在验证码验证，爬虫代码问题
  
- [威科先行](https://law.wkinfo.com.cn/)
  - 裁判文书筛选条件（40万数据）--下载权限问题
    - 案件类型：民事
    - 案由：房地产开发经营合同纠纷 房屋买卖合同纠纷 房屋租赁合同纠纷 物业服务合同纠纷（待定）
    - 审判日期：近3年
    - 文书类型：判决书 调解书 裁定书
    
     | 文书类型 | 适用场景 | 案例价值 |
     | --- | --- | --- |
     | 判决书 | 合同纠纷终局裁决 | 供完整的案情分析、法律适用和判决理由 |
     | 调解书 | 双方达成和解 | 展示行业常见调解方案和条款设计 |
     | 裁定书 | 程序性争议（如管辖权） | 解决合同执行中的程序问题 |
  
****- [中国民事类案检索数据集C3RD|法律检索数据集|人工智能数据集](https://aistudio.baidu.com/datasetdetail/205651)****
   - 所有民事案件--需要数据筛选  train训练集 60 、 dev开发集 20 、 test测试集 20
   - 数据清洗只得到与房地产相关的案例

##3. 合同范本
- [国家市场监督管理总局-合同示范文本库](https://htsfwb.samr.gov.cn/)

##4. 处理流程
用户输入
    │
    ▼
场景分析模块 → 智谱API分析场景类型
    │
    ├──▶ 案例检索模块 → 返回相似判决案例
    │
    ├──▶ 法规检索模块 → 返回相关法律法规
    │
    ├──▶ 合同分析模块 → 分析合同问题点
    │
    └──▶ 建议生成模块 → 生成综合建议
