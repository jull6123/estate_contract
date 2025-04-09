# estate_contract

#1. 项目创建
##1. 准备工作
1. github的dns---`ping github.com`得到ip---在hosts文件中补充 
2. github创建新仓库 estate_contract
3. 下载至本地
   1. **D:** --> 切换至D盘
   2. **git clone http地址**
   3. pycharm控制台命令行创建django结构：
      1. ```pip install django ```
      2. ```django-admin startproject core ```
    
##2. 文件目录作用解析：
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
         