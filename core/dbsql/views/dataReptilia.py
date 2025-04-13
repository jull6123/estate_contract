# python爬虫
# 法律法规数据集的下载
# from datasets import load_dataset
#
# # 1. 加载数据
# ds = load_dataset("cfa532/CHLAWS", split="train[:10]")  # 只加载前1000条
#
# print(ds)  # 输出数据集划分（train/test等）
# print(ds["train"][0])  # 查看第一条训练数据

# 2. 数据清洗
# ds = ds.filter(lambda x: len(x["answer"]) > 50)  # 过滤短答案

# 3. 转换为迭代器
# for example in ds:
#     print(f"Q: {example['question']}")
#     print(f"A: {example['answer'][:50]}...")
#     break

# pip install - i https://pypi.tuna.tsinghua.edu.cn/ simple selenium
import json

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import time

# 初始化selenium
# url = 'https://wenshu.court.gov.cn/website/wenshu/181029CR4M5A62CH/index.html?'
# # url = 'https://wenshu.court.gov.cn/website/wenshu/181217BMTKHNT2W0/index.html?pageId=74d222489a442fa2cf24578bc04e7387&s8=03'
#
#
# option = webdriver.ChromeOptions()
# option.add_argument('--start-maximized')
# option.add_experimental_option('excludeSwitches', ['enable-automation'])
#
# # 'profile.default_content_settings.popups': 0  ==  禁用弹出窗口
# # 'download.default_directory': 'D:\Desktop\wenshu'  == 设置默认下载路径
# # 'profile.default_content_setting_values.automatic_downloads': 1 == 并设置自动下载的选项
# prefs = {'profile.default_content_settings.popups': 0,
# 		 'download.default_directory': 'D:\program_files\estate_data\case_data',
#          'profile.default_content_setting_values.automatic_downloads': 1}
# option.add_experimental_option('prefs', prefs)
#
# driver = webdriver.Chrome(options=option)
# # 设置打开的浏览器窗口最大化
# driver.maximize_window()
# driver.set_page_load_timeout(30)
# driver.get(url)
#
# # 转到登录界面自动输入手机号密码进行登录
# driver.find_element(By.XPATH, '//*[@id="loginLi"]/a').click()
# text = driver.page_source
# time.sleep(10)  # 等待页面渲染
# # 进入iframe框
# iframe = driver.find_elements(By.TAG_NAME, 'iframe')[0]
# driver.switch_to.frame(iframe)
#
# # 下面的‘手机号’‘密码’输入自己中国裁判文书网注册的真实手机号密码
# username = driver.find_element(By.XPATH, '//*[@id="root"]/div/form/div/div[1]/div/div/div/input')
# username.send_keys('19857174940')
#
# time.sleep(3)
#
# username = driver.find_element(By.XPATH, '//*[@id="root"]/div/form/div/div[2]/div/div/div/input')
# username.send_keys('Zyx1540@#')
#
# time.sleep(3)
#
# code = driver.find_element(By.XPATH, '//*[@id="root"]/div/form/div/div[3]/div/div/div/input')
# code.send_keys('验证码')
#
# time.sleep(2)
# driver.find_element(By.XPATH, '//*[@id="root"]/div/form/div/div[4]/span').click()
# time.sleep(3)
#
# # 必须加上表单退出，否者就是死元素无法定位
# driver.switch_to.default_content()
#
# # 这行代码的作用就相当于你手动点了一下‘刑事案件’那个按钮
# # 要下载民事案件就把下一行代码里的刑事案件改成‘民事案件’，以此类推
# driver.find_element(By.LINK_TEXT, '民事案件').click()
#
# time.sleep(10)
# # testHtml(driver.page_source)
#
# _lastWindow = driver.window_handles[-1]
# driver.switch_to.window(_lastWindow)
#
# # 选择案件批量下载
#
# # 这行代码的作用就相当于你手动点了一下‘法院层级’那个按钮
# # driver.find_element(By.LINK_TEXT, '法院层级').click()
#
# # 按照裁判日期排序显示最新600条
# # driver.find_element(By.LINK_TEXT, '裁判日期').click()
# # 按照裁判日期从前到后显示最老600条
# # driver.find_element(By.LINK_TEXT, '裁判日期').click()
#
# # 按照审判程序排序显示最新600条
# # driver.find_element(By.LINK_TEXT, '审判程序').click()
# # 按照审判程序从前到后显示最老600条
# # driver.find_element(By.LINK_TEXT, '审判程序').click()
# #
# # # 按照法院层级、地域及法院进行检索
# # driver.find_element(By.LINK_TEXT, '高级法院(146202)').click()
# # time.sleep(3)
# # driver.find_element(By.LINK_TEXT, '上海市(1188)').click()
#
# # 在搜索框输入关键词进行高级检索
# # 定位到搜索框
# keyword = driver.find_element(By.XPATH, '//*[@id="_view_1545034775000"]/div/div[1]/div[2]/input')
# time.sleep(3)
# # 输入高级检索关键词，例如 工程纠纷
# keyword.send_keys('房屋租赁')
# time.sleep(3)
# # 点击搜索按钮
# driver.find_element(By.XPATH, '//*[@id="_view_1545034775000"]/div/div[1]/div[3]').click()
#
# # 将每页文件数设置为最大,15条
# page_size_box = Select(driver.find_element(By.XPATH, '//*[@id="_view_1545184311000"]/div[8]/div/select'))
# page_size_box.select_by_visible_text('15')
#
# def test_exceptions(xpath):
#     try:
#         driver.find_element(By.XPATH, xpath)
#         return True
#     except:
#         return False
#
# page = 1
# # 最多显示600条文件,也就是40页
# while page <= 40:
#     time.sleep(5+page/10)
#     for i in range(15):
#         time.sleep(5+i/10)
#         event_xpath = '//*[@id="_view_1545184311000"]/div[' + str(i+3) + ']/div[6]/div/a[2]'
#         if test_exceptions(event_xpath) == True:
#             driver.find_element(By.XPATH, event_xpath).click()
#         else:
#             event_xpath = '//*[@id="_view_1545184311000"]/div[' + str(i+3) + ']/div[5]/div/a[2]'
#             if test_exceptions(event_xpath) == True:
#                 driver.find_element(By.XPATH, event_xpath).click()
#
#     # 下一页按钮,不能用Xpath定位,因为“下一页”按钮位置不固定
#     # driver.find_element(By.LINK_TEXT, '下一页').click()
#     time.sleep(5)
#     driver.find_element(By.LINK_TEXT, '下一页').click()
#     # 必须加上表单退出，否者就是死元素无法定位
#     driver.switch_to.default_content()
#     page += 1
#
# # 关闭整个浏览器窗口并终止与浏览器的会话
# driver.quit()

# import fitz
from io import BytesIO
import requests
# from bs4 import BeautifulSoup

def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

#     soup = BeautifulSoup(html, 'html.parser')
#     titles = soup.find_all('h2', class_='title')
#     for title in titles:
#         print(title.get_text())
# def parse_data(html):


def safe_fetch_api(url):
    try:
        # 添加请求头和超时
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            raise ValueError(f"API返回异常状态码: {response.status_code}")

        data = response.json()

        if not isinstance(data.get("Data"), list):
            raise ValueError("API返回数据格式异常")

        return {
            "items": [
                {
                    "id": item.get("Id", ""),
                    "title": item.get("Title", ""),
                    # 添加其他需要的字段
                }
                for item in data["Data"]
            ],
            "pagination": {
                "total": data.get("TotalCount", 0),
                "total_page": data.get("TotalPage", 1),
                "current_page": data.get("Page", 1),
                "page_size": data.get("PageSize", 10)
            }
        }

    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {str(e)}")
    except json.JSONDecodeError:
        print("响应不是有效的JSON格式")
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
    return None
    # url = 'https://htsfwb.samr.gov.cn/api/File/DownTemplate?id=43e369b7-dce2-4da2-aad8-89fc15e54220&type=2'
    # html = requests.get(url)
    # print(html.content)
    # print(html.apparent_encoding)
    # print(html.encoding)
    # MyHtml = html.content.decode('iso-8859-1')
    # print(MyHtml)

    #
    #
    # url = 'https://htsfwb.samr.gov.cn/api/File/DownTemplate?id=43e369b7-dce2-4da2-aad8-89fc15e54220&type=2'
    #
    # try:
    #     # 1. 获取文件
    #     response = requests.get(url, timeout=10)
    #     response.raise_for_status()  # 检查HTTP错误
    #
    #     # 2. 判断内容类型
    #     content_type = response.headers.get('Content-Type', '')
    #
    #     if 'pdf' in content_type.lower():
    #         # 处理PDF
    #         pdf_data = BytesIO(response.content)
    #         doc = fitz.open(pdf_data)
    #         text = "\n".join([page.get_text() for page in doc])
    #         print("PDF内容提取：", text[:1000])  # 打印前1000字
    #
    #     elif 'json' in content_type.lower():
    #         # 处理JSON
    #         data = response.json()
    #         print("JSON数据：", data)
    #
    #     else:
    #         # 处理文本
    #         text = response.content.decode('utf-8', errors='ignore')
    #         print("文本内容：", text[:1000])
    #
    # except Exception as e:
    #     print(f"处理失败: {str(e)}")

if __name__ == "__main__":

    url = 'https://htsfwb.samr.gov.cn/api/content/SearchTemplates?s=1&key=%E6%88%BF%E5%B1%8B&&loc=false&p=1'
    # html = fetch_data(url)
    html = safe_fetch_api(url)
    print(html)
# 合同示范文本word的爬虫下载
# import time
# from bs4 import BeautifulSoup #4.10.0
# from selenium import webdriver
# from xlrd import open_workbook #1.2.0
# from xlutils.copy import copy #2.0.0
#
# url_search = 'https://htsfwb.samr.gov.cn/'  # 待爬取网页网址
# # 调用selenium工具
# options = webdriver.ChromeOptions()
# options.add_experimental_option('excludeSwitches', ['enable-automation'])
#
# driver = webdriver.Chrome(executable_path=r"xxxx", options=options)  # 这里配置你自己的webdriver路径
#
# driver.get(url_search)
