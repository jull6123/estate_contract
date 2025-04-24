import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import requests

'''
针对合同示范文本的爬虫：关键词为”房屋、物业“--->文本去重--->地区级与国家级分别存储
存储sys_contract合同表中

当用户由相应的合同询问请求时，查找对应合同进行分析与向量化处理
'''
class ContractTemplateCrawler:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN'
        }
        self.cookies = {
            'samr': 'isopen'
        }
        self.items_gol = []
        self.items_loc = []


    def fetch_page(self, key: str, loc: bool, page: int) -> Dict:
        params = {
            'key': key,
            'loc': 'true' if loc else 'false',
            'p': page
        }

        response = requests.get(
            self.base_url,
            params=params,
            headers=self.headers,
            cookies=self.cookies,
            timeout=10
        )
        response.raise_for_status()
        if response.status_code == 200 and response.text.strip():
            return response.json()
        return None


    def get_total_pages(self, key: str, loc: bool) -> int:
        first_page = self.fetch_page(key, loc, 1)
        if first_page:
            return first_page['TotalPage']
        return 0


    def fetch_all_pages(self, key: str, loc: bool, max_workers: int = 4) -> List[Dict]:
        total_pages = self.get_total_pages(key, loc)
        if total_pages == 0:
            print("No data available or failed to get total pages")
            return []
        print(f"Found {total_pages} pages to fetch for key='{key}', loc={loc}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for page in range(1, total_pages + 1):
                futures.append(executor.submit(self.fetch_page, key, loc, page))
                # 随即延迟
                time.sleep(3)

            for i, future in enumerate(futures, 1):
                result = future.result()
                if result:
                    if loc:
                        self.items_loc.extend(result['Data'])
                    else:
                        self.items_gol.extend(result['Data'])
                    print(f"Fetched page {i}/{total_pages} - got {len(result['Data'])} contracts")
                time.sleep(2)


    def remove_duplicates_by_id(self, items: List[Dict]) -> List[Dict]:
        seen_ids = set()
        unique_items = []

        for item in items:
            item_id = item.get('Id')
            if item_id is None:
                continue
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_items.append(item)
        return unique_items

    def save_contract_data(self):
        try:
            base_dir = Path('D:/program_files/estate_data/contranct_data')
            output_file = base_dir / 'contract_items.json'
            os.makedirs(base_dir, exist_ok=True)

            temp_file = output_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(
                    self.items_loc + self.items_gol,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    default=str
                )

            if os.path.exists(output_file):
                os.replace(temp_file, output_file)
            else:
                os.rename(temp_file, output_file)

            print(f"合同数据已成功保存至: {output_file}")
            return str(output_file)
        except json.JSONEncodeError as e:
            raise IOError(f"JSON序列化失败: {str(e)}")
        except OSError as e:
            raise IOError(f"文件操作失败: {str(e)}")
        except Exception as e:
            raise IOError(f"保存合同数据时发生意外错误: {str(e)}")


    def crawl(self) -> None:
        keywords = [
            "房屋",
            "物业"
        ]
        for key in keywords:
            start_time = time.time()
            self.fetch_all_pages(key, True)
            self.fetch_all_pages(key, False)
            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f} seconds. Total items: {len(self.items_loc) + len(self.items_gol)}")
            time.sleep(3)

        self.items_loc = self.remove_duplicates_by_id(self.items_loc)
        self.items_gol = self.remove_duplicates_by_id(self.items_gol)
        # json文件存储，便于合同信息的数据表同步
        self.save_contract_data()

        print(f"\n\n地方级-{len(self.items_loc)}")
        for item in self.items_loc:
            print(item)

        print(f"\n\n国家级-{len(self.items_gol)}")
        for item in self.items_gol:
            print(item)


class ContractDowmloadWordCrawler:
    def __init__(self, download_dir_loc: str, download_dir_gol: str):
        self.download_dir_loc = Path(download_dir_loc)
        self.download_dir_loc.mkdir(parents=True, exist_ok=True)
        self.download_dir_gol = Path(download_dir_gol)
        self.download_dir_gol.mkdir(parents=True, exist_ok=True)

        # 配置请求头和会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Referer': 'https://htsfwb.samr.gov.cn/',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh',
        })
        self.session.cookies.update({'samr': 'isopen'})


    def download_word(self, loc: bool, item_id: str, title: str = None) -> bool:
        download_url = f"https://htsfwb.samr.gov.cn/api/File/DownTemplate?id={item_id}&type=1"

        try:
            response = self.session.get(download_url, timeout=(10, 30))

            if response.status_code == 200:
                filename = f"{title}.docx"
                if loc:
                    filepath = self.download_dir_loc / filename
                else:
                    filepath = self.download_dir_gol / filename

                # 写入文件
                with open(filepath, 'wb') as f:
                    f.write(response.content)

                print(f"成功下载: {filename}")
                return True
            else:
                print(f"下载失败[{response.status_code}]: {item_id}")
                return False
        except Exception as e:
            print(f"下载出错[{item_id}]: {str(e)}")
            return False


    def batch_download(self, items: list, loc: bool, delay: tuple = (1, 3)) -> None:
        print(f"开始批量下载 {len(items)} 个文档...")
        success_count = 0
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict) or 'Id' not in item:
                print(f"跳过无效item: {item}")
                continue
            print(f"正在下载第 {i}/{len(items)} 个: {item.get('Title', item['Id'])}")

            if self.download_word(loc, item['Id'], item.get('Title')):
                success_count += 1

            if i < len(items):
                time.sleep(random.uniform(*delay))

        print(f"下载完成! 成功 {success_count}/{len(items)} 个文档")


def contract_download():
    BASE_URL = "https://htsfwb.samr.gov.cn/api/content/SearchTemplates"
    template_crawler = ContractTemplateCrawler(BASE_URL)
    template_crawler.crawl()

    downloader = ContractDowmloadWordCrawler('D:\program_files\estate_data\contranct_data\\contract_loc',
                                             'D:\program_files\estate_data\contranct_data\\contract_gol')
    downloader.batch_download(template_crawler.items_loc, True)
    downloader.batch_download(template_crawler.items_gol, False)

if __name__ == "__main__":
    contract_download()