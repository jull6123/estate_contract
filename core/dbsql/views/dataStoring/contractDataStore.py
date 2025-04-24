'''
合同数据入表
cd core
python manage.py makemigrations dbsql
python manage.py migrate

问答需要时，
1. 直接得到地址下载文档
2. 分析文档内容与提供的合同内容对比分析

# {
#     'Id': 'f214fd21-8858-4ce7-8491-00fd0bf7a191',
#     'Title': '南京市房屋租赁合同（试行）（居间服务版）',
#     'Brief': '甲方（出租人）向乙方（承租人）出租房屋，丙方（经纪机构）提供中介服务，约定租金、居间服务费用等内容。',
#     'Tags': None,
#     'Department': '南京市工商局（市场监管局）南京市住房保障和房产局',
#     'IsPublished': True,
#     'ModifiedOn': '2022-05-13T09:34:53.003',
#     'PublishedOn': '2017',
#     'IsLocal': True,
#     'Region': '江苏',  //为空或none时为gol级别
#     'Type': 3,
#     'IsRecent': False
# }
'''
import json

from django.utils import timezone
from datetime import datetime
import uuid
from  core.dbsql.models import sysContract

def get_contarct(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        all_items = json.load(f)
    return all_items

# 标准合同范本指标数据存储入表
def save_contracts(all_items):
    contracts_to_create = []

    for item in all_items:
        # 转换时间格式
        modified_on = datetime.strptime(
            item['ModifiedOn'],
            '%Y-%m-%dT%H:%M:%S.%f'
        ) if 'ModifiedOn' in item else timezone.now()

        published_on = item.get('PublishedOn', '')
        contract = sysContract(
            Id=uuid.UUID(item['Id']),
            Title=item['Title'],
            Brief=item['Brief'],
            Tags=item.get('Tags'),
            Department=item['Department'],
            IsPublished=item.get('IsPublished', True),
            ModifiedOn=modified_on,
            PublishedOn=published_on,
            IsLocal=item.get('IsLocal', True),
            Region=item.get('Region'),
            Type=item['Type'],
            IsRecent=item.get('IsRecent', False)
        )
        contracts_to_create.append(contract)

    # 使用bulk_create批量创建，提高效率
    sysContract.objects.bulk_create(contracts_to_create)

    print(f"成功入库 {len(contracts_to_create)} 条合同数据")


if __name__ == "__main__":
    # 1. 加载JSON文件
    json_directory = "D:\program_files\estate_data\case_data\estate_case_dataset\\test"  # 替换为你的JSON文件目录
    items = get_contarct(json_directory)

    # 2. 数据入表
    save_contracts(items)