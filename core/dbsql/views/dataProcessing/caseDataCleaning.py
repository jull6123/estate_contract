import json
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time

'''
原数据集为民事判决书（json格式）-->处理为房地产纠纷案例判决书（具体分裂为：土地纠纷、房产纠纷、房屋买卖、物业纠纷（总案例数量达到10w+）
将案例向量化存储，后根据问题的关键词检索
'''

# 定义房地产纠纷的具体分类
LAND_DISPUTE = "土地纠纷"
ESTATE_DISPUTE = "房产纠纷"
HOUSE_TRANSACTION = "房屋买卖"
# HOUSE_RENT = "房屋租赁"--在房屋买卖与房产纠纷中，并无单独分类
PROPERTY_DISPUTE = "物业纠纷"

class LegalDataCleaner:
    def __init__(self, input_dir, output_dir, num_workers=4):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.stats = defaultdict(lambda: defaultdict(int))

        self.category_data = {
            LAND_DISPUTE: [],
            ESTATE_DISPUTE: [],
            HOUSE_TRANSACTION: [],
            # HOUSE_RENT: [],
            PROPERTY_DISPUTE: []
        }


    # 处理单个json文件
    def _process_single_item(self, item):
        for old_idx, ctx in item.get('ctxs', {}).items():
            # 仅保留 房地产纠纷 民事案例
            if isinstance(ctx, dict) and ctx.get('Category', {}).get('cat_1') == "房地产纠纷":
                cat_2 = ctx.get('Category', {}).get('cat_2')
                # 对房地产纠纷案例进行模块细分
                self._process_lable_item(ctx, cat_2)
        return None

    # 房地产纠纷案例-模块分类
    def _process_lable_item(self, ctx, name):
        # 土地纠纷
        if name == LAND_DISPUTE:
            self.category_data[LAND_DISPUTE].append(ctx)
        # 房产纠纷
        elif name == ESTATE_DISPUTE:
            self.category_data[ESTATE_DISPUTE].append(ctx)
        # 房屋买卖
        elif name == HOUSE_TRANSACTION:
            self.category_data[HOUSE_TRANSACTION].append(ctx)
        # 物业纠纷
        elif name == PROPERTY_DISPUTE:
            self.category_data[PROPERTY_DISPUTE].append(ctx)


    def _process_single_file(self, input_path):
        file_stats = {
            'total_items': 0,
            'processed_items': 0
        }

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)

            file_stats['total_items'] += 1
            self._process_single_item(file_data)
            file_stats['processed_items'] += 1
            return file_stats

        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            return None


    def _process_split(self, split):
        """处理单个split目录"""
        split_input_dir = self.input_dir / split
        json_files = list(split_input_dir.glob('*.json'))
        if not json_files:
            print(f"No JSON files found in {split_input_dir}")
            return

        print(f"Processing {len(json_files)} files in {split}...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for json_file in json_files:
                futures.append(executor.submit(
                    self._process_single_file,
                    json_file
                ))

            for future in futures:
                result = future.result()
                if result:
                    for k, v in result.items():
                        self.stats[split][k] += v

        elapsed = time.time() - start_time
        print(f"Finished {split} in {elapsed:.2f} seconds")
        print(f"  Processed: {self.stats[split]['processed_items']}")

    def _save_category_data(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for category, data in self.category_data.items():
            output_file = self.output_dir / f"{category}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(data)} items to {output_file}")

    def run(self):
        print(f"Starting data processing from {self.input_dir}")
        print(f"Results will be saved to {self.output_dir}")

        # 处理所有数据
        for split in ['train', 'dev', 'test']:
            self._process_split(split)

        # 保存分类数据
        self._save_category_data()

        # 保存统计信息
        stats_path = self.output_dir / 'processing_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.stats), f, indent=2)

        total_processed = sum(s['processed_items'] for s in self.stats.values())
        print(f"\nTotal processed items: {total_processed}")

        for category, data in self.category_data.items():
            print(f"{category}: {len(data)} items")
        print(f"Detailed stats saved to {stats_path}")


def data_cleaning():
    input_dir = 'D:/program_files/estate_data/case_data/wenshu_ms_dataset'
    output_dir = 'D:/program_files/estate_data/case_data/estate_case_dataset'
    workers = 4

    print("房地产纠纷数据分类工具")
    print("=" * 50)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("分类类型:")
    print(f"1. {LAND_DISPUTE}")
    print(f"2. {ESTATE_DISPUTE}")
    print(f"3. {HOUSE_TRANSACTION}")
    # print(f"4. {HOUSE_RENT}")
    print(f"4. {PROPERTY_DISPUTE}")
    print("=" * 50)

    cleaner = LegalDataCleaner(input_dir, output_dir, workers)
    cleaner.run()
