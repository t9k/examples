import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('language',
                    help='The language code for the Wikipedia dump')
args = parser.parse_args()

language = args.language

url = (f'https://dumps.wikimedia.org/{language}wiki/latest/'
       f'{language}wiki-latest-pages-articles.xml.bz2')
output_dir = f'wiki-{language}'
output_file = f'{language}wiki-latest-pages-articles.xml.bz2'

# 下载 Wikipedia dump 文件
os.system(f'wget {url}')

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 抽取文本
os.system(
    ('python -m wikiextractor.WikiExtractor '
     f'{language}wiki-latest-pages-articles.xml.bz2 --json -o {output_dir}'))

# 合并文件
with open(os.path.join(output_dir, 'all'), 'w') as outfile:
    for root, _, files in os.walk(output_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as infile:
                for line in infile:
                    outfile.write(line)

# 删除 Wikipedia dump 文件
os.remove(output_file)
