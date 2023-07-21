import argparse
import subprocess
from shutil import rmtree

import requests
from datasets import load_dataset

HUGGING_FACE_HUB_RETRY = 3
GIT_CLONE_RETRY = 3

parser = argparse.ArgumentParser(
    description='Download and save a Hugging Face dataset')
parser.add_argument('dataset_id',
                    type=str,
                    help='ID of the dataset in Hugging Face Hub')
parser.add_argument('--output-dir',
                    type=str,
                    help='Output directory of the saved dataset')

if __name__ == '__main__':
    args = parser.parse_args()
    dataset_id = args.dataset_id
    dataset_name = dataset_id.split('/')[-1]
    output_dir = args.output_dir

    dataset = None

    for attempt in range(HUGGING_FACE_HUB_RETRY):
        try:
            print(
                f'Try downloading by `load_dataset`: {attempt + 1}/{HUGGING_FACE_HUB_RETRY}'
            )
            dataset = load_dataset(dataset_id)
        except (ConnectionError, requests.exceptions.ConnectionError):
            continue
        break

    if not dataset:
        try:
            subprocess.check_output(['git', '--version'])
        except FileNotFoundError:
            print('Error: git command not found. Please install git.')
            exit(1)

        repo_url = f'https://huggingface.co/datasets/{dataset_id}'
        repo_dir = f'./{dataset_name}'
        for attempt in range(GIT_CLONE_RETRY):
            print(
                f'Try downloading by `git clone`: {attempt + 1}/{GIT_CLONE_RETRY}'
            )
            ret = subprocess.call(['git', 'clone', repo_url])
            if ret == 0:
                break
            else:
                rmtree(repo_dir, ignore_errors=True)

        if ret > 0:
            print(
                'Error: Failed to download dataset. Please retry or use a network proxy.'
            )
            exit(1)

        dataset = load_dataset(repo_dir)
        rmtree(repo_dir)

    if not output_dir:
        output_dir = f'/t9k/mnt/datasets/{dataset_id}'

    dataset.save_to_disk(output_dir)

    print(f'Dataset {dataset_id} saved to: {output_dir}')


from transformers.models.llama.tokenization_llama import LlamaTokenizer