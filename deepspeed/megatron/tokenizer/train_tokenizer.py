import argparse
import json

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Dataset file path')
parser.add_argument('save_path', help='Save path')
args = parser.parse_args()

def get_training_corpus():
    with open(args.data_path, 'r') as file:
        json_list = []
        for line in file:
            json_obj = json.loads(line)
            json_list.append(json_obj['text'])
            if len(json_list) == 100:
                yield json_list
                json_list = []
        if json_list:
            yield json_list

training_corpus = get_training_corpus()

gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer = gpt2_tokenizer.train_new_from_iterator(training_corpus, 52000)
tokenizer.save_pretrained('my-gpt-tokenizer')
