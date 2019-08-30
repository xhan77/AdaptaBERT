import re, os, sys, pickle
from sys import argv

def raw2list(infile, dataset="conll"):
    data_list = []
    tags = []
    tokens = []
    fin = open(infile, 'r')
    nlflag = True
    for line in fin:
        line = line.strip()
        if line == '':
            if not nlflag:
                data_list.append((tokens, tags))
                tags = []
                tokens = []
                nlflag = True
        else:
            parts = line.split()
            if dataset == "conll":
                if len(parts) != 4:
                    raise NameError('conll file parse error, please check again')
                tokens.append(parts[0])
                tags.append(parts[3])
            elif dataset == "twitter":
                if len(parts) != 2:
                    raise NameError('twitter file parse error, please check again')
                tokens.append(parts[0])
                tags.append(parts[1])
            else:
                raise NameError('unknown dataset, cannot convert tags')
            nlflag = False
    fin.close()
    if dataset == "conll":
        for i in range(len(data_list)):
            for j in range(len(data_list[i][1])):
                if data_list[i][1][j][0] == 'B':
                    data_list[i][1][j] = 'B'
                elif data_list[i][1][j][0] == 'I':
                    if j == 0:
                        data_list[i][1][j] = 'B'
                    elif data_list[i][1][j - 1][0] == 'O':
                        data_list[i][1][j] = 'B'
                    else:
                        data_list[i][1][j] = 'I'
                else:
                    data_list[i][1][j] = 'O'
    return data_list


def general2list(infile, sample_size):
    import random
    random.seed(2019)

    fin = open(infile, 'r')
    line_count = 0
    for line in fin:
        line_count += 1
    fin.close()
    valid_lines = set(random.sample(range(line_count), sample_size))

    data_list = []
    fin = open(infile, 'r')
    line_count = 0
    for line in fin:
        if line_count not in valid_lines:
            line_count += 1
            continue
        line = line.strip()
        parts = line.split()
        tokens = parts
        tags = [''] * len(tokens)
        data_list.append((tokens, tags))
        line_count += 1
    fin.close()
    return data_list

if __name__ == '__main__':

    op = sys.argv[1]

    if op == "general":
        data_list = raw2list("../resources/conll_ner/eng.train", dataset="conll")
        pickle.dump(data_list, open("conll_train.pkl", 'wb'))

        data_list = raw2list("../resources/conll_ner/eng.testa", dataset="conll")
        pickle.dump(data_list, open("conll_test.pkl", 'wb'))

        data_list = raw2list("../resources/twitter_ner/train_notypes", dataset="twitter")
        pickle.dump(data_list, open("twitter_train.pkl", 'wb'))

        data_list = raw2list("../resources/twitter_ner/test_notypes", dataset="twitter")
        pickle.dump(data_list, open("twitter_test.pkl", 'wb'))

    if op == "general_twitter_text":
        data_list = general2list("../resources/twitter_general/tweets-en-2016.text", 1000000)
        pickle.dump(data_list, open("twitter_general.pkl", 'wb'))

    if op == "sep_twitter_train":
        max_seq_length = 128
        input_file = "twitter_train.pkl"
        bert_model = "bert-base-cased"
        do_lower_case = False

        from collections import defaultdict
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

        data_list = pickle.load(open(input_file, 'rb'))
        print("original number of examples:", len(data_list))
        split_map = defaultdict(list)

        for ex_index, elem in enumerate(data_list):
            tokens = elem[0]
            split_map[ex_index].append(0)

            bert_tokens = ["[CLS]"]
            for token_index, token in enumerate(tokens):
                new_tokens = tokenizer.tokenize(token)
                if len(bert_tokens) + len(new_tokens) > max_seq_length - 1: # account for the [SEP] token
                    split_map[ex_index].append(token_index)
                    bert_tokens = ["[CLS]"]
                    new_tokens = tokenizer.tokenize(token)
                bert_tokens.extend(new_tokens)
            split_map[ex_index].append(len(tokens))

        new_data_list = []
        for ex_index in split_map:
            elem = data_list[ex_index]
            tokens = elem[0]
            labels = elem[1]
            splits = split_map[ex_index]
            for i in range(len(splits) - 1):
                start = splits[i]
                end = splits[i + 1]
                new_data_list.append((tokens[start : end], labels[start : end]))

        print("updated number of examples:", len(new_data_list))
        pickle.dump(new_data_list, open("sep_twitter_train.pkl", 'wb'))

    if op == "sep_twitter_test":
        max_seq_length = 128
        input_file = "twitter_test.pkl"
        bert_model = "bert-base-cased"
        do_lower_case = False

        from collections import defaultdict
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

        data_list = pickle.load(open(input_file, 'rb'))
        print("original number of examples:", len(data_list))
        split_map = defaultdict(list)

        for ex_index, elem in enumerate(data_list):
            tokens = elem[0]
            split_map[ex_index].append(0)

            bert_tokens = ["[CLS]"]
            for token_index, token in enumerate(tokens):
                new_tokens = tokenizer.tokenize(token)
                if len(bert_tokens) + len(new_tokens) > max_seq_length - 1: # account for the [SEP] token
                    split_map[ex_index].append(token_index)
                    bert_tokens = ["[CLS]"]
                    new_tokens = tokenizer.tokenize(token)
                bert_tokens.extend(new_tokens)
            split_map[ex_index].append(len(tokens))

        new_data_list = []
        for ex_index in split_map:
            elem = data_list[ex_index]
            tokens = elem[0]
            labels = elem[1]
            splits = split_map[ex_index]
            for i in range(len(splits) - 1):
                start = splits[i]
                end = splits[i + 1]
                new_data_list.append((tokens[start : end], labels[start : end]))

        print("updated number of examples:", len(new_data_list))
        pickle.dump(new_data_list, open("sep_twitter_test.pkl", 'wb'))
