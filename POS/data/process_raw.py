# code partly adapted from the FEMA project (Yi and Eisenstein, 2016)

import re, os, sys, pickle
from sys import argv

emetag2ptbtag = {
    "," : ",",
    "." : ".",
    "\'" : "\'\'",
    "\"" : "\'\'",
    "$" : "PRP$",
    "ADJ" : "JJ",
    "ADJR" : "JJR",
    "ADJS" : "JJS",
    "ADV" : "RB",
    "ADVR" : "RBR",
    "ADVS" : "RBS",
    "ALSO" : "RB",
    "BAG" : "VBG",
    "BE" : "VB",
    "BED" : "VBD",
    "BEI" : "VB",
    "BEN" : "VBN",
    "BEP" : "VBZ",
    "C" : "IN",
    "CONJ" : "CC",
    "D" : "DT",
    "DAG" : "VBG",
    "DAN" : "VBN",
    "DO" : "VB",
    "DOD" : "VBD",
    "DOI" : "VB",
    "DON" : "VBN",
    "DOP" : "VBP",
    "ELSE" : "RB",
    "EX" : "EX",
    "FOR" : "IN",
    "FP" : "CC",
    "FW" : "FW",
    "HAG" : "VBG",
    "HAN" : "VBN",
    "HV" : "VB",
    "HVD" : "VBD",
    "HVI" : "VB",
    "HVN" : "VBN",
    "HVP" : "VBP",
    "INTJ" : "UH",
    "MD" : "MD",
    "N" : "NN",
    "N$" : "NN",
    "NEG" : "RB",
    "NPR" : "NNP",
    "NPR$" : "NNP",
    "NPRS" : "NNPS",
    "NPRS$" : "NNPS",
    "NS" : "NNS",
    "NS$" : "NNS",
    "NUM" : "CD",
    "NUM$" : "CD",
    "ONE" : "PRP",
    "ONES" : "PRP",
    "ONE$" : "PRP$",
    "OTHER" : "PRP",
    "OTHER$" : "PRP",
    "OTHERS$" : "PRP",
    "OTHERS" : "PRP",
    "P" : "IN",
    "PRO" : "PRP",
    "PRO$" : "PRP$",
    "Q" : "JJ",
    "QS" : "RBS",
    "QR" : "RBR",
    "RP" : "RB",
    "SUCH" : "RB",
    "TO" : "TO",
    "VAG" : "VBG",
    "VAN" : "VBN",
    "VB" : "VB",
    "VBD" : "VBD",
    "VBI" : "VB",
    "VBN" : "VBN",
    "VBP" : "VBP",
    "WADV" : "WRB",
    "WARD" : "VB",
    "WD" : "WDT",
    "WPRO" : "WP",
    "WPRO$" : "WP$",
    "WQ" : "IN",
    "X" : "#", # should be "X"? but "X" is not in the original PTB tag set
}

PTB2coarse = {
    'NNPS': 'noun',
    'TO': 'infinitival to',
    'RB': 'adverb',
    'RP': 'adverb',
    'NNS': 'noun',
    'WP': 'wh-words',
    '(': 'IGNORE',
    'WP$': 'wh-words',
    'POS': 'IGNORE',
    'VBZ': 'verb',
    'VBN': 'verb',
    'SYM': 'IGNORE',
    '\'\'': 'IGNORE',
    'EX': 'existential there',
    'UH': 'interjection',
    '.': 'IGNORE',
    'PRP$': 'pronoun',
    '``': 'IGNORE',
    'IN': 'complementizer/preposition',
    ')': 'IGNORE',
    'MD': 'modal',
    'VBD': 'verb',
    'RBR': 'adverb',
    'VB': 'verb',
    ':': 'IGNORE',
    '#': 'IGNORE',
    'JJR': 'adjective',
    'RBS': 'adverb',
    'WRB': 'wh-words',
    'NNP': 'noun',
    'LS': 'IGNORE',
    'NN': 'noun',
    'VBG': 'verb',
    'VBP': 'verb',
    ',': 'IGNORE',
    'CD': 'cardinal number',
    'JJ': 'adjective',
    'JJS': 'adjective',
    'DT': 'determiner',
    'PRP': 'pronoun',
    'FW': 'foreign word',
    '$': 'IGNORE',
    'WDT': 'wh-words',
    'PDT': 'determiner',
    'CC': 'conjunction'
}

coarse2index = {
    'IGNORE': -1,
    'adjective': 0,
    'adverb': 1,
    'cardinal number': 2,
    'complementizer/preposition': 3,
    'conjunction': 4,
    'determiner': 5,
    'existential there': 6,
    'foreign word': 7,
    'interjection': 8,
    'infinitival to': 9,
    'modal': 10,
    'noun': 11,
    'pronoun': 12,
    'verb': 13,
    'wh-words': 14
}

def conll2list(infile, dataset="PTB"):
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
            if len(parts) < 2:
                raise NameError('conll file parse error, please check again')
            tokens.append(parts[0])
            if dataset == "PTB":
                tags.append(parts[1])
            elif dataset == "PPCEME":
                tags.append(emetag2ptbtag[parts[1]])
            else:
                raise NameError('unknown dataset, cannot convert tags')
            nlflag = False
    fin.close()
    return data_list

def emepos2conll(infile, outfile, debug=False):
    """
    historical tagset
    skip CODE tag
    generate empty line when meet ID tag
    """
    if not debug:
        fout = open(outfile, 'w')
    fin = open(infile, 'r')
    for line in fin:
        parts = line.strip().split("/")
        if len(parts) != 2 or parts[1] == "CODE": continue
        if parts[1] == "ID":
            if debug:
                print("\n", end='', flush=True)
            else:
                fout.write("\n")
        else:
            tag = parts[1]
            if "+" in tag: tag = tag[:tag.find("+")]
            tag = re.sub("\d", "", tag)
            if debug:
                print("%s\t%s\n" %(parts[0],tag), end='', flush=True)
            else:
                fout.write("%s\t%s\n" %(parts[0],tag))
    fin.close()
    if not debug:
        fout.close()

def ptbpos2conll(infile, outfile, debug=False):
    """
    process PTB2 pos files
    """
    if not debug:
        fout = open(outfile, 'w')
    fin = open(infile, 'r')
    nlflag = True
    esflag = False
    for line in fin:
        if line.strip() == "" or line.startswith("="):
            if (not nlflag) and (esflag):
                if debug:
                    print("\n", end='', flush=True)
                else:
                    fout.write("\n")
                nlflag = True
                esflag = False
        else:
            parts = line.strip().split()
            for part in parts:
                if part == "[" or part == "]": continue
                idx = part.rfind("/")
                tok, tag = part[:idx], part[idx+1:]
                tok = tok.replace("\/", "/")
                if "|" in tag: tag = tag[:tag.find("|")]
                if debug:
                    print("%s\t%s\n" %(tok,tag), end='', flush=True)
                else:
                    fout.write("%s\t%s\n" %(tok,tag))
                if tag == ".":
                    esflag = True
            nlflag = False
    fin.close()
    if not debug:
        fout.write("\n")
        fout.close()

if __name__ == '__main__':

    dataset = sys.argv[1]

    if dataset == "PTB_all":
        base = "../resources/conair_treebank3_pos/all_wsj/"
        newbase = "processed/PTB_all/"
        pos_files = [base+fname for fname in os.listdir(base) if fname.endswith('.pos')]
        conll_files = [newbase+fname.replace(".pos", ".conll") for fname in os.listdir(base) if fname.endswith('.pos')]
        for fname, nfname in zip(pos_files, conll_files):
            ptbpos2conll(fname, nfname)
        data_list = []
        for fname in conll_files:
            data_list.extend(conll2list(fname))
        pickle.dump(data_list, open("processed/PTB_all.pkl", 'wb'))

    if dataset == "PTB_train":
        base = "../resources/conair_treebank3_pos/wsj_00-18/"
        newbase = "processed/PTB_train/"
        pos_files = [base+fname for fname in os.listdir(base) if fname.endswith('.pos')]
        conll_files = [newbase+fname.replace(".pos", ".conll") for fname in os.listdir(base) if fname.endswith('.pos')]
        for fname, nfname in zip(pos_files, conll_files):
            ptbpos2conll(fname, nfname)
        data_list = []
        for fname in conll_files:
            data_list.extend(conll2list(fname))
        pickle.dump(data_list, open("processed/PTB_train.pkl", 'wb'))

    if dataset == "PTB_dev":
        base = "../resources/conair_treebank3_pos/wsj_19-21/"
        newbase = "processed/PTB_dev/"
        pos_files = [base+fname for fname in os.listdir(base) if fname.endswith('.pos')]
        conll_files = [newbase+fname.replace(".pos", ".conll") for fname in os.listdir(base) if fname.endswith('.pos')]
        for fname, nfname in zip(pos_files, conll_files):
            ptbpos2conll(fname, nfname)
        data_list = []
        for fname in conll_files:
            data_list.extend(conll2list(fname))
        pickle.dump(data_list, open("processed/PTB_dev.pkl", 'wb'))

    # this generates PPCEME_train #, PPCEME_test, and PPCEME_all
    if dataset == "PPCEME_all":
        base = "../resources/conair_PPCEME_pos/all/"
        newbase = "processed/PPCEME_all/"
        pos_files = [base+fname for fname in os.listdir(base) if fname.endswith('.pos')]
        conll_files = [newbase+fname.replace(".pos", ".conll") for fname in os.listdir(base) if fname.endswith('.pos')]
        for fname, nfname in zip(pos_files, conll_files):
            emepos2conll(fname, nfname)
        data_list = []
        for fname in conll_files:
            data_list.extend(conll2list(fname, dataset="PPCEME"))
        pickle.dump(data_list, open("processed/PPCEME_all.pkl", 'wb'))

#         import random
#         random.seed(2019)
#         random.shuffle(data_list)
#         split_point = int(len(data_list) * 0.75)
#         pickle.dump(data_list[:split_point], open("processed/PPCEME_train.pkl", 'wb'))
#         pickle.dump(data_list[split_point:], open("processed/PPCEME_test.pkl", 'wb'))

    if dataset == "PPCEME_train":
        base = "../resources/conair_PPCEME_pos/train/"
        newbase = "processed/PPCEME_train/"
        pos_files = [base+fname for fname in os.listdir(base) if fname.endswith('.pos')]
        conll_files = [newbase+fname.replace(".pos", ".conll") for fname in os.listdir(base) if fname.endswith('.pos')]
        for fname, nfname in zip(pos_files, conll_files):
            emepos2conll(fname, nfname)
        data_list = []
        for fname in conll_files:
            data_list.extend(conll2list(fname, dataset="PPCEME"))
        pickle.dump(data_list, open("processed/PPCEME_train.pkl", 'wb'))
        
    if dataset == "PPCEME_test":
        base = "../resources/conair_PPCEME_pos/test/"
        newbase = "processed/PPCEME_test/"
        pos_files = [base+fname for fname in os.listdir(base) if fname.endswith('.pos')]
        conll_files = [newbase+fname.replace(".pos", ".conll") for fname in os.listdir(base) if fname.endswith('.pos')]
        for fname, nfname in zip(pos_files, conll_files):
            emepos2conll(fname, nfname)
        data_list = []
        for fname in conll_files:
            data_list.extend(conll2list(fname, dataset="PPCEME"))
        pickle.dump(data_list, open("processed/PPCEME_test.pkl", 'wb'))

    if dataset == "taglist": # generate PTB tag list and COARSE tag map [WARNING: regenerate this would require a re-training of the trained model!]
        ptb_pkl_file = "processed/PTB_all.pkl"
        ptb_data = pickle.load(open(ptb_pkl_file, 'rb'))
        ptb_tagset = set()
        for ptb_elem in ptb_data:
            ptb_tagset.update(ptb_elem[1])
        ptb_taglist = list(ptb_tagset)
        pickle.dump(ptb_taglist, open('processed/PTB_taglist.pkl', 'wb'))
        
        coarse_map = []
        for tag in ptb_taglist:
            coarse_map.append(coarse2index[PTB2coarse[tag]])
        pickle.dump(coarse_map, open('processed/coarse_map.pkl', 'wb'))
        
    if dataset == "sep_PPCEME_train":
        max_seq_length = 256
        input_file = "processed/PPCEME_train.pkl"
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
        pickle.dump(new_data_list, open("processed/sep_PPCEME_train.pkl", 'wb'))
        
    if dataset == "sep_PPCEME_test":
        max_seq_length = 256
        input_file = "processed/PPCEME_test.pkl"
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
        pickle.dump(new_data_list, open("processed/sep_PPCEME_test.pkl", 'wb'))
