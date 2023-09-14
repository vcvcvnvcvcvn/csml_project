##this script is used to gen translation memory for bt text.
##output file: all.json, train.json, dev.json, test.json, id2text.pkl,bm25_src.pkl, src_editdis_alpha_0.7.pkl
##steps: bpe -> text2json -> make vocab -> bm25 ->build memory
from collections import Counter
from tqdm import tqdm
import json
import os
import argparse
import logging
from elasticsearch import Elasticsearch
import re
import sacrebleu
import editdistance
import pickle
from elasticsearch.helpers import bulk
import random
import numpy as np


lang = ['zh','en']
max_len = 500
min_len = 1
src_lang = lang[0]
trg_lang = lang[1]
output_file = 'all.json'
train_data_src = '../iwslt-2014/ende/train.de'
train_data_tgt = '../iwslt-2014/ende/train.en'
dev_data_src = '../iwslt-2014/ende/dev.de'
dev_data_tgt = '../iwslt-2014/ende/dev.en'
test_data_src = '../iwslt-2014/ende/test.de'
test_data_tgt = '../iwslt-2014/ende/test.en'
file_list = {
    'train':[train_data_src,train_data_tgt,'train.json'],
    'dev':[dev_data_src,dev_data_tgt,'dev.json'],
    'test':[test_data_src,test_data_tgt,'test.json']
}
ratio = 1.5
vocab_src = 'src.vocab'
vocab_tgt = 'tgt.vocab'
tm_size = 5

rbt_embd = pickle.load(open('zhen_embd.pkl','rb'))

def get_rbt_sim(src_idx,tm_idx):
    src_embd = rbt_embd[src_idx]
    tm_embd = rbt_embd[tm_idx]
    rbt_sim = np.dot(src_embd,tm_embd)/np.linalg.norm(src_embd)*np.linalg.norm(tm_embd)
    return rbt_sim


def tags2tidy(path,output):
    with open(output, 'w') as fo:
        for x in open(path).readlines():
            if x[0]=='<' and x[-1]=='>':
                continue
            else:
                fo.write(x+'\n')





def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--build_index', action='store_true',
        help='whether to train an index from scratch',default=True,)
    parser.add_argument('--search_index', action='store_true',
        help='whether to search from a built index',default=True,)
    parser.add_argument('--index_file', type=str, default='index_file.json')#####!!!!!
    parser.add_argument('--search_file', type=str, default='all.json')
    parser.add_argument('--output_file', type=str, default='bm25_src.pkl')
    parser.add_argument('--index_name', type=str,default='ende_src')
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--allow_hit', action='store_true',default=True)
    return parser.parse_args()

def get_json(f):
    return [json.loads(x) for x in open(f).readlines()]

def get_edit_sim(src,tm):
    
    a = src.split()
    b = tm.split()
    edit_distance = editdistance.eval(a, b)

    edit_sim = 1 - edit_distance / max(len(src), len(tm))

    return edit_sim

def make_vocab(batch_seq, char_level=False):
    cnt = Counter()
    for seq in batch_seq:
        cnt.update(seq)
    if not char_level:
        return cnt
    char_cnt = Counter()
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt

def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n'%(x,y))



def debpe(bpe):
    return re.sub(r'(@@ )|(@@ ?$)', '', bpe)

def get_unedited_words(src_sent, src_tm_sent):
    # Here first sentence is the edited sentence and the second sentence is the target sentence
    # If src_sent is the first sentence, then insertion and deletion should be reversed.
    """ Dynamic Programming Version of edit distance and finding unedited words."""
    a = src_sent.split()
    b = src_tm_sent.split()
    edit_distance = editdistance.eval(a, b)

    edit_distance = 1 - edit_distance / max(len(src_sent), len(src_tm_sent))

    return edit_distance

def get_bleu(src,tm):
    src = [src]
    tm = [tm]
    return sacrebleu.corpus_bleu(tm,[src],force=True,lowercase=False,tokenize='none').score

def get_topk_sent_id(src, src_sim, k=6):
    scores = list(map(lambda x: -get_unedited_words(src, x), src_sim))
    topk = sorted(zip(scores, range(len(scores))), key=lambda x: x[0])[:k]
    ans = [it[1] for it in topk]
    return ans

def tidy_txt():
    ###train bpe.code
    # print('*'*30)
    # print('train bpe.code')
    # gen_bpe_code_cmd = 'subword-nmt learn-joint-bpe-and-vocab --input {} {} -s 32000 -o bpe.code --write-vocabulary {} {} --num-workers 4'
    # os.system(gen_bpe_code_cmd.format(train_data_src,train_data_tgt,'vocab.'+src_lang,'vocab.'+trg_lang))
    # gen_bpe_text = 'subword-nmt apply-bpe -c bpe.code --vocabulary vocab.{} < {} > {}.{}.bpe'
    # for task in ['train','dev','test']:
    #     for idx,la in enumerate(['src','tgt']):
    #         print('gen '+task+'.'+la+'.bpe')
    #         os.system(gen_bpe_text.format(lang[idx],file_list[task][idx],task,la))
    # print('*'*30)
    print('bpe complete, then do text2json')
    tot_lines = 0
    for task in ['train','dev','test']:
        with open(file_list[task][2], "w") as outfile:
            src_file = task+'.src.bpe'
            tgt_file = task+'.tgt.bpe'
            for src_line, tgt_line in tqdm(zip(open(src_file).readlines(), open(tgt_file).readlines())):
                src_line = src_line.strip()#.split()
                tgt_line = tgt_line.strip()#.split()
                if min_len <= len(src_line) <= max_len and min_len <= len(tgt_line) <= max_len:
                    # if len(src_line)/len(tgt_line) > ratio:
                    #     continue
                    # if len(tgt_line)/len(src_line) > ratio:
                    #     continue
                    temp_dict = {}
                    temp_dict[src_lang] = src_line
                    temp_dict[trg_lang] = tgt_line
                    temp_dict['id'] = task+'_'+str(tot_lines)
                    json_object = json.dumps(temp_dict)+'\n'
                    outfile.write(json_object) 
                tot_lines += 1
    os.system('cat train.json dev.json test.json > all.json')
    print('*'*30)
    print('text2json done, then make vocab')
    src_lines = []
    tgt_lines = []
    tot_lines = 0
    for src_line, tgt_line in tqdm(zip(open('train.src.bpe').readlines(), open('train.tgt.bpe').readlines())):
        src_line = src_line.strip().split()
        tgt_line = tgt_line.strip().split()
        tot_lines += 1
        if min_len <= len(src_line) <= max_len and min_len <= len(tgt_line) <= max_len:
            if len(src_line)/len(tgt_line) > ratio:
                continue
            if len(tgt_line)/len(src_line) > ratio:
                continue
            src_lines.append(src_line)
            tgt_lines.append(tgt_line)
    tgt_lines.extend(src_lines)
    src_vocab = make_vocab(src_lines)
    tgt_vocab = make_vocab(tgt_lines)
    print (output_file, len(src_lines), tot_lines)
    write_vocab(src_vocab, vocab_src)
    write_vocab(tgt_vocab, vocab_tgt)
    print('*'*30)
    print('vocab done, then retrieval based on bm25')



def retriebal_bm25(args):
    pkl_dict = {
        'train':[],
        'dev':[],
        'test':[]
    }
    data_path = {
        'train':'train.json',
        'dev':'dev.json',
        'test':'test.json'
    }
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.WARNING)

    es = Elasticsearch([{u'host': "localhost", u'port': "9200"}])
    for task in ['train','dev','test']:
        print('begin task: '+task)
        search_file = data_path[task]
        if args.build_index:
            ids,queries, responses = [], [],[]
            samples = [json.loads(x) for x in open(args.index_file).readlines()]
            for sample in samples:
                queries.append(sample[src_lang])
                responses.append(sample[trg_lang])
                ids.append(sample['id'])


            logger.info('build with elasticsearch')

            body = {
                "settings": {
                    "index": {
                        "analysis": {
                            "analyzer": "standard"
                        },
                        "number_of_shards": "1",
                        "number_of_replicas": "1",
                    }
                },
                "mappings": {
                    "properties": {
                        "query": {
                            "type": "text",
                            "similarity": "BM25",
                            "analyzer": "standard",
                        },
                        "response": {
                            "type": "text",
                        },
                        "id": {
                            "type": "text",
                        }
                    }
                }
            }
            if es.indices.exists(index=args.index_name):
                es.indices.delete(index=args.index_name)
            es.indices.create(index=args.index_name,body=body)

            index = args.index_name

            actions = []

            for idx, (query, response,iid) in tqdm(enumerate(zip(queries, responses,ids)), total=len(queries)):
                action = {
                    "_index": index,
                    "_source": {
                        "query": debpe(query),
                        "response": response,
                        "id":iid,
                    }
                }
                actions.append(action)
                if len(actions) >= 1000:
                    success, _ = bulk(es, actions, raise_on_error=True)
                    actions = []
            if actions:
                success, _ = bulk(es, actions, raise_on_error=True)
            es.indices.refresh(index)
            info = es.indices.stats(index=index)
            print('total document indexed', info["indices"][index]["primaries"]["docs"]["count"]) 

        if args.search_index:
            queries, responses = [],[]
            with open(search_file, 'r') as f:

                data = [json.loads(x) for x in f.readlines()]
                for d in data:
                    queries.append(d[src_lang])

            logger.info('search with elasticsearch')

            index = args.index_name
            query_body = {
                    "query": {
                    "match":{
                        "query": None
                    }
                    },
                    "size": args.topk
            }

                
            for query in tqdm(queries,total=len(queries)):
                query_body["query"]["match"]["query"] = debpe(query)
                es_result = es.search(index=index, body=query_body)
                ret_q = [ item["_source"]["query"] for item in es_result["hits"]["hits"]]
                ret_i = [ item["_source"]["id"] for item in es_result["hits"]["hits"]]
                pkl_dict[task].append(ret_i)
                
    pickle.dump(pkl_dict,open('bm25_src.pkl','wb'))


def reordering_json():
    temp_ls = get_json('bt1.json')
    with open('bt.json', "w") as outfile:
        for _dict in temp_ls:
            temp_dict = {}
            temp_dict['en'] = _dict['en']
            temp_dict['de'] = _dict['de']
            json_object = json.dumps(temp_dict)+'\n'
            outfile.write(json_object)
    os.system('cat train1.json bt.json > train0.json')





def renumbering():
    
    tot_lines = -1
    for task in ['train','dev','test']:
        temp_ls = get_json(task+'0.json')
        with open(task+'.json', "w") as outfile:
            for temp_dict in temp_ls:
                tot_lines+=1
                temp_dict['id'] = task+'_'+str(tot_lines)
                json_object = json.dumps(temp_dict)+'\n'
                outfile.write(json_object)

def renumbering_index_file():
    temp_ls = get_json('train1.json')#######
    tot_lines = -1
    with open('index_file.json', "w") as outfile:
        for temp_dict in temp_ls:
            tot_lines+=1
            temp_dict['id'] = 'train_'+str(tot_lines)
            json_object = json.dumps(temp_dict)+'\n'
            outfile.write(json_object)





if __name__ == "__main__":
    #reordering_json()
    args = parse_args()
    #tidy_txt()
    #renumbering_index_file()
    #renumbering()
    os.system('cat train.json dev.json test.json > all.json')
    retriebal_bm25(args)
    print('*'*30)
    print('retrieval based on bm25 done, begin to gen memory')
    all_ls = get_json('all0.json')
    id2text = {x['id']:{src_lang:x[src_lang],trg_lang:x[trg_lang]} for x in all_ls}
    pickle.dump(id2text,open('id2text.pkl','wb'))
    id2text = {x['id']:debpe(x[src_lang]) for x in all_ls}
    bm25_dict = pickle.load(open('bm25_src.pkl','rb'))
    output_dict = {}
    output_dict_shuffle = {}
    all_ls = get_json('all.json')
    for task in ['train','dev','test']:
        sorted_retrieval_ls = []
        sorted_retrieval_ls_shuffle = []
        alpha = 0.7
        bm25_ls = bm25_dict[task]
        for idx,retrieval_ls in tqdm(enumerate(bm25_ls),total=len(bm25_ls)):
            src = debpe(all_ls[idx][src_lang])
            src_id = all_ls[idx]['id']
            if len(retrieval_ls)>0:
                # iid = -1 if idx<159319 else int(retrieval_ls[0].split('_')[-1])
                iid = int(retrieval_ls[0].split('_')[-1])
                #candidate_ls = [(x,id2text[x]) for x in retrieval_ls if int(x.split('_')[-1]) != iid]
                candidate_ls = [(x,id2text[x]) for x in retrieval_ls if x != retrieval_ls[0]]
                candidata_ls = [x for x in candidate_ls if x[1].strip() != src.strip()]
            else:
                candidate_ls = []
                candidata_ls = []
            if len(candidate_ls) < tm_size:
                temp_ret = [x[0] for x in candidate_ls]
                temp_ret_shuffle = temp_ret.copy()
                random.shuffle(temp_ret_shuffle)
                sorted_retrieval_ls.append(temp_ret)
                sorted_retrieval_ls_shuffle.append(temp_ret_shuffle)
                continue
            ret = []
            while len(ret) < tm_size:
                max_sim = -1
                for c_id,c_text in candidate_ls:

                    #edit_sim_src = get_edit_sim(src,c_text)
                    edit_sim_src = get_rbt_sim(src_id,c_id)
                    edit_sim_tm = 0
                    if ret:
                        #edit_sim_tm = sum(get_edit_sim(c_text,id2text[x]) for x in ret)/len(ret)
                        edit_sim_tm = sum(get_rbt_sim(c_id,x) for x in ret)/len(ret)
                    edit_sim = edit_sim_src - alpha * edit_sim_tm

                    if edit_sim > max_sim:
                        max_sim = edit_sim
                        max_sim_id = c_id
                        max_sim_text = c_text

                ret.append(max_sim_id)
                try:
                    candidate_ls.remove((max_sim_id,max_sim_text))
                except ValueError:
                    print('candidate_ls:',[x[0] for x in candidata_ls])
                    print('max_sim_score:',max_sim)
                    print(ret)
                    print(max_sim_id)
                    exit()
            ret_shuffle = ret.copy()
            random.shuffle(ret_shuffle)
            sorted_retrieval_ls.append(ret)
            sorted_retrieval_ls_shuffle.append(ret_shuffle)

        
        output_dict[task] = sorted_retrieval_ls
        output_dict_shuffle[task] = sorted_retrieval_ls_shuffle
        
    pickle.dump(output_dict,open('src_editdis_alpha_'+str(alpha)+'.pkl','wb'))
    #pickle.dump(output_dict_shuffle,open('src_editdis_alpha_'+str(alpha)+'_btandfromorigin_shuffle.pkl','wb'))