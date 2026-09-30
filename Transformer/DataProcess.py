data_dir="./DATA/rawdata"
dataset_name="ted2020"
from pathlib import Path
prefix=Path(data_dir).absolute() / dataset_name
src_lang='en'
tgt_lang='zh'
data_prefix=f'{prefix}/train_dev.raw'


def strQ2B(ustring):
    ss=[]
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code==12288:
            inside_code=32
        elif (inside_code>=65281 and inside_code<=65374 ):
            inside_code -=65248
        ss.append(chr(inside_code))
    return ''.join(ss)

import re
def clean_s(s:str, lang):
    if lang=='en':
        s=re.sub(r"\([^()]*\)","",s)
        s=s.replace('-',"")
        s=re.sub('([.,;!?()\"])', r' \1 ', s)
    elif lang=='zh':
        s=strQ2B(s)
        s=re.sub(r"\([^()]*\)","",s)
        s.replace(' ','')
        s = s.replace('—', '')
        s=s.replace('“','"')
        s=s.replace('”','"')
        s=s.replace('_','')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s)
    s=' '.join(s.strip().split())

    return s

def len_s(s:str, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())


def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return

    with open(f'{prefix}.{l1}','r') as l1_in_f:
        with open(f'{prefix}.{l2}','r') as l2_in_f:
            with open(f'{prefix}.clean.{l1}','w') as l1_out_f:
                with open(f'{prefix}.clean.{l2}','w') as l2_out_f:
                    for s1 in l1_in_f:
                        s1=s1.strip()
                        s2=l2_in_f.readline().strip()
                        s1=clean_s(s1,l1)
                        s2=clean_s(s2,l2)
                        s1_len=len_s(s1,l1)
                        s2_len=len_s(s2,l2)
                        if min_len>0 :
                            if s1_len<min_len or s2_len<min_len:
                                continue
                        if max_len>0:
                            if s1_len>max_len or s2_len>max_len:
                                continue
                        if ratio>0:
                            if s1_len/s2_len>ratio or s2_len/s1_len>ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)

clean_corpus(data_prefix,src_lang,tgt_lang)
valid_ratio=0.01
train_ratio=1 - valid_ratio

if (prefix/f'train.clean.{src_lang}').exists() and \
    (prefix/f'train.clean.{tgt_lang}').exists() and \
    (prefix/f'valid.clean.{src_lang}').exists() and \
    (prefix/f'valid.clean.{tgt_lang}').exists():
        print(f'train/valid splits exists. skipping split.')
else:
    line_num =sum(1 for line in open(f'{data_prefix}.clean.{src_lang}'))
    labels=list(range(line_num))
    import random
    import os
    random.shuffle(labels)
    for lang in [src_lang, tgt_lang]:
        train_f=open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w')
        valid_f=open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w')
        count=0
        for line in open(f'{data_prefix}.clean.{lang}', 'r'):
            if labels[count]/line_num<train_ratio:
                train_f.write(line)
            else:
                valid_f.write(line)
            count+=1

        train_f.close()
        valid_f.close()