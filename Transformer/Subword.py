import sentencepiece as spm
data_dir="./DATA/rawdata"
dataset_name="ted2020"
from pathlib import Path
prefix=Path(data_dir).absolute() / dataset_name
src_lang='en'
tgt_lang='zh'
data_prefix=f'{prefix}/train_dev.raw'

vocab_size=8000
if (prefix/f'spm{vocab_size}.model').exists():
    print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train')

else:
    spm.SentencePieceTrainer.train(
        input=','.join([f'{prefix}/train.clean.{src_lang}',
                        f'{prefix}/train.clean.{tgt_lang}',
                        f'{prefix}/valid.clean.{src_lang}',
                        f'{prefix}/valid.clean.{tgt_lang}']),
        model_prefix=prefix/f'spm{vocab_size}',
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram',
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )




spam_model=spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))

in_tag={
    'train': 'train.clean',
    'valid': 'valid.clean',
}

# for split in ['train', 'valid']:
#     for lang in [src_lang, tgt_lang]:
#         out_path=prefix/f'{split}.{lang}'
#         if out_path.exists():
#             print(f"{out_path} exists. skipping spm_encode.")

#         else:
#             with open(out_path,'w') as out_f:
#                 with open(prefix/f'{in_tag[split]}.{lang}', 'r') as in_f:
#                     for line in in_f:
#                         line=line.strip()
#                         tok=spam_model.encode(line, out_type=str)
#                         print(' '.join(tok), file=out_f)

text = "Machine translation is interesting."
print(spam_model.encode(text, out_type=str))
print(spam_model.encode(text, out_type=int))
print(spam_model.decode(spam_model.encode(text, out_type=int)))

from tqdm import tqdm

def encode_file_to_ids(input_path, output_path):
    input_path=Path(input_path)
    output_path=Path(output_path)

    if output_path.exists():
        print(output_path,' exists, skip')
        return
    with open(input_path, "r") as fin,\
        open(output_path,"w") as fout:
        for line in tqdm(fin, desc=f"encode {input_path.name}"):
            ids=spam_model.encode(line.strip(), out_type=int, add_eos=True)
            fout.write(" ".join(map(str,ids))+"\n")

for split, tag in in_tag.items():
    for lang in [src_lang, tgt_lang]:
        encode_file_to_ids(
            prefix/f"{tag}.{lang}",
            prefix/f"{split}.ids.{lang}"
        )