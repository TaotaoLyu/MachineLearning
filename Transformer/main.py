import sentencepiece as spm
data_dir="./DATA/rawdata/ted2020"
dataset_name="ted2020"
from pathlib import Path
prefix=Path(data_dir).absolute() / dataset_name
src_lang='en'
tgt_lang='zh'
data_prefix=f'{prefix}/train_dev.raw'
spm_model_path='./DATA/rawdata/ted2020/spm8000.model'
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import logging
import sys

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("hw5.seq2seq.modern")

lr_factor=2.0
lr_warmup=4000


from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from argparse import Namespace
config = Namespace(
    data_dir=str(prefix),
    savedir="./checkpoints/rnn-modern",
    source_lang="en",
    target_lang="zh",

    num_workers=2,
    batch_size=64,
    accum_steps=2,

    lr_factor=2.0,
    lr_warmup=4000,
    weight_decay=1e-4,

    clip_norm=1.0,
    max_epoch=30,

    beam=5,
    max_len_a=1.2,
    max_len_b=10,
    len_penalty=1.0,

    keep_last_epochs=5,
    resume=None,

    use_wandb=False,
)

import torch
class ParallelIDDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        super().__init__()
        self.src_file=Path(src_file)
        self.tgt_file=Path(tgt_file)

        with open(self.src_file, 'r') as f:
            self.src_lines=f.read().splitlines()
        with open(self.tgt_file, 'r') as f:
            self.tgt_lines=f.read().splitlines()

        if len(self.src_lines)!=len(self.tgt_lines):
            raise ValueError(
                f"pallel data size mismatch: {len(self.src_lines)} != {len(self.tgt_lines)}"
            )

    def __len__(self):
        return len(self.src_lines)

    @staticmethod
    def parse_ids(line:str):
        if not line.strip():
            return torch.empty(0,dtype=torch.long)
        return torch.tensor([int(x) for x in line.split()], dtype=torch.long)

    def __getitem__(self, index):
        return {
            "id": index,
            "source": self.parse_ids(self.src_lines[index]),
            "target": self.parse_ids(self.tgt_lines[index])
        }

def collate_translation(samples, pad_id, bos_id):
    ids=torch.tensor([s["id"] for s in samples], dtype=torch.long)
    src_list= [s["source"] for s in samples]
    tgt_list=[s["target"] for s in samples]

    src_length=torch.tensor([len(x) for x in src_list], dtype=torch.long)

    src_tokens=pad_sequence(
        src_list,
        batch_first=True,
        padding_value=pad_id
    )

    target=pad_sequence(
        tgt_list,
        batch_first=True,
        padding_value=pad_id
    )
    prev_list=[]
    for tgt in tgt_list:
        bos=torch.tensor([bos_id],dtype=torch.long)
        prev_list.append(torch.cat([bos,tgt[:-1]]))

    prev_output_tokens=pad_sequence(
        prev_list,
        batch_first=True,
        padding_value=pad_id
    )

    ntokens=int(target.ne(pad_id).sum().item())

    return {
        "id": ids,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input":{
            "src_tokens": src_tokens,
            "src_length": src_length,
            "prev_out_tokens": prev_output_tokens,
        },
        "target":target
    }

import lightning as L
from functools import partial
class TranslationDataModule(L.LightningDataModule):
    def __init__(self, pad_id, bos_id, batch_size=64, num_workers=2, extra_train_dirs=None):
        super().__init__()
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.extra_train_dirs=[Path(x) for x in (extra_train_dirs or [])]

        self.collate_fn = partial(collate_translation, pad_id=pad_id, bos_id=bos_id)

    def _dataset(self, directory, split):
        directory=Path(directory)
        return ParallelIDDataset(
            directory/f"{split}.ids.{src_lang}",
            directory/f"{split}.ids.{tgt_lang}",
        )

    def setup(self, stage=None):
        main_train=self._dataset(data_dir,"train")
        extra= [ self._dataset(d, "train") for d in self.extra_train_dirs]

        self.train_ds=(
            ConcatDataset([main_train,*extra]) if extra else main_train
        )

        self.valid_ds=self._dataset(data_dir, "valid")

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers>0
        )

    def train_dataloader(self):
        return self._loader(self.train_ds,True)

    def val_dataloader(self):
        return self._loader(self.valid_ds,False)

sp=spm.SentencePieceProcessor(model_file=str(spm_model_path))
VOCAB_SIZE=len(sp)+1

# PAD_ID=sp.pad_id()
PAD_ID=VOCAB_SIZE-1
BOS_ID=sp.bos_id()
EOS_ID=sp.eos_id()
UNK_ID=sp.unk_id()
print(f"PAD_ID: {sp.pad_id()} or {PAD_ID}, BOS_ID: {BOS_ID}, EOS_ID: {EOS_ID}, UNK_ID: {UNK_ID}, VOCAB_SIZE: {VOCAB_SIZE}")

dm=TranslationDataModule(
    PAD_ID,
    BOS_ID
)

dm.setup()


batch=next(iter(dm.val_dataloader()))
print(batch["net_input"]["src_tokens"].shape)
print(batch["net_input"]["src_length"].shape)
print(batch["target"].shape)
print(batch["net_input"]["prev_out_tokens"].shape)

def strip_special(ids, pad_id, bos_id, eos_id):
    out = []
    for x in ids:
        x = int(x)
        if x == pad_id or x == bos_id:
            continue
        if x == eos_id:
            break
        out.append(x)
    return out

src0 = strip_special(
    batch["net_input"]["src_tokens"][0].tolist(),
    PAD_ID,
    BOS_ID,
    EOS_ID,
)
tgt0 = strip_special(
    batch["target"][0].tolist(),
    PAD_ID,
    BOS_ID,
    EOS_ID,
)

print("Source:", sp.decode(src0))
print("Target:", sp.decode(tgt0))


import torch.nn as nn
class RNNEncoder(nn.Module):
    def __init__(self, args, embed_tokens, padding_idx):
        super().__init__()
        self.embed_tokens=embed_tokens
        self.padding_idx=padding_idx

        self.embed_dim = args.encoder_embed_dim
        self.hidden_dim = args.encoder_ffn_embed_dim
        self.num_layers = args.encoder_layers

        self.dropout_in=nn.Dropout(args.dropout)
        self.rnn=nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=args.dropout if self.num_layers>1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        self.dropout_out=nn.Dropout(args.dropout)

    def combine_bidir(self, hidden, batch_size):
        # [2L, B, H] => [L, B, 2H]
        hidden=hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        # [L, 2, G, H]
        hidden=hidden.permute(0,2,1,3).contiguous()
        return hidden.view(self.num_layers, batch_size, 2*self.hidden_dim)

    def forward(self, src_tokens, src_length=None):
        batch_size=src_tokens.size(0)
        # src_tokens [B,T] -> x [B,T, embed_dim]
        x=self.embed_tokens(src_tokens)
        x=self.dropout_in(x)

        h0=x.new_zeros(
            2* self.num_layers,
            batch_size,
            self.hidden_dim
        )

        outputs, hidden=self.rnn(x,h0)
        outputs=self.dropout_out(outputs)
        hidden=self.combine_bidir(hidden, batch_size)

        padding_mask=src_tokens.eq(self.padding_idx)

        return {
            "encoder_out": outputs, # [B, T, 2*hidden_dim]
            "encoder_hidden": hidden,  # [L, B, 2*hidden_dim]
            "padding_mask": padding_mask,
        }
import torch.nn.functional as F
class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()
        self.input_proj=nn.Linear(
            input_embed_dim,
            source_embed_dim,
            bias=bias
        )

        self.output_proj=nn.Linear(
            input_embed_dim+source_embed_dim,
            output_embed_dim,
            bias=bias
        )

    def forward(self, inputs, encoder_outputs, encoder_paddinng_mask):
        #  input [B, T, E]
        #  encoder_outputs [B, S, E]
        #  padding mask [B, S]
        query = self.input_proj(inputs)
        scores=torch.bmm(
            query,
            encoder_outputs.transpose(1,2),
        )
        # [B, T, S]

        if encoder_paddinng_mask is not None:
            scores=scores.float().masked_fill(       # [B, T, S]
                encoder_paddinng_mask.unsqueeze(1),  # [B, 1, S]
                float("-inf")
            ).type_as(scores)
        # [B, T, S]
        attn=F.softmax(scores,dim=-1)

        context=torch.bmm(attn, encoder_outputs)
        # [B, T, H]

        x=torch.cat([context, inputs], dim=-1)
        x=torch.tanh(self.output_proj(x))

        return x, attn
    
class RNNDecoder(nn.Module):
    def __init__(self, args, embed_tokens, padding_idx, vocab_size):
        super().__init__()
        self.embed_tokens=embed_tokens
        self.padding_idx=padding_idx

        assert args.decoder_layers==args.encoder_layers
        assert args.decoder_ffn_embed_dim==2*args.encoder_ffn_embed_dim

        self.embed_dim=args.decoder_embed_dim
        self.hidden_dim=args.decoder_ffn_embed_dim
        self.num_layers=args.decoder_layers

        self.drop_in=nn.Dropout(args.dropout)
        self.attention=AttentionLayer(
            input_embed_dim=self.embed_dim,
            source_embed_dim=self.hidden_dim,
            output_embed_dim=self.embed_dim,
            bias=False
        )

        self.rnn=nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=args.dropout if self.num_layers>1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.drop_out=nn.Dropout(args.dropout)

        self.project_out_dim=(
            nn.Linear(self.hidden_dim, self.embed_dim)
            if self.hidden_dim!=self.embed_dim
            else nn.Identity()
        )

        # [B, T, hidden_size(2*H)]=>[B, T, embed_dim]

        self.output_projection=nn.Linear(
            self.embed_dim,
            vocab_size,
            bias=False,
        )
        if args.share_decoder_input_output_embed:
            self.output_projection.weight=self.embed_tokens.weight


    def forward(self, prev_output_tokens, encoder_out):
        encoder_outputs=encoder_out["encoder_out"]
        encoder_hidden=encoder_out["encoder_hidden"]
        encoder_padding_mask=encoder_out["padding_mask"]

        x=self.embed_tokens(prev_output_tokens)
        x=self.drop_in(x)

        x, attn=self.attention(
            x,
            encoder_outputs,
            encoder_padding_mask,
        )
        # can optimization
        x, final_hidden=self.rnn(x, encoder_hidden)
        x=self.drop_out(x)
        x=self.project_out_dim(x)
        logits=self.output_projection(x)

        return logits, {
            "attention": attn,
            "hidden": final_hidden, # [L, B, H]
        }





class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self, src_tokens, src_length, prev_out_tokens):
        encoder_out=self.encoder(src_tokens,src_length)
        logits, extra=self.decoder(prev_out_tokens, encoder_out)
        return logits, extra


def init_params(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    elif isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)

    elif isinstance(module, nn.RNNBase):
        for name, param in module.named_parameters():
            if "weight" in name or "bias" in name:
                nn.init.uniform_(param, -0.1, 0.1)


arch_args=Namespace(
    model_type="rnn",

    encoder_embed_dim=256,
    encoder_ffn_embed_dim=512,
    encoder_layers=1,

    decoder_embed_dim=256,
    decoder_ffn_embed_dim=1024,
    decoder_layers=1,

    share_decoder_input_output_embed=True,
    dropout=0.3,

)

transformer_args = Namespace(
    model_type="transformer",

    encoder_embed_dim=256,
    encoder_ffn_embed_dim=1024,
    encoder_layers=4,

    decoder_embed_dim=256,
    decoder_ffn_embed_dim=1024,
    decoder_layers=4,

    encoder_attention_heads=4,
    decoder_attention_heads=4,
    activation_fn="relu",

    share_decoder_input_output_embed=True,
    dropout=0.3,
    max_source_positions=4096,
    max_target_positions=4096,
)


def build_model(args, vocab_size, pad_id):
    encoder_embed_tokens=nn.Embedding(
        vocab_size,
        args.encoder_embed_dim,
        padding_idx=pad_id
    )
    decoder_embed_tokens=nn.Embedding(
        vocab_size,
        args.decoder_embed_dim,
        padding_idx=pad_id
    )

    if args.model_type=="rnn":
        encoder=RNNEncoder(
            args,
            encoder_embed_tokens,
            pad_id
        )
        decoder=RNNDecoder(
            args,
            decoder_embed_tokens,
            pad_id,
            vocab_size,
        )

    elif args.model_type=="transformer":
        encoder=TransformerEncoderModel(
            args,
            encoder_embed_tokens,
            pad_id
        )
        decoder=TransformerDecoderModel(
            args,
            decoder_embed_tokens,
            pad_id,
            vocab_size
        )

    model=Seq2Seq(encoder,decoder)
    model.apply(init_params)

    with torch.no_grad() :
        model.encoder.embed_tokens.weight[pad_id].zero_()
        model.decoder.embed_tokens.weight[pad_id].zero_()

    return model

import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEncoderModel(nn.Module):
    def __init__(self, args, embed_tokens, padding_idx):
        super().__init__()

        self.embed_tokens=embed_tokens
        self.padding_idx=padding_idx
        self.d_model=args.encoder_embed_dim

        self.positional=PositionalEncoding(
            self.d_model,
            dropout=args.dropout,
            max_len=args.max_source_positions,
        )

        layer=nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=args.encoder_attention_heads,
            dim_feedforward=args.encoder_ffn_embed_dim,
            dropout=args.dropout,
            activation=args.activation_fn,
            batch_first=True,
            norm_first=True
        )

        self.encoder=nn.TransformerEncoder(
            layer,
            num_layers=args.encoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )

    def forward(self, src_tokens, src_length=None):
        padding_mask=src_tokens.eq(self.padding_idx)

        x=self.embed_tokens(src_tokens)*math.sqrt(self.d_model)
        x=self.positional(x)

        memory=self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        return {
            "encoder_out": memory,
            "encoder_hidden": None,
            "padding_mask": padding_mask
        }

class TransformerDecoderModel(nn.Module):
    def __init__(self, args, embed_tokens, padding_idx, vocab_size):
        super().__init__()
        self.embed_tokens=embed_tokens
        self.padding_idx=padding_idx
        self.d_model=args.decoder_embed_dim

        self.positional=PositionalEncoding(
            self.d_model,
            dropout=args.dropout,
            max_len=args.max_target_positions,
        )

        layer=nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=args.decoder_attention_heads,
            dim_feedforward=args.decoder_ffn_embed_dim,
            dropout=args.dropout,
            activation=args.activation_fn,
            batch_first=True,
            norm_first=True
        )

        self.decoder=nn.TransformerDecoder(
            layer,
            num_layers=args.decoder_layers,
            norm=nn.LayerNorm(self.d_model)
        )

        self.output_projection=nn.Linear(
            self.d_model,
            vocab_size,
            bias=True
        )

        if args.share_decoder_input_output_embed:
            self.output_projection.weight=self.embed_tokens.weight

    def forward(self, prev_output_tokens, encoder_out):
        memory=encoder_out["encoder_out"]
        memory_padding_mask=encoder_out["padding_mask"]

        tgt_padding_mask=prev_output_tokens.eq(self.padding_idx)
        tgt_len=prev_output_tokens.size(1)


        causal_mask=torch.triu(
            torch.ones(
                tgt_len,
                tgt_len,
                dtype=torch.bool,
                device=prev_output_tokens.device,
            ),
            diagonal=1
        )

        x=self.embed_tokens(prev_output_tokens)*math.sqrt(self.d_model)
        x=self.positional(x)  

        x=self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )

        logits=self.output_projection(x)

        return logits, {}

# model=build_model(arch_args, VOCAB_SIZE, PAD_ID)
model=build_model(transformer_args, VOCAB_SIZE, PAD_ID)
print(model)

with torch.no_grad():
    logits, extra=model(**batch["net_input"])

print("logits: ", logits.shape)


def compute_translation_loss(logits, target, pad_id, smoothing=0.1):
    # logits [B, T, V]
    # target [B, T]
    vocab_size=logits.size(-1)

    loss=F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=smoothing,
        reduction="sum"
    )

    ntokens=target.ne(pad_id).sum().clamp_min(1)
    return loss/ntokens

def noam_rate(step, model_size, factor, warmup):
    step=step+1
    return factor*(
        model_size**(-0.5)
        *min(step**(-0.5), step*warmup**(-1.5))
    )

import numpy as np
steps=np.arange(100_000)
rates=[
    noam_rate(
        int(s),
        arch_args.encoder_embed_dim,
        # transformer_args.encoder_embed_dim,
        lr_factor,
        lr_warmup,
    )
    for s in steps
]
import matplotlib.pyplot as plt
plt.plot(steps, rates)
plt.xlabel("optimizer step")
plt.ylabel("learning rate")
# plt.show()
plt.savefig("learning_rate_curve.png", dpi=300, bbox_inches="tight")


@torch.no_grad()
def beam_search_sigle(
    model,
    src_tokens,
    src_length,
    bos_id,
    eos_id,
    beam_size=5,
    max_len_a=1.2,
    max_len_b=10,
    len_penalty=1.0
):
    model.eval()
    encoder_out=model.encoder(
        src_tokens,
        torch.tensor([src_length], device=src_tokens.device)
    )

    max_len=max(1, int(max_len_a*src_length+max_len_b))

    beams=[([bos_id], 0.0, False)]

    def normalized_score(item):
        tokens, score, _ =item
        length=max(1, len(tokens)-1)

        return score/ (length**len_penalty)

    for _ in range(max_len):
        candidates=[]

        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens,score, True))
                continue
            prev=torch.tensor(
                tokens,
                dtype=torch.long,
                device=src_tokens.device,
            ).unsqueeze(0)

            logits, _=model.decoder(prev, encoder_out)
            log_probs=F.log_softmax(logits[:,-1,:], dim=-1).squeeze(0)

            log_probs[PAD_ID] = float("-inf")
            log_probs[BOS_ID] = float("-inf")
            topk_logp, topk_ids= torch.topk(log_probs, beam_size)

            for logp, token_id in zip(topk_logp.tolist(), topk_ids.tolist()):
                newtokens=tokens+[token_id]
                candidates.append(
                    (
                        newtokens,
                        score+logp,
                        token_id==eos_id
                    )
                )
        candidates.sort(key=normalized_score, reverse=True)
        beams=candidates[:beam_size]

        if all(x[2] for x in beams):
            break

    best=max(beams, key=normalized_score)

    return best[0][1:]

import sacrebleu

class TranslationSystem(L.LightningModule):
    def __init__(
            self,
            arch_args,
            vocab_size,
            pad_id,
            bos_id,
            eos_id,
            spm_model_path,
            source_lang="en",
            target_lang="zh",
            lr_factor=0.2,
            lr_warmup=4000,
            weight_decay=1e-4,
            beam=5,
            max_len_a=1.2,
            max_len_b=10,
            len_penalty=1.0,
            label_smoothing=0.1,
            savedir="./checkpoints/model",
            ):
        super().__init__()

        if isinstance(arch_args, Namespace):
            arch_args=vars(arch_args)

            self.save_hyperparameters()

            self.arch_args=Namespace(**arch_args)

            self.model=build_model(
                self.arch_args,
                vocab_size,
                pad_id,
            )

            self.pad_id=pad_id
            self.bos_id=bos_id
            self.eos_id=eos_id
            self.sp=spm.SentencePieceProcessor(model_file=str(spm_model_path))

            self.val_srcs=[]
            self.val_hyps=[]
            self.val_refs=[]

    def forward(self, **net_inout):
        return self.model(**net_inout)

    def _loss(self, batch):
        logits, _ = self.model(**batch["net_input"])
        loss=compute_translation_loss(
            logits,
            batch["target"],
            self.pad_id,
            smoothing=self.hparams.label_smoothing,
        )

        return loss

    def training_step(self, batch, batch_idx):
        loss=self._loss(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["nsentences"]
        )

        return loss

    def on_validation_epoch_start(self):
        self.val_srcs=[]
        self.val_hyps=[]
        self.val_refs=[]

    def validation_step(self, batch, batch_idx):
        loss=self._loss(batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["nsentences"],
        )

        if self.trainer.sanity_checking:
            return loss


        src_tokens=batch["net_input"]["src_tokens"]
        src_lengths=batch["net_input"]["src_length"]
        targets=batch["target"] 

        for i in range(src_tokens.size(0)):
            src_len=int(src_lengths[i].item()) 
            src=src_tokens[i:i+1, :src_len]

            hyp_ids=beam_search_sigle(
                self.model,
                src,
                src_len,
                self.bos_id,
                self.eos_id,
                beam_size=self.hparams.beam,
                max_len_a=self.hparams.max_len_a,
                max_len_b=self.hparams.max_len_b,
                len_penalty=self.hparams.len_penalty
            )    

            src_ids=strip_special(
                src.squeeze(0).tolist(),
                self.pad_id,
                self.bos_id,
                self.eos_id,
            )  

            ref_ids=strip_special(
                targets[i].tolist(),
                self.pad_id,
                self.bos_id,
                self.eos_id,               
            )

            hyp_ids=strip_special(
                hyp_ids,
                self.pad_id,
                self.bos_id,
                self.eos_id,
            )

            self.val_srcs.append(self.sp.decode(src_ids))
            self.val_refs.append(self.sp.decode(ref_ids))
            self.val_hyps.append(self.sp.decode(hyp_ids))


        return loss


    def on_validation_epoch_end(self):
        if not self.val_hyps:
            return


        tokenize="zh" if self.hparams.target_lang=="zh" else "13a"

        bleu=sacrebleu.corpus_bleu(
            self.val_hyps,
            [self.val_refs],
            tokenize=tokenize
        )

        self.log("val_bleu", bleu.score, prog_bar=True)

        show_id=np.random.randint(len(self.val_hyps))
        logger.info("example source: "+self.val_srcs[show_id])
        logger.info("example hypothesis: "+self.val_hyps[show_id])
        logger.info("example reference: "+self.val_refs[show_id])
        logger.info("BLEU: %.4f", bleu.score)

        savedir=Path(self.hparams.savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        sample_file=savedir/f"sample_epoch_{self.current_epoch:02d}.txt"
        with open(sample_file, "w") as f:
            for s, h, r in zip(self.val_srcs, self.val_hyps, self.val_refs):
                f.write(f"SRC\t{s}\n")
                f.write(f"HYP\t{h}\n")
                f.write(f"REF\t{r}\n")


    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(
            self.parameters(),
            lr=1.0,
            betas=(0.9,0.98),
            eps=1e-9,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler=torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step:noam_rate(
                step,
                self.arch_args.encoder_embed_dim,
                self.hparams.lr_factor,
                self.hparams.lr_warmup,
            )

        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }


# system = TranslationSystem(
#     arch_args=arch_args,
#     vocab_size=VOCAB_SIZE,
#     pad_id=PAD_ID,
#     bos_id=BOS_ID,
#     eos_id=EOS_ID,
#     spm_model_path=str(spm_model_path),
#     source_lang=config.source_lang,
#     target_lang=config.target_lang,
#     lr_factor=config.lr_factor,
#     lr_warmup=config.lr_warmup,
#     weight_decay=config.weight_decay,
#     beam=config.beam,
#     max_len_a=config.max_len_a,
#     max_len_b=config.max_len_b,
#     len_penalty=config.len_penalty,
#     savedir=config.savedir,
# )

#  !!!!!!!!!!!!!!
config.savedir = "./checkpoints/transformer-modern"
arch_args = transformer_args

system = TranslationSystem(
    arch_args=arch_args,
    vocab_size=VOCAB_SIZE,
    pad_id=PAD_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    spm_model_path=str(spm_model_path),
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    lr_factor=config.lr_factor,
    lr_warmup=config.lr_warmup,
    weight_decay=config.weight_decay,
    beam=config.beam,
    max_len_a=config.max_len_a,
    max_len_b=config.max_len_b,
    len_penalty=config.len_penalty,
    savedir=config.savedir,
)

print(system.model)
print(
    "params:",
    f"{sum(p.numel() for p in system.parameters()):,}"
)

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

def make_trainer(config):
    savedir=Path(config.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    epoch_ckpt=ModelCheckpoint(
        dirpath=savedir,
        filename="best-{epoch:02d}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True
    )

    best_ckpt=ModelCheckpoint(
        dirpath=savedir,
        filename="best-{epoch:02d}-{val_bleu:.2f}",
        auto_insert_metric_name=False,
        monitor="val_bleu",
        mode="max",
        save_top_k=1,
    )
    lr_monitor=LearningRateMonitor(logging_interval="step")

    if config.use_wandb:
        exp_logger=WandbLogger(
            project="hw5.seq2seq.modern",
            name=savedir.name,
        )
    else:
        exp_logger=CSVLogger(
            save_dir=str(savedir/"logs"),
            name="lightning"
        )

    trainer=L.Trainer(
        max_epochs=config.max_epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=config.accum_steps,
        gradient_clip_val=config.clip_norm,
        callbacks=[epoch_ckpt, best_ckpt, lr_monitor] ,
        logger=exp_logger,
        deterministic=True,
        num_sanity_val_steps=0,
        log_every_n_steps=50
    )

    return trainer, epoch_ckpt, best_ckpt

trainer, epoch_ckpt, best_ckpt=make_trainer(config)

config.resume=str(Path(config.savedir) / "last.ckpt")
# control training
trainer.fit(
    system,
    datamodule=dm,
    ckpt_path=config.resume,
)

# def average_lighting_checkpoints(checkpoint_paths, output_path):
#     checkpoint_paths = [Path(p) for p in checkpoint_paths]
#     if not checkpoint_paths:
#         raise ValueError("no checkpoints")

#     checkpoints=[
#         torch.load(p, map_location="cpu", weights_only=False)
#         for p in checkpoint_paths
#     ]

#     state_dicts=[c["state_dict"] for c in checkpoints]
#     avg_state={}

#     for key in state_dicts[0]:
#         value=state_dicts[0][key]

#         if torch.is_floating_point(value):
#             acc=value.clone().float()
#             for sd in state_dicts[1:]:
#                 acc.add_(sd[key].float())
#             acc.div_(len(state_dicts))
#             avg_state[key]=acc.to(value.dtype)
#         else:
#             avg_state[key]=state_dicts[-1][key].clone()


#     out=checkpoints[-1]
#     out["state_dict"]=avg_state


#     torch.save(out, output_path)
#     print("saved: ", output_path)

# checkdir=Path(config.savedir)
# epoch_files=sorted(checkdir.glob("epoch-*.ckpt"))
# last_5=epoch_files[-5:]

# avg_path=checkdir / "avg_last_5.ckpt"
# average_lighting_checkpoints(last_5, avg_path)

# ckpt=torch.load(avg_path, map_location="cpu", weights_only=False)
# system.load_state_dict(ckpt["state_dict"])



# def move_to_device(obj, device):
#     if torch.is_tensor(obj):
#         return obj.to(device)
#     if isinstance(obj, dict):
#         return {k: move_to_device(v, device) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [move_to_device(v, device) for v in obj]
#     if isinstance(obj, tuple):
#         return tuple(move_to_device(v, device) for v in tuple)
#     return obj

# @torch.no_grad()
# def inference_batch(system, batch):
    