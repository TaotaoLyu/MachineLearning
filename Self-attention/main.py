import math

import torch.cuda

CUDA_ID=0
VAL_RATION=0.1
DATA_DIR="Dataset/"
SEGMENT_LEN=128
BATCH_SIZE=128
N_WORKERS=10
EPOCH=200
MODEL_PATH="./model.ckpt"

def get_device():
    return f"cuda:{CUDA_ID}" if torch.cuda.is_available() else "cpu"
device=get_device()

print("Device: {}".format(device))

import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import json
from pathlib import Path

class VoiceDataset(Dataset):
    def __init__(self):
        super().__init__()
        mapping_path=Path(DATA_DIR)/"mapping.json"
        mapping=json.load(mapping_path.open())
        self.speaker2id=mapping["speaker2id"]
        self.id2speaker={}

        meta_path=Path(DATA_DIR)/"metadata.json"
        meta=json.load(meta_path.open())
        self.n_mels=meta["n_mels"]      
        self.speakers=meta["speakers"]
        self.speakers_num=len(self.speakers)
        self.data=[]

        for speaker in self.speakers:
            for utterances in self.speakers[speaker]:
                self.data.append([utterances["feature_path"],self.speaker2id[speaker]])
            self.id2speaker[self.speaker2id[speaker]]=speaker
         
    def get_speakers_num(self):
        return self.speakers_num

    def __getitem__(self, index):
        feature_path=self.data[index][0]
        speakerid=self.data[index][1]
        mel=torch.load(DATA_DIR+feature_path)

        if len(mel)>SEGMENT_LEN:
            start=random.randint(0,len(mel)-SEGMENT_LEN)
            mel=torch.FloatTensor(mel[start:start+SEGMENT_LEN])
        else:
            mel=torch.FloatTensor(mel)
        
        speakerid=torch.tensor(speakerid, dtype=torch.long)
        return mel, speakerid


    def __len__(self):
        return len(self.data)


dataset=VoiceDataset()
speaker_num=dataset.get_speakers_num()
train_len=int((1-VAL_RATION)*len(dataset))
lengths=[train_len, len(dataset)-train_len]
train_set, val_set=random_split(dataset, lengths)

from torch.nn.utils.rnn import pad_sequence
# batch ((mel,id), (mel, id))
def Collect_fn(batch):
    # mel (batch, (128?, dim))
    mel, speakerid=zip(*batch)
    mel=pad_sequence(mel,batch_first=True, padding_value=-20)
    # mel [batch, 128, 40]
    speakerid=torch.tensor(speakerid).long()

    return mel,speakerid



train_loader=DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=N_WORKERS, pin_memory=True, collate_fn=Collect_fn)
Val_loader=DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=N_WORKERS, pin_memory=True, collate_fn=Collect_fn)

import torch.nn as nn
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self, spearkers_num):
        super().__init__()
        self.prenet=nn.Linear(40, 80)

        self.self_attention_block=nn.TransformerEncoderLayer(d_model=80, dim_feedforward=256, nhead=2, batch_first=True)

        self.self_attention=nn.TransformerEncoder(self.self_attention_block, 5)

        self.pred_layer=nn.Sequential(
            nn.Linear(80,80),
            nn.ReLU(),
            nn.Linear(80,spearkers_num)
        )


    def forward(self,x):
        x=self.prenet(x)
        x=self.self_attention(x)

        x=x.mean(dim=1)

        x=self.pred_layer(x)

        return x

    # def loss(outs, labels):
    #     retrun 
        
def lr_lambda(current_step, num_warmup_steps=10, num_training_steps=EPOCH):
    # Warmup
    if current_step < num_warmup_steps:
      return float(current_step+1) / float(max(1, num_warmup_steps))
    # decadence
    progress = float(current_step - num_warmup_steps) / float(
      max(1, num_training_steps - num_warmup_steps)
    )
    return max(
      0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress))
    )

from torch.optim.lr_scheduler import LambdaLR
best_acc=0.0
def Train():
    global best_acc
    model=Classifier(spearkers_num=speaker_num).to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.0001)
    schedular=LambdaLR(optimizer=optimizer,lr_lambda=lr_lambda)

    train_loss_history=[]
    train_acc_history=[]
    val_loss_history=[]
    val_acc_history=[]
    for epoch in range(EPOCH):
        model.train()

        loss=[]
        accuracy=[]
        for data in train_loader:
            mels, lables=data
            mels, lables=mels.to(device), lables.to(device)
            optimizer.zero_grad()
            outputs=model(mels)
            batch_loss=criterion(outputs,lables)
            batch_loss.backward()
            optimizer.step()

            preds=outputs.argmax(1)
            acc=torch.mean((preds==lables).float())
            loss.append(batch_loss.cpu().item())
            accuracy.append(acc.cpu().item())
        schedular.step()
        train_loss=sum(loss)/len(loss)
        train_acc=sum(accuracy)/len(accuracy)
        print("epoch: %d Train Loss: %.2f Train Acc: %.2f"%(epoch, train_loss, train_acc))

        # do val 

        model.eval()

        loss=[]
        accuracy=[]
        for data in Val_loader:
            mels, lables=data
            mels, lables=mels.to(device), lables.to(device)
            with torch.no_grad() :
                outputs=model(mels)
            batch_loss=criterion(outputs,lables)
            preds=outputs.argmax(1)
            acc=torch.mean((preds==lables).float())
            loss.append(batch_loss.cpu().item())
            accuracy.append(acc.cpu().item())

        val_loss=sum(loss)/len(loss)
        val_acc=sum(accuracy)/len(accuracy)
        print("epoch: %d Val Loss: %.2f Val Acc: %.2f"%(epoch, val_loss, val_acc))

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        fig, axes=plt.subplots(1, 2, figsize=(10, 4))
        epochs=range(1, epoch+2)
        axes[0].plot(epochs, train_loss_history, label="train")
        axes[0].plot(epochs, val_loss_history, label="validation")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(epochs, train_acc_history, label="train")
        axes[1].plot(epochs, val_acc_history, label="validation")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig("training_metrics.png")
        plt.close(fig)
            
        if best_acc < val_acc:
            best_acc=val_acc
            torch.save(model.state_dict(),MODEL_PATH)


Train()