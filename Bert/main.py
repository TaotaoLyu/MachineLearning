import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForQuestionAnswering
from torch.optim import AdamW

from tqdm.auto import tqdm

device="cuda" if torch.cuda.is_available() else "cpu"

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available() :
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

same_seeds(0)


fp16_training=False

if fp16_training:
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    device=accelerator.device



model=BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data=json.load(reader)

    return data["questions"], data["paragraphs"]

train_questions, train_paragraphs=read_data("DATA/hw7_train.json")
dev_questions, dev_paragraphs=read_data("DATA/hw7_dev.json")
test_questions, test_paragraphs=read_data("DATA/hw7_test.json")


train_question_tokenized=tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_question_tokenized=tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_question_tokenized=tokenizer([test_question["question_text"] for test_question in train_questions], add_special_tokens=False)

train_paragraphs_tokenized=tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized=tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized=tokenizer(test_paragraphs, add_special_tokens=False)


class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split=split
        self.questions=questions
        self.tokenized_questions=tokenized_questions
        self.tokenized_paragraphs=tokenized_paragraphs
        self.max_question_len=40
        self.max_paragraph_len=150

        self.doc_stride=150

        self.max_seq_len=1+self.max_question_len+1+self.max_paragraph_len+1


    def __len__(self):
        return len(self.questions)


    def __getitem__(self, index):
        question=self.questions[index]
        tokenized_question=self.tokenized_questions[index]
        tokenized_paragraph=self.tokenized_paragraphs[question["paragraph_id"]]

        if self.split=="train":
            answer_start_token=tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token=tokenized_paragraph.char_to_token(question["answer_end"])

            mid=(answer_start_token+answer_end_token)//2
            paragraph_start=max(0, min(mid-self.max_paragraph_len//2, len(tokenized_paragraph)-self.max_paragraph_len))
            paragraph_end=paragraph_start+self.max_paragraph_len

            input_ids_question=[101]+tokenized_question.ids[:self.max_question_len]+[102]
            input_ids_paragraph=tokenized_paragraph.ids[paragraph_start:paragraph_end]+[102]

            answer_start_token+=len(input_ids_question)-paragraph_start
            answer_end_token+=len(input_ids_question)-paragraph_start

            input_ids, token_type_ids, attention_mask=self.padding(input_ids_question, input_ids_paragraph)

            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        else:
            input_ids_list, token_type_ids_list, attention_mask_list=[], [], []
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                input_ids_question=[101]+tokenized_question.ids[:self.max_question_len]+[102]
                input_ids_paragraph=tokenized_paragraph.ids[i:i+self.max_paragraph_len]+[102]

                input_ids, token_type_ids, attention_mask=self.padding(input_ids_question, input_ids_paragraph)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)
                

    def padding(self, input__ids_question, input_ids_paragraph):
        padding_len=self.max_seq_len-len(input__ids_question)-len(input_ids_paragraph)
        input_ids=input__ids_question+input_ids_paragraph+[0]*padding_len
        token_type_ids=[0]*len(input__ids_question)+[1]*len(input_ids_paragraph)+[0]*padding_len

        attention_mask=[1]*(len(input__ids_question)+len(input_ids_paragraph))+[0]*padding_len


        return input_ids, token_type_ids, attention_mask



train_set=QA_Dataset("train", train_questions, train_question_tokenized, train_paragraphs_tokenized)
dev_set=QA_Dataset("dev", dev_questions, dev_question_tokenized, dev_paragraphs_tokenized)
test_set=QA_Dataset("test", test_questions, test_question_tokenized, test_paragraphs_tokenized)

train_batch_size=16

train_loader=DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader=DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader=DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)


# bug
def evaluate(data, output):
    answer=''
    max_prob=float('-inf')
    num_of_windows=data[0].shape[1]
    for k in range(num_of_windows):
        start_prob, start_index=torch.max(output.start_logits[k], dim=0)
        end_prob, end_index=torch.max(output.end_logits[k], dim=0)

        prob=start_prob+end_prob

        if prob>max_prob:
            max_prob=prob
            answer=tokenizer.decode(data[0][0][k][start_index:end_index+1])


    return answer.replace(' ', '')


num_epoch=1
validation=True
logging_step=100
learning_rate=1e-4
optimizer=AdamW(model.parameters(), lr=learning_rate)

if fp16_training:
    model, optimizer, train_loader=accelerator.prepare(model, optimizer, train_loader)

model.train()
print("Start Training ...")

for epoch in range(num_epoch):
    step=0
    train_loss=train_acc=0

    for data in tqdm(train_loader):
        data=[i.to(device) for i in data]

        output=model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

        start_index=torch.argmax(output.start_logits, dim=1)
        end_index=torch.argmax(output.end_logits, dim=1)

        train_acc+=((start_index==data[3])&(end_index==data[4])).float().mean()
        train_loss+=output.loss.item()

        if fp16_training:
            accelerator.backward(output.loss)
        else:
            output.loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        step+=1


        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
            train_loss = train_acc = 0


    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc=0
            for i, data in enumerate(tqdm(dev_loader)):
                output=model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                                attention_mask=data[2].squeeze(dim=0).to(device))

                dev_acc+=evaluate(data, output)==dev_questions[i]["answer_text"]

            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
        model.train()


print("Saving Model ...")
model_save_dir="saved_model"
model.save_pretrained(model_save_dir)
