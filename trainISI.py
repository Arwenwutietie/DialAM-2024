import evaluate
import torch
from torch import nn
import os
accuracy = evaluate.load("accuracy")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

import csv

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

with open('I-S-I.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    csv_table = list(reader)

id2label = {0: "Default Inference", 1: "Default Rephrase",2:"Default Conflict"}
label2id={"Default Inference":0,"Default Rephrase":1,"Default Conflict":2}


#before_table = []
#for row in csv_table:
#    before_row = {
#        'text': (row['I1']+"=="+row['I2']),
#        'label':(label2id[row['real']])
#    }
#    before_table.append(before_row)

import pandas as pd

thedf = pd.read_csv("original-TA-YA-S.csv")

def findYA(nodeid,thedf):
    result = thedf[thedf["last"] == nodeid]
    if not result.empty:
        real_values = result["real"].values[0]
        return real_values
    else:
        return "Arguing"


df = pd.read_csv("I-S-I.csv")
df=df.drop(['Unnamed: 0','from','last','predict'],axis=1)
#df.columns = ['I1','real']
df.loc[:, 'label'] = 0
df.head(10)

for index, row in df.iterrows():
    df.loc[index, 'label'] = label2id[row['real']]
    df.loc[index, 'YA']=findYA(row['mid'],thedf)
    #if index%1000==0:
    #    print(index)

df=df.drop(['real'],axis=1)  #

text1 = []
label1 = []
text2 = []
label2 = []
text3 = []
label3 = []

index1 = 0
index2 = 0
df_len=len(df)

for index, row in df.iterrows():
    if index < df_len*(2/5):
        text1.append(row['I1'] + " -> " + row['I2'])
        label1.append(row['label'])
    elif index>df_len*(2/5) and index<df_len*(1/2):
        text2.append(row['I1'] + " -> " + row['I2'])
        label2.append(row['label'])

from datasets import Dataset, DatasetDict


dataset1 = Dataset.from_dict({"label":label1 ,"text": text1 })
dataset2 = Dataset.from_dict({"label":label2 ,"text": text2 })
dataset3 = Dataset.from_dict({"label":label3 ,"text": text3 })

dataset_dict = DatasetDict()
notuse_dataset_dict = DatasetDict()

dataset_dict["train"] = dataset1
dataset_dict["test"] = dataset2
notuse_dataset_dict["try"]=dataset3

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized = dataset_dict.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=3, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="final I-S-I debertabse",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

