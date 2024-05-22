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

with open('new_table.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    csv_table = list(reader)

id2label = {0: "Asserting", 1: "Pure Questioning", 2: "Challenging", 3: "Assertive Questioning",
            4: "Rhetorical Questioning", 5: "Agreeing", 6: "Default Illocuting", 7: "Arguing", 8: "Restating",
            9: "Disagreeing"}
label2id = {"Asserting": 0, "Pure Questioning": 1, "Challenging": 2, "Assertive Questioning": 3,
            "Rhetorical Questioning": 4, "Agreeing": 5, "Default Illocuting": 6, "Arguing": 7, "Restating": 8,
            "Disagreeing": 9}

before_table = []
for row in csv_table:
    before_row = {
        'text': (row['L'] + "==" + row['I']),
        'label': (label2id[row['real']])
    }
    before_table.append(before_row)

import pandas as pd

df = pd.read_csv("new_table.csv")
df = df.drop(['from', 'mid', 'last', 'predict'], axis=1)
# df.columns = ['text','real']
df.loc[:, 'label'] = 0
df.head(10)

for index, row in df.iterrows():
    df.loc[index, 'label'] = label2id[row['real']]

df = df.drop(['real'], axis=1)  #
df.head(2)

text1 = []
label1 = []
text2 = []
label2 = []
text3 = []
label3 = []

"""for index,row in df.iterrows():
    if index<16000:
        #if row["label"]==0:
        text1.append(row['text'])
        label1.append(row['label'])
    else:
        #if len(label2)<1250:
        text2.append(row['text'])
        label2.append(row['label'])"""

index1 = 0
index2 = 0
'''
prefix="Illocutionary relations include 0: Asserting, 1: Pure Questioning,2:Challenging,3:Assertive Questioning,4:Rhetorical Questioning,5:Agreeing,6:Default Illocuting,7:Arguing,8:Restating,9:Disagreeing.0: Asserting - Making a statement or expressing a belief with confidence.1: Pure Questioning - Asking a question to seek information or clarification.2: Challenging - Expressing doubt or disagreement, questioning the validity or correctness of something.3: Assertive Questioning - Asking a question in a confident or assertive manner, often implying a certain expectation or assumption.4: Rhetorical Questioning - Asking a question that is not meant to be answered, but rather to make a point or emphasize a statement.5: Agreeing - Expressing consent or alignment with a statement or opinion.6: Default Illocuting - Default or general illocutionary act with no specific categorization.7: Arguing - Engaging in a dispute or presenting reasons and evidence to support or challenge a claim.8: Restating - Expressing the same idea or information using different words or phrasing.9: Disagreeing - Expressing a difference of opinion or belief, indicating a lack of agreement with a statement or viewpoint. The illocutionary relation between the two sentences is [mask]."
'''
prefix = "Illocutionary relations include 0: Asserting, 1: Pure Questioning,2:Challenging,3:Assertive Questioning,4:Rhetorical Questioning,5:Agreeing,6:Default Illocuting,7:Arguing,8:Restating,9:Disagreeing.The illocutionary relation between the two sentences is [mask]."
for index, row in df.iterrows():
    if index < 16000:
        text1.append('[CLS]' + prefix + '[SEP]' + row['L'] + '[SEP]' + row['I'])
        label1.append(row['label'])

    else:
        text2.append('[CLS]' + prefix + '[SEP]' + row['L'] + '[SEP]' + row['I'])
        label2.append(row['label'])

from datasets import Dataset, DatasetDict

dataset1 = Dataset.from_dict({"label": label1, "text": text1})
dataset2 = Dataset.from_dict({"label": label2, "text": text2})
dataset3 = Dataset.from_dict({"label": label3, "text": text3})

dataset_dict = DatasetDict()
notuse_dataset_dict = DatasetDict()

dataset_dict["train"] = dataset1
dataset_dict["test"] = dataset2
notuse_dataset_dict["try"] = dataset3

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


id2label = {0: "Asserting", 1: "Pure Questioning", 2: "Challenging", 3: "Assertive Questioning",
            4: "Rhetorical Questioning", 5: "Agreeing", 6: "Default Illocuting", 7: "Arguing", 8: "Restating",
            9: "Disagreeing"}
label2id = {"Asserting": 0, "Pure Questioning": 1, "Challenging": 2, "Assertive Questioning": 3,
            "Rhetorical Questioning": 4, "Agreeing": 5, "Default Illocuting": 6, "Arguing": 7, "Restating": 8,
            "Disagreeing": 9}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=10, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="prompt_model(2-epochs,full asserting)",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
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

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.device_count() > 1:
#   model = nn.DataParallel(model)  # 使用DataParallel模块进行多GPU并行
# model.to(device)

# model= torch.nn.DataParallel(model, device_ids=[0,1])

trainer.train()
trainer.evaluate()