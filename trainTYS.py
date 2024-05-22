import evaluate
import torch
import numpy as np
from torch import nn
import os
import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer
from datasets import Dataset, DatasetDict

accuracy = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

id2label = {0: "Asserting", 1: "Pure Questioning", 2: "Challenging", 3: "Assertive Questioning",
            4: "Rhetorical Questioning", 5: "Agreeing", 6: "Default Illocuting", 7: "Arguing", 8: "Restating",
            9: "Disagreeing"}

# id2label = {    0: "Default Inference",
#                1: "Default Rephrase",
#                2: "Default Conflict"    }
label2id = {y: x for x, y in id2label.items()}

print(label2id)

csv_file = 'data_tables/LTL_Y_ISI.csv'
df = pd.read_csv(csv_file)

text, label = [[], []], [[], []]

df_len = len(df)

prompt = "The above relation is classified as [mask]."

for index, row in df.iterrows():
    if index < df_len * (4 / 5):
        text[0].append('[CLS]' + row['L1'] + '[SEP]' + row['S'] + '[SEP]' + row['L2'] + '[SEP]')  # +'[SEP]'
        label[0].append(label2id[row['YA']])
    elif index > df_len * (4 / 5) and index < df_len:
        text[1].append('[CLS]' + row['L1'] + '[SEP]' + row['S'] + '[SEP]' + row['L2'] + '[SEP]')
        label[1].append(label2id[row['YA']])

dataset1 = Dataset.from_dict({"label": label[0], "text": text[0]})
dataset2 = Dataset.from_dict({"label": label[1], "text": text[1]})

dataset_dict = DatasetDict()

dataset_dict["train"] = dataset1
dataset_dict["test"] = dataset2

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)


tokenized = dataset_dict.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=len(id2label), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="finalLSL(2-epoch)",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
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

# torch.cuda.set_device(5)
# trainer = trainer.cuda()
# tokenized["train"].cuda()
# tokenized["test"].cuda()


trainer.train()
trainer.evaluate()