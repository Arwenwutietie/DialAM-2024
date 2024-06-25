# KnowComp at DialAM-2024

This is the official code repository for the [DialAM-2024](https://dialam.arg.tech/) paper: KnowComp at DialAM-2024: Fine-tuning Pre-trained Language Models
for Dialogical Argument Mining with Inference Anchoring Theory.
Our system officially ranks No.2th on the ILO - General evaluation and achieves an F1 score of 78.90.

![Overview](https://github.com/Arwenwutietie/DialAM-2024/blob/main/Pipeline%20(1).jpg)

## 1. Data Preparation

Please download the original dataset
from [DialAM-2024](https://dialam.arg.tech/index.php).
For convenience, our model uses pre-processed data in folders corresponding to different subtasks.
You can also download our dataset [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ywufe_connect_ust_hk/Eo8JZiEcd9ZApNU5JJ22ubEBgnWkkLQTQ0CRZFq5cEeHaA?e=XYMgKG).

## 2. Required Packages

Required packages are listed in `requirements.txt`. Install them by running:

```bash
pip install -r requirements.txt
```

## 3. Training
Our code is divided into three seperate folders(trainISI, trainLYI, trainTYS) corresponding to the three subtasks in DialAM-2024.
The default pretrained model is microsoft/deberta-v3-base. You can train the model with the pre-processed data provided in each folder.
The gpt version of the code is also attached for your reference in an independent folder called gpt version.


## 4. Citing this work

Please use the bibtex below for citing our paper:

```bibtex
@inproceedings{KnowCompDialAM-2024,
  author       = {Yuetong Wu and
                  Yukai Zhou and
                  Baixuan Xu and
                  Weiqi Wang and
                  Yangqiu Song},
  title        = {KnowComp at DialAM-2024: Fine-tuning Pre-trained Language Models 
                  for Dialogical Argument Mining with Inference Anchoring Theory},
  year         = {2024},
  booktitle    = {Proceedings of the 11th Workshop on Argument Mining, {DialAM} 2024}
}
```
