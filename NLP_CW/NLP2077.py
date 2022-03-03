# -*- coding: utf-8 -*-
"""jonas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HrO9_fUvOaDklVloYwmGMLWRCGZHm5Ww
"""

import os
from dpm_preprocessing import DPMProprocessed
import torch
# from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, RobertaConfig
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, AutoModelForMultipleChoice, Trainer
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import f1_score
from random import randint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dependencies
# ! python -m pip install nltk
# ! python -m pip install wordcloud
# ! python -m pip install Unidecode
# ! python -m pip install beautifulsoup4
os.environ["WANDB_DISABLED"] = "true"
os.system("python -m pip install nltk")
os.system("python -m pip install wordcloud")
os.system("python -m pip install Unidecode")
os.system("python -m pip install beautifulsoup4")

model_name = "google/bigbird-roberta-large"
#model_name = 'bert-base-uncased' 
assert model_name in ['google/bigbird-roberta-large', 'bert-base-uncased', 'google/electra-small-discriminator',
                      "microsoft/deberta-v2-xlarge"]

model_path = f'./models/pcl_{model_name}_finetuned/model/'
tokenizer_path = f'./models/pcl_{model_name}_finetuned/tokenizer/'
MAX_SEQ_LEN = 1024

WORKING_ENV = 'JONAS'  #  Can be JONAS, SERVER
assert WORKING_ENV in ['JONAS', 'SERVER']

if WORKING_ENV == 'SERVER':
    temp_model_path = f'/hy-tmp/pcl/{model_name}/'
if WORKING_ENV == 'JONAS':
    temp_model_path = f'./experiment/pcl/{model_name}/'
    temp_model_mc_path = f'./experiment/pcl/mc/{model_name}/'


class PCLDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, input_set):
        self.tokenizer = tokenizer
        self.texts = list(input_set['text'])
        self.labels = list(input_set['label'])

    def collate_fn(self, batch):
        texts = []
        labels = []

        for b in batch:
            texts.append(b['text'])
            labels.append(b['label'])

        encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LEN)
        encodings['labels'] = torch.tensor(labels)
        return encodings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {'text': self.texts[idx],
                'label': self.labels[idx]}
        return item

class PCLDatasetMC(torch.utils.data.Dataset):

    def __init__(self, tokenizer, input_set):
        self.tokenizer = tokenizer
        self.texts = list(input_set['text'])
        self.text_spans = list(input_set['text_span'])

    def collate_fn(self, batch):
        #import pdb;pdb.set_trace()
        b  = batch[0] #only works with batch size == 1
        prompt = b['text']
        choice_true = b['text_span']
        false_len = len(choice_true.split()) #the wrong choice has the same size as the right choice
        false_start = randint(0, len(prompt.split()) - false_len - 1)
        choice_false= ' '.join(prompt.split()[false_start:false_start+false_len]) #randomly crop words from the base sentence, same size as the true answer to avoid bias of size

        if randint(0,1): #randomly select which is going to be choice1 or choice2, to avoid learning everytime that choice1 is true
            choice1 = choice_true
            choice2 = choice_false
            labels = torch.tensor(0).unsqueeze(0)
        else:
            choice1 = choice_false
            choice2 = choice_true
            labels = torch.tensor(1).unsqueeze(0)

        encoding = tokenizer([prompt, prompt], [choice1, choice2], return_tensors="pt", padding=True)
        encoding['labels'] = torch.tensor(labels)
        return encoding

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {'text': self.texts[idx],
                'text_span': self.text_spans[idx]}
        return item


config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
model_mc = AutoModelForMultipleChoice.from_pretrained(model_name, config=config).to(device)
model_mc.bert = model.bert #share model

dpm_pp = DPMProprocessed('.', 'task4_test.tsv')

df_type = 'BACKTRANS'  #  Can be UNBALANCED, BACKTRANS, OVERSAMPLING
assert df_type in ['UNBALANCED', 'BACKTRANS', 'OVERSAMPLING']

if df_type == 'UNBALANCED':
    train_df_path = 'traindf.pickle'
    val_df_path = 'valdf.pickle'
if df_type == 'BACKTRANS':
    train_df_path = 'traindf_backtrans.pickle'
    val_df_path = 'valdf_backtrans.pickle'

if not os.path.isfile(train_df_path) or not os.path.isfile(val_df_path):
    train_df, val_df = dpm_pp.get_unbalanced_split(0.1)
    train_df.to_pickle('traindf.pickle')
    val_df.to_pickle('valdf.pickle')
else:
    train_df = pd.read_pickle(train_df_path)
    val_df = pd.read_pickle(val_df_path)

print("Training set length: ", len(train_df))
print("Validation set length: ", len(val_df))

par_id_val = val_df['par_id'].tolist()

train_task2 = dpm_pp.train_task2_df.drop(dpm_pp.train_task2_df[dpm_pp.train_task2_df['par_id'].map(lambda id: (id in par_id_val) )].index)

train_dataset = PCLDataset(tokenizer, train_df)
eval_dataset = PCLDataset(tokenizer, val_df)

train_dataset_MC = PCLDatasetMC(tokenizer, train_task2)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # weight_scale = len(train_df[train_df['label']==0])/len(train_df[train_df['label']==1])
        weight_scale = 1
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, weight_scale]).to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return ((loss, outputs) if return_outputs else loss)

class CustomTrainerMC(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        #import pdb;pdb.set_trace()
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        # weight_scale = len(train_df[train_df['label']==0])/len(train_df[train_df['label']==1])
        #weight_scale = 1
        #loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, weight_scale]).to(device))
        #loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return ((loss, outputs) if return_outputs else loss)


#validation_loader = DataLoader(eval_dataset)


def compute_metric_eval(arg):
    logits, labels_gold = arg[0], arg[1]
    labels_pred = np.argmax(logits, axis=1)
    return {'f1_macro': f1_score(labels_gold, labels_pred, average='macro'),
            'pcl_f1':
                classification_report(labels_gold, labels_pred, target_names=["Not PCL", "PCL"], output_dict=True)[
                    'PCL']['f1-score']}  # more metrics can be added


training_args = TrainingArguments(
    output_dir=temp_model_path,
    learning_rate=1e-6,
    logging_steps=100,
    eval_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=0.1,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model='pcl_f1'
)

training_args_mc = TrainingArguments(
    output_dir = temp_model_mc_path,
    learning_rate=1e-6,
    logging_steps=100,
    per_device_train_batch_size=1,
    num_train_epochs=0.5,
    # metric_for_best_model='pcl_f1'
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=eval_dataset.collate_fn,
    compute_metrics=compute_metric_eval,
    eval_dataset=eval_dataset
)

trainer_mc = CustomTrainerMC(
    model = model_mc,
    args = training_args_mc,
    data_collator = train_dataset_MC.collate_fn,
    train_dataset = train_dataset_MC
)

for _ in range(30):
    trainer.train()
    trainer_mc.train()

trainer.save_model(model_path)
tokenizer.save_pretrained(tokenizer_path)

train_df.to_pickle('train_df.pickle')
val_df.to_pickle('val_df.pickle')

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config).to(device)
#model_mc = AutoModelForMultipleChoice.from_pretrained(model_path, config=config).to(device)

train_df = pd.read_pickle('train_df.pickle')
val_df = pd.read_pickle('val_df.pickle')

train_dataset = PCLDataset(tokenizer, train_df)
eval_dataset = PCLDataset(tokenizer, val_df)


def predict_pcl(input, tokenizer, model, threshold=0.5):
    model.eval()
    encodings = tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LEN)
    encodings = encodings.to(device)
    output = model(**encodings)
    logits = output.logits
    preds = np.zeros(logits.shape)
    preds[preds > threshold] = 1

    return {'prediction': preds, 'confidence': logits[:, 1]}


def evaluate(model, tokenizer, data_loader, threshold=0.5):
    preds = []
    tot_labels = []
    confidences = []
    with torch.no_grad():
        for data in (data_loader):
            labels = {}
            labels['label'] = data['label']

            tweets = data['text']

            pred = predict_pcl(tweets, tokenizer, model, threshold)

            preds.append(np.array(pred['prediction'].cpu()))
            tot_labels.append(np.array(labels['label'].cpu()))
            confidences.append(np.array(pred['confidence'].cpu()))

    # with the saved predictions and labels we can compute accuracy, precision, recall and f1-score

    return preds, tot_labels, confidences


validation_loader = DataLoader(eval_dataset)

preds, tot_labels, confidences = evaluate(model, tokenizer, validation_loader)
tot_labels = np.array(tot_labels)
preds = np.array(preds)
report = classification_report(tot_labels, preds, target_names=["Not PCL", "PCL"], output_dict=True)
print(report)

print(report['accuracy'])
print(report['Not PCL']['f1-score'])
print(report['PCL']['f1-score'])

# define threshold
pcl_count_by_threshold = []
non_pcl_count_by_threshold = []
f1_by_threshold = []
for percentage in range(100):
    threshold = percentage / 100
    pcl_count = (confidences > threshold).sum()
    non_pcl_count = (confidences <= threshold).sum()
    pred = np.zeros(tot_labels.shape)
    pred[confidences > threshold] = 1
    f1_by_threshold.append(f1_score(tot_labels, pred, labels=['Non PCL', 'PCL'])['PCL'])

best_threshold = np.argmax(f1_by_threshold) / 100
"""# Test set"""

dpm_pp.load_test()
test_df = dpm_pp.test_set_df
test_df['label'] = 0
test_dataset = PCLDataset(tokenizer, test_df)

test_loader = DataLoader(test_dataset)

preds, tot_labels, confidences = evaluate(model, tokenizer, test_loader, best_threshold)
tot_labels = np.array(tot_labels)
preds = np.array(preds)
# report = classification_report(tot_labels, preds, target_names=["Not PCL","PCL"], output_dict= True)
# print(report)

# print(report['accuracy'])
# print(report['Not PCL']['f1-score'])
# print(report['PCL']['f1-score'])

# preds.shape
preds.shape

from collections import Counter

preds = preds.reshape(-1)
Counter(preds)


# helper function to save predictions to an output file
def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi]) + '\n')


labels2file([[k] for k in preds], 'task1.txt')
os.system("cat task1.txt | head -n 10")
os.system("zip submission.zip task1.txt")
