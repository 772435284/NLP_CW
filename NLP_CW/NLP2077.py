# -*- coding: utf-8 -*-
"""jonas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HrO9_fUvOaDklVloYwmGMLWRCGZHm5Ww
"""

import os

# dependencies
# ! python -m pip install nltk
# ! python -m pip install wordcloud
# ! python -m pip install Unidecode
# ! python -m pip install beautifulsoup4
import pandas

os.environ["WANDB_DISABLED"] = "true"
os.system("python -m pip install nltk")
os.system("python -m pip install wordcloud")
os.system("python -m pip install Unidecode")
os.system("python -m pip install beautifulsoup4")

from dpm_preprocessing import DPMProprocessed
import torch
# from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, RobertaConfig
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, \
    AutoModelForPreTraining, AutoModel, BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import f1_score
from random import randint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model_name = "microsoft/deberta-v2-xlarge"
model_name = 'bert-base-uncased'
assert model_name in ['roberta-base', 'bert-base-uncased', 'google/electra-small-discriminator',
                      "microsoft/deberta-v2-xlarge"]

model_path = f'./models/pcl_{model_name}_finetuned/model/'
tokenizer_path = f'./models/pcl_{model_name}_finetuned/tokenizer/'
MAX_SEQ_LEN = 256

WORKING_ENV = 'JONAS'  #  Can be JONAS, SERVER
assert WORKING_ENV in ['JONAS', 'SERVER']

if WORKING_ENV == 'SERVER':
    temp_model_path = f'/hy-tmp/pcl/{model_name}/'
    temp_model_mc_path = f'./experiment/pcl/mc/{model_name}/'
if WORKING_ENV == 'JONAS':
    temp_model_path = f'./experiment/pcl/{model_name}/'
    temp_model_mc_path = f'./experiment/pcl/mc/{model_name}/'


class PCLDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, input_set):
        self.tokenizer = tokenizer
        self.texts = list(input_set['text'])
        self.labels = list(input_set['label'])
        self.categories = list(input_set['categories'])

    def collate_fn(self, batch):
        texts = []
        labels = []
        categories = []
        for b in batch:
            texts.append(b['text'])
            labels.append(b['label'])
            categories.append(b['category'])
        encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LEN)
        encodings['labels'] = torch.tensor(labels)
        encodings['categories'] = torch.tensor(categories)

        return encodings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {'text': self.texts[idx],
                'label': self.labels[idx],
                'category': self.categories[idx]}
        return item


config = AutoConfig.from_pretrained(model_name)
# config_mc = AutoConfig.from_pretrained(model_name)
# config_mc.num_labels = 7
# config_mc.problem_type = 'multi_label_classification'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForPreTraining.from_pretrained(model_name, config=config).to(device)
# model_mc = AutoModelForSequenceClassification.from_pretrained(model_name, config=config_mc).to(device)
# model_mc.bert = model.bert  # share model

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


def aggregate_label(rows):
    label = np.zeros(7)
    for index, row in rows.iterrows():
        label = np.logical_or(label, row['label'])
    label = label.astype('float32')
    d = {'categories': [label]}
    # ser = pd.Series([label], copy=False)
    return pd.DataFrame(data=d)


def join_with_categories(df):
    task2_df = dpm_pp.train_task2_df.copy()
    # task2_df_grouped = task2_df.groupby('par_id')[['par_id', 'label']].apply(aggregate_label)
    task2_df_grouped = task2_df.groupby('par_id')[['label']].apply(aggregate_label)

    return pd.merge(df, task2_df_grouped, on=['par_id'], how='left')

def fillna(df):
    df_copy = pd.DataFrame()
    for index, row in df.iterrows():
        if row.isna().any():
            temp_df = pd.DataFrame({"par_id": row["par_id"],
                                    "art_id": row["art_id"],
                                    "keyword": row["keyword"],
                                    "country": row["country"],
                                    "text": row["text"],
                                    "label": row["label"],
                                    "orig_label": row["orig_label"],
                                    "lenght": row[ "lenght"],
                                    "categories": [np.zeros(7)]})
            df_copy = df_copy.append(temp_df)
        else:
            df_copy = df_copy.append(row)
    return df_copy


if not os.path.isfile("traindf_with_categories.pickle") or not os.path.isfile("valdf_with_categories.pickle"):
    train_df = fillna(join_with_categories(train_df))
    val_df = fillna(join_with_categories(val_df))
    train_df.to_pickle("traindf_with_categories.pickle")
    val_df.to_pickle("valdf_with_categories.pickle")
else:
    train_df = pd.read_pickle("traindf_with_categories.pickle")
    val_df = pd.read_pickle("valdf_with_categories.pickle")

train_dataset = PCLDataset(tokenizer, train_df)
eval_dataset = PCLDataset(tokenizer, val_df)


# train_dataset_MC = PCLDatasetMC(tokenizer, dpm_pp.train_task2_df)
# train_dataset_MC.drop(train_dataset_MC[train_dataset_MC['par_id'].map(lambda id: id in par_id_val)])


# class CustomTrainerMC(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.logits
#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
#         return ((loss, outputs) if return_outputs else loss)


# training_args_mc = TrainingArguments(
#     output_dir=temp_model_mc_path,
#     learning_rate=1e-6,
#     logging_steps=100,
#     per_device_train_batch_size=1,
#     num_train_epochs=0.1,
#     # metric_for_best_model='pcl_f1'
# )

# trainer_mc = CustomTrainerMC(
#     model=model_mc,
#     args=training_args_mc,
#     # data_collator=train_dataset_MC.collate_fn,
#     # train_dataset=train_dataset_MC
# )

''' Custom model '''


# class BERT_hate_speech(BertPreTrainedModel):
#
#     def __init__(self, config):
#         super().__init__(config)
#
#         # BERT Model
#         self.bert = BertModel(config)
#
#         # Task A
#         self.projection_a = torch.nn.Sequential(torch.nn.Dropout(0.2),
#                                                 torch.nn.Linear(config.hidden_size, 2))
#
#         # Task B
#         # TBA
#
#         # Task C
#         # TBA
#
#         self.init_weights()
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             inputs_embeds=None,
#             labels=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         # Logits A
#         logits_a = self.projection_a(outputs[1])
#
#         return logits_a
#
#
# model = BERT_hate_speech.from_pretrained("bert-base-cased")


class MultiHeadPretrainedModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # BERT Model
        self.model = BertModel(config)

        # pcl classification
        self.projection_cls = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                                  torch.nn.Linear(config.hidden_size, 2))

        # Head0
        self.projection_0 = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                                torch.nn.Linear(config.hidden_size, 2))

        # Head1
        self.projection_1 = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                                torch.nn.Linear(config.hidden_size, 2))
        # Head2
        self.projection_2 = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                                torch.nn.Linear(config.hidden_size, 2))

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pcl
        logits_cls = self.projection_cls(outputs[1])

        # Logits 0
        logits_0 = self.projection_0(outputs[1])

        # Logits 0
        logits_1 = self.projection_1(outputs[1])

        # Logits 0
        logits_2 = self.projection_2(outputs[1])

        return logits_cls, logits_0, logits_1, logits_2


model = MultiHeadPretrainedModel.from_pretrained(model_name)

losses = []
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        categories = inputs.get('categories')
        categories = categories.detach().cpu().numpy()
        categories_0 = np.logical_or(categories[:, 0], categories[:, 1])
        categories_1 = np.logical_or(categories[:, 2], categories[:, 3])
        categories_2 = np.logical_or(categories[:, 4], categories[:, 5], categories[:, 6])
        # forward pass
        inputs.pop('categories', None)
        logits_cls, logits_0, logits_1, logits_2 = model(**inputs)
        # logits = outputs.logits
        # weight_scale = len(train_df[train_df['label']==0])/len(train_df[train_df['label']==1])
        weight_scale = 1
        loss_cls = nn.CrossEntropyLoss()
        loss_0 = nn.CrossEntropyLoss()
        loss_1 = nn.CrossEntropyLoss()
        loss_2 = nn.CrossEntropyLoss()

        alpha = 0.1

        loss = loss_cls(logits_cls.view(-1, self.model.config.num_labels), labels.view(-1)) \
               + alpha * loss_0(logits_0.view(-1, self.model.config.num_labels), labels.view(-1)) \
               + alpha * loss_1(logits_1.view(-1, self.model.config.num_labels), labels.view(-1)) \
               + alpha * loss_2(logits_2.view(-1, self.model.config.num_labels), labels.view(-1))
        output = [loss, logits_cls]
        losses.append(logits_cls)
        return (loss, output) if return_outputs else loss


# validation_loader = DataLoader(eval_dataset)


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
    eval_steps=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model='pcl_f1'
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=eval_dataset.collate_fn,
    compute_metrics=compute_metric_eval,
    eval_dataset=eval_dataset
)
trainer.train()

trainer.save_model(model_path)
tokenizer.save_pretrained(tokenizer_path)

train_df.to_pickle('train_df.pickle')
val_df.to_pickle('val_df.pickle')

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = MultiHeadPretrainedModel.from_pretrained(model_path, config=config).to(device)

train_df = pd.read_pickle('train_df.pickle')
val_df = pd.read_pickle('val_df.pickle')

train_dataset = PCLDataset(tokenizer, train_df)
eval_dataset = PCLDataset(tokenizer, val_df)


def predict_pcl(input, tokenizer, model, threshold=0.5):
    model.eval()
    encodings = tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=256)
    encodings = encodings.to(device)
    output = model(**encodings)
    logits = output.logits
    prob = nn.functional.softmax(logits)[:, 1].cpu()
    preds = np.array([int(prob > threshold)])
    return {'prediction': preds, 'confidence': prob}


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

            preds.append(np.array(pred['prediction']))
            tot_labels.append(np.array(labels['label'].cpu()))
            confidences.append(np.array(pred['confidence'].cpu()))

    # with the saved predictions and labels we can compute accuracy, precision, recall and f1-score

    return preds, tot_labels, confidences


validation_loader = DataLoader(eval_dataset)

preds, tot_labels, confidences = evaluate(model, tokenizer, validation_loader)
tot_labels = np.array(tot_labels)
preds = np.array(preds)
confidences = np.array(confidences)
report = classification_report(tot_labels, preds, target_names=["Not PCL", "PCL"], output_dict=True)
print(report)

print(report['accuracy'])
print(report['Not PCL']['f1-score'])
print(report['PCL']['f1-score'])

# define threshold
pcl_count_by_threshold = []
non_pcl_count_by_threshold = []
f1_by_threshold = []
precision_by_threshold = []
recall_by_threshold = []
for percentage in range(100):
    threshold = percentage / 100
    pcl_count = (confidences > threshold).sum()
    non_pcl_count = (confidences <= threshold).sum()
    pred = np.zeros(tot_labels.shape)
    pred[confidences > threshold] = 1
    pcl_count_by_threshold.append(pcl_count)
    non_pcl_count_by_threshold.append(non_pcl_count)
    f1_by_threshold.append(
        classification_report(tot_labels, pred, target_names=["Not PCL", "PCL"], output_dict=True)['PCL']['f1-score'])
    precision_by_threshold.append(
        classification_report(tot_labels, pred, target_names=["Not PCL", "PCL"], output_dict=True)['PCL']['precision'])
    recall_by_threshold.append(
        classification_report(tot_labels, pred, target_names=["Not PCL", "PCL"], output_dict=True)['PCL']['recall'])

best_threshold = np.argmax(f1_by_threshold) / 100

# best_threshold
np.max(f1_by_threshold)

import matplotlib.pyplot as plt

x = np.arange(0, 100)
l1 = plt.plot(x, pcl_count_by_threshold, 'r', label='pcl_count_by_threshold')
l2 = plt.plot(x, non_pcl_count_by_threshold, 'g', label='non_pcl_count_by_threshold')

# plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')
plt.title('Frequency by threshold')
plt.xlabel('threshold')
plt.ylabel('column')
plt.legend()
plt.show()

plt.clf()
l3 = plt.plot(x, f1_by_threshold, 'r', label='f1_by_threshold')
l4 = plt.plot(x, precision_by_threshold, 'g', label='precision_by_threshold')
l5 = plt.plot(x, recall_by_threshold, 'b', label='recall_by_threshold')
plt.title('Metrics by threshold')
plt.xlabel('threshold')
plt.ylabel('column')
plt.legend()
plt.show()

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
os.system("!cat task1.txt | head -n 10")
os.system("!zip submission.zip task1.txt")
