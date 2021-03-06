# -*- coding: utf-8 -*-
"""NLP_CW.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z6vBzEf1jnmZW4tRse0_0AUS3EwBHpko

# Patronizing and Condescending Language Detection
"""

import sys,os
#dependencies
os.system('pip list')
os.system('python -m pip install nltk')
os.system('python -m pip install wordcloud')
os.system('python -m pip install Unidecode')
os.system('python -m pip install beautifulsoup4')

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from dont_patronize_me import DontPatronizeMe
import random
from wordcloud import WordCloud
from transformers import BertTokenizer
from transformers import BertPreTrainedModel, BertModel
from transformers import Trainer, TrainingArguments
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm
import os
from collections import Counter
from urllib import request
#from DPM_preprocessing_over_sampling2_train_test_80_20 import DPM_preprocessing
from dpm_preprocessing import DPMProprocessed

sys.path.append('.')

os.environ["WANDB_DISABLED"] = "true"

# check gpu
# check which gpu we're using
os.system('nvidia-smi')
cuda_available = torch.cuda.is_available()

if cuda_available:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print('Cuda available? ',cuda_available)

# helper function to save predictions to an output file
def labels2file(p, outf_path):
	with open(outf_path,'w') as outf:
		for pi in p:
			outf.write(','.join([str(k) for k in pi])+'\n')

"""# 1.1 Data analysis of the training data"""

# dpm = DontPatronizeMe('.', 'task4_test.tsv')

dpm_pp = DPMProprocessed(os.getcwd(), 'task4_test.tsv')


# dpm.load_task1()
# dpm.load_task2(return_one_hot=True)
df = dpm_pp.train_task1_df
# df['lenght'] = df['text'].apply(count_words)
# hist = df['lenght'].hist(by=df['label'], bins = 50, alpha = 0.5)
#hist = df['lenght'].hist(by=df['orig_label'], bins = 10, alpha = 0.5)
#hist = df['country'].hist(by=df['label'])
#hist = df['keyword'].hist(by=df['orig_label'])
# plt.savefig('histo.jpg', dpi=500)
# print(df.shape)


#Preprocessing and over-sampling

hist = df['lenght'].hist(by=df['label'], bins = 50, alpha = 0.5)
#hist = df['lenght'].hist(by=df['orig_label'], bins = 10, alpha = 0.5)
#hist = df['country'].hist(by=df['label'])
#hist = df['keyword'].hist(by=df['orig_label'])
plt.savefig('histo.jpg', dpi=500)



if not os.path.isfile('traindf.pickle') or not os.path.isfile('valdf.pickle'):
  train_df, val_df = dpm_pp.get_oversampled_split()
  train_df.to_pickle('traindf.pickle')
  val_df.to_pickle('valdf.pickle')
else:
  train_df = pd.read_pickle('traindf.pickle')
  val_df = pd.read_pickle('valdf.pickle')

total_df = pd.concat([train_df, val_df])
print(train_df.shape)
print(val_df.shape)
print(total_df.shape)

def generate_cloud(label,label_type,df):

    if label_type == "label ":
        text = df[df['label'] == label]['text'].values
    else:
        text = df[df['orig_label'] == label]['text'].values
    
    wordcloud = WordCloud().generate(str(text))
    
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

generate_cloud(0,"label ",dpm_pp.train_task1_df)

generate_cloud(1,"label ",dpm_pp.train_task1_df)

"""# 1.2 Qualitative assessment of the dataset"""

# To be done

"""# 2 Modelling

## 2.1 Load training, validation and test set
"""

trids = pd.read_csv('train_semeval_parids-labels.csv')
teids = pd.read_csv('dev_semeval_parids-labels.csv')
trids.head()

trids.par_id = trids.par_id.astype(str)
teids.par_id = teids.par_id.astype(str)
print("Original Training Length: ",len(trids))
print("Original Test Length: ",len(teids))

train_rows = [] # will contain par_id, label and text
val_rows = []
for idx in range(len(trids)):  
  parid = trids.par_id[idx]
  #print(parid)
  # select row from original dataset to retrieve `text` and binary label

  train_row = train_df[train_df.par_id==parid]
  for index, row in train_row.iterrows():
    text = train_row.text.values[0]
    label = train_row.label.values[0]
    train_rows.append({
      'par_id':parid,
      'text':text,
      'label':label
    })
#   print(train_row)

  val_row = val_df[val_df.par_id==parid]
  for index, row in val_row.iterrows():
    text = val_row.text.values[0]
    label = val_row.label.values[0]
    val_rows.append({
      'par_id':parid,
      'text':text,
      'label':label
    })
#   print(val_row)

    
trdf1 = pd.DataFrame(train_rows)
valdf1 = pd.DataFrame(val_rows)
# from sklearn.model_selection import train_test_split
# trdf1, valdf1 = train_test_split(trdf1, test_size=0.1)

#over-sampling
print(trdf1.shape)
print(valdf1.shape)

rows = [] # will contain par_id, label and text
for idx in range(len(teids)):  
  parid = teids.par_id[idx]
  #print(parid)
  # select row from original dataset
  text = dpm_pp.train_task1_df.loc[dpm_pp.train_task1_df.par_id == parid].text.values[0]
  label = dpm_pp.train_task1_df.loc[dpm_pp.train_task1_df.par_id == parid].label.values[0]
  rows.append({
      'par_id':parid,
      'text':text,
      'label':label
  })

tedf1 = pd.DataFrame(rows)

# downsample negative instances
# Training
# pcldf = trdf1[trdf1.label==1]
# npos = len(pcldf)
# training_set1 = pd.concat([pcldf,trdf1[trdf1.label==0][:npos*2]])
training_set1 = trdf1
# Validation
# pcldf = valdf1[valdf1.label==1]
# npos = len(pcldf)
# validation_set1 = pd.concat([pcldf,valdf1[valdf1.label==0][:npos*2]])
validation_set1 = valdf1
# Testing
# pcldf = tedf1[tedf1.label==1]
# npos = len(pcldf)
# test_set1 = pd.concat([pcldf,tedf1[tedf1.label==0][:npos*2]])
test_set1 = tedf1
print("Training set length: ",len(training_set1))
print("Validation set length: ",len(validation_set1))
print("Testing set length: ",len(test_set1))
print(training_set1.shape)
training_set1

class DpmDataset(torch.utils.data.Dataset):

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

        #The maximum sequence size for BERT is 512 but here the tokenizer truncate sentences longer than 128 tokens.  
        # We also pad shorter sentences to a length of 128 tokens
        encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
        encodings['label'] =  torch.tensor(labels)
        
        return encodings
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
       
        item = {'text': self.texts[idx],
                'label': self.labels[idx]}
        return item

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

train_dataset = DpmDataset(tokenizer, training_set1)

# Print some example in training set
batch = [sample for sample in train_dataset]

encodings = train_dataset.collate_fn(batch[:10])

for key, value in encodings.items():
  print(f"{key}: {value.numpy().tolist()}")

"""## 2.2 Construct Model"""

class pcl_detection(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        # BERT Model
        self.bert = BertModel(config)
        
        # Task A
        self.projection_a = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                                torch.nn.Linear(config.hidden_size, 2))
        
        # Task B
        # TBA
        
        # Task C
        # TBA
        
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
 
        outputs = self.bert(
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

        # Logits A
        logits_a = self.projection_a(outputs[1])
        
        return logits_a

# Define Loss function
class Trainer_pcl_detection(Trainer):
    def compute_loss(self, model, inputs):
        labels = {}
        labels['label'] = inputs.pop('label')

        outputs = model(**inputs)

        loss_task_a = nn.CrossEntropyLoss()
        labels = labels['label']
        loss = loss_task_a(outputs.view(-1, 2), labels.view(-1))
        
        return loss

"""## 2.3 Train"""

def main():
    
    #model = BERT_hate_speech.from_pretrained(model_name)\
    model = pcl_detection.from_pretrained("bert-base-cased")
    #call our custom BERT model and pass as parameter the name of an available pretrained model    
    training_args = TrainingArguments(
        output_dir='./experiment/hate_speech',
        learning_rate = 0.0001,
        logging_steps= 100,
        per_device_train_batch_size=8,
        num_train_epochs = 10,
    )
    trainer = Trainer_pcl_detection(
        model=model,                         
        args=training_args,                 
        train_dataset=train_dataset,                   
        data_collator=train_dataset.collate_fn
    )
    trainer.train()

    trainer.save_model('./models/pcl_bert_finetuned/')

main()

"""## 2.4 Validation"""

def predict_pcl(input, tokenizer, model): 
  model.eval()
  encodings = tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=256)
  encodings.to(device)
  output = model(**encodings)
  preds = torch.max(output, 1)

  return {'prediction':preds[1], 'confidence':preds[0]}

def evaluate(model, tokenizer, data_loader):

  total_count = 0
  correct_count = 0 

  preds = []
  tot_labels = []

  with torch.no_grad():
    for data in tqdm(data_loader): 

      labels = {}
      labels['label'] = data['label']

      tweets = data['text']

      pred = predict_pcl(tweets, tokenizer, model)

      preds.append(np.array(pred['prediction'].cpu()))
      tot_labels.append(np.array(labels['label'].cpu()))

  # with the saved predictions and labels we can compute accuracy, precision, recall and f1-score
  

  return preds, tot_labels

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#your saved model name here
model_name = './models/pcl_bert_finetuned/' 
model = pcl_detection.from_pretrained(model_name)
model.to(device)

###########################
# Validation set
###########################

val_dataset = DpmDataset(tokenizer, validation_set1)
# we don't batch our test set unless it's too big
test_loader = DataLoader(val_dataset)

preds, tot_labels = evaluate(model, tokenizer, test_loader)
tot_labels = np.array(tot_labels)
preds = np.array(preds)
report = classification_report(tot_labels, preds, target_names=["Not PCL","PCL"], output_dict= True)
print(report)

print(report['accuracy'])
print(report['Not PCL']['f1-score'])
print(report['PCL']['f1-score'])

"""## 2.5 Test on provided test set"""

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#your saved model name here
model_name = './models/pcl_bert_finetuned/' 
model = pcl_detection.from_pretrained(model_name)
model.to(device)
###########################
# Test set
###########################

test_dataset = DpmDataset(tokenizer, tedf1)
# we don't batch our test set unless it's too big
test_loader = DataLoader(test_dataset)

preds, tot_labels = evaluate(model, tokenizer, test_loader)
tot_labels = np.array(tot_labels)
preds = np.array(preds)
report = classification_report(tot_labels, preds, target_names=["Not PCL","PCL"], output_dict= True)
print(report)

print(report['accuracy'])
print(report['Not PCL']['f1-score'])
print(report['PCL']['f1-score'])

"""## 2.6 Test on Competition Test Set and Upload result"""

dpm_pp.load_test()

# Add dummy labels
dpm_pp.test_set_df['label'] = 0
dpm_pp.test_set_df

def predict_pcl(input, tokenizer, model): 
  model.eval()
  encodings = tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=256)
  encodings.to(device)
  output = model(**encodings)
  preds = torch.max(output, 1)

  return {'prediction':preds[1], 'confidence':preds[0]}

def evaluate(model, tokenizer, data_loader):

  total_count = 0
  correct_count = 0 

  preds = []
  tot_labels = []

  with torch.no_grad():
    for data in tqdm(data_loader): 

      tweets = data['text']

      pred = predict_pcl(tweets, tokenizer, model)

      preds.append(pred['prediction'].item())

  return preds

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#your saved model name here
model_name = './models/pcl_bert_finetuned/' 
model = pcl_detection.from_pretrained(model_name)
model.to(device)

###########################
# Competition Test set
###########################

test_dataset = DpmDataset(tokenizer, dpm_pp.test_set_df)
# we don't batch our test set unless it's too big
test_loader = DataLoader(test_dataset)

preds = evaluate(model, tokenizer, test_loader)

preds = np.array(preds)
preds = preds.reshape(-1)

Counter(preds)

labels2file([[k] for k in preds], 'task1.txt')

# Generate pseduo output for task 2 for upload
# random predictions for task 2
preds_task2 = [[random.choice([0,1]) for k in range(7)] for k in range(0,len(dpm_pp.test_set_df))]
labels2file(preds_task2, 'task2.txt')


os.system('cat task1.txt | head -n 10')
# os.system('cat task2.txt | head -n 10')
# os.system('zip submission.zip task1.txt task2.txt')
os.system('zip submission.zip task1.txt')


