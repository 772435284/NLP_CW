#!/usr/bin/env python
# coding: utf-8

# In[1]:


#dependencies
get_ipython().system(' python -m pip install nltk')
get_ipython().system(' python -m pip install wordcloud')
get_ipython().system(' python -m pip install Unidecode')
get_ipython().system(' python -m pip install beautifulsoup4')


# In[2]:


from dpm_preprocessing import DPMProprocessed
import torch
#from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, RobertaConfig
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.empty_cache()
device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

import os
os.environ["WANDB_DISABLED"] = "true"


model_name = "microsoft/deberta-v2-xlarge"
assert model_name in ['roberta-base', 'bert-base-uncased', 'google/electra-small-discriminator', "microsoft/deberta-v2-xlarge"]

model_path = f'./models/pcl_{model_name}_finetuned/model/'
tokenizer_path = f'./models/pcl_{model_name}_finetuned/tokenizer/'
MAX_SEQ_LEN = 256

WORKING_ENV = 'SERVER' # Can be JONAS, SERVER
assert WORKING_ENV in ['JONAS', 'SERVER']

if WORKING_ENV == 'SERVER':
    temp_model_path = f'/hy-tmp/pcl/{model_name}/'
if WORKING_ENV == 'JONAS': 
    temp_model_path = f'./experiment/pcl/{model_name}/'


# In[ ]:


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
        encodings['labels'] =  torch.tensor(labels)
        return encodings
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
       
        item = {'text': self.texts[idx],
                'label': self.labels[idx]}
        return item


# In[ ]:


config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name , config = config).to(device)


# In[ ]:


dpm_pp = DPMProprocessed('.', 'task4_test.tsv')


df_type = 'BACKTRANS' # Can be UNBALANCED, BACKTRANS, OVERSAMPLING
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

print("Training set length: ",len(train_df))
print("Validation set length: ",len(val_df))

train_dataset = PCLDataset(tokenizer, train_df)
eval_dataset = PCLDataset(tokenizer, val_df)


# In[ ]:


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


# In[ ]:


validation_loader = DataLoader(eval_dataset)
def compute_metric_eval(arg):
    logits, labels_gold = arg[0], arg[1]
    labels_pred = np.argmax(logits, axis = 1)
    return {'f1_macro' :f1_score(labels_gold, labels_pred, average='macro'), 
            'pcl_f1': classification_report(labels_gold, labels_pred, target_names=["Not PCL","PCL"], output_dict= True)['PCL']['f1-score']} #more metrics can be added

training_args = TrainingArguments(
        output_dir=temp_model_path,
        learning_rate = 1e-6,
        logging_steps= 100,
        eval_steps = 500,
        per_device_train_batch_size=4,
        per_device_eval_batch_size = 4,
        num_train_epochs = 3,
        evaluation_strategy= "steps",
        load_best_model_at_end=True,
        metric_for_best_model='pcl_f1'
        )

trainer = CustomTrainer(
        model=model,                         
        args=training_args,                 
        train_dataset=train_dataset,                   
        data_collator=eval_dataset.collate_fn,
        compute_metrics = compute_metric_eval,
        eval_dataset = eval_dataset
    )
trainer.train()


# In[ ]:


trainer.save_model(model_path)
tokenizer.save_pretrained(tokenizer_path)

train_df.to_pickle('train_df.pickle')
val_df.to_pickle('val_df.pickle')


# In[ ]:


config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path , config = config).to(device)


# In[ ]:



train_df = pd.read_pickle('train_df.pickle')
val_df = pd.read_pickle('val_df.pickle')

train_dataset = PCLDataset(tokenizer, train_df)
eval_dataset = PCLDataset(tokenizer, val_df)


# In[ ]:


def predict_pcl(input, tokenizer, model): 
  model.eval()
  encodings = tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=256)
  encodings = encodings.to(device)
  output = model(**encodings)
  logits = output.logits
  preds = torch.max(logits, 1)

  return {'prediction':preds[1], 'confidence':preds[0]}

def evaluate(model, tokenizer, data_loader):

  preds = []
  tot_labels = []

  with torch.no_grad():
    for data in (data_loader): 

      labels = {}
      labels['label'] = data['label']

      tweets = data['text']

      pred = predict_pcl(tweets, tokenizer, model)

      preds.append(np.array(pred['prediction'].cpu()))
      tot_labels.append(np.array(labels['label'].cpu()))

  # with the saved predictions and labels we can compute accuracy, precision, recall and f1-score
  

  return preds, tot_labels


# In[ ]:


validation_loader = DataLoader(eval_dataset)

preds, tot_labels = evaluate(model, tokenizer, validation_loader)
tot_labels = np.array(tot_labels)
preds = np.array(preds)
report = classification_report(tot_labels, preds, target_names=["Not PCL","PCL"], output_dict= True)
print(report)

print(report['accuracy'])
print(report['Not PCL']['f1-score'])
print(report['PCL']['f1-score'])


# # Test set

# In[ ]:


dpm_pp.load_test()
test_df = dpm_pp.test_set_df
test_df['label'] = 0
test_dataset = PCLDataset(tokenizer, test_df)

test_loader = DataLoader(test_dataset)

preds, tot_labels = evaluate(model, tokenizer, test_loader)
tot_labels = np.array(tot_labels)
preds = np.array(preds)
# report = classification_report(tot_labels, preds, target_names=["Not PCL","PCL"], output_dict= True)
# print(report)

# print(report['accuracy'])
# print(report['Not PCL']['f1-score'])
# print(report['PCL']['f1-score'])


# In[ ]:


# preds.shape
preds.shape


# In[ ]:


from collections import Counter
preds = preds.reshape(-1)
Counter(preds)


# In[ ]:


# helper function to save predictions to an output file
def labels2file(p, outf_path):
	with open(outf_path,'w') as outf:
		for pi in p:
			outf.write(','.join([str(k) for k in pi])+'\n')


# In[ ]:


labels2file([[k] for k in preds], 'task1.txt')
get_ipython().system('cat task1.txt | head -n 10')
get_ipython().system('zip submission.zip task1.txt ')


# In[ ]:




