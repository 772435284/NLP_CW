# -*- coding: utf-8 -*-
"""jonas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HrO9_fUvOaDklVloYwmGMLWRCGZHm5Ww
"""




import os
from dpm_preprocessing import DPMProprocessed
import torch
#from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, RobertaConfig
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import f1_score

device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

#dependencies
# ! python -m pip install nltk
# ! python -m pip install wordcloud
# ! python -m pip install Unidecode
# ! python -m pip install beautifulsoup4
os.environ["WANDB_DISABLED"] = "true"
os.system("python -m pip install nltk")
os.system("python -m pip install wordcloud")
os.system("python -m pip install Unidecode")
os.system("python -m pip install beautifulsoup4")



model_name = "microsoft/deberta-v2-xlarge"
assert model_name in ['roberta-base', 'bert-base-uncased', 'google/electra-small-discriminator', "microsoft/deberta-v2-xlarge"]

model_path = f'./models/pcl_{model_name}_finetuned/model/'
model_pretrained_path = f'./models/pcl_{model_name}_pretrained/model/'
tokenizer_path = f'./models/pcl_{model_name}_finetuned/tokenizer/'
MAX_SEQ_LEN = 256

WORKING_ENV = 'SERVER' # Can be JONAS, SERVER
assert WORKING_ENV in ['JONAS', 'SERVER']

if WORKING_ENV == 'SERVER':
    temp_model_path = f'/hy-tmp/pcl/{model_name}/'
    temp_pretrain_path = f'/hy-tmp/pcl/pretrain/{model_name}/'
if WORKING_ENV == 'JONAS': 
    temp_model_path = f'./experiment/pcl/{model_name}/'
    temp_pretrain_path = f'./experiment/pcl/pretrain/{model_name}/'

dpm_pp = DPMProprocessed('.', 'task4_test.tsv')

df_type = 'UNBALANCED' # Can be UNBALANCED, BACKTRANS, OVERSAMPLING
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

class PCLDatasetPretrain(torch.utils.data.Dataset):

    def __init__(self, tokenizer, input_set):

        self.tokenizer = tokenizer
        self.texts = list(input_set['text'])
        self.mlm_probability = 0.15
        
    def collate_fn(self, batch):
        batch = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LEN)

        inputs, labels = self.mask_tokens(batch["input_ids"])
        return {"input_ids": inputs, "labels": labels}

    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
       
        item = self.texts[idx]
        return item

class Trainer_MLM(Trainer):
    def compute_loss(self, model, inputs):
        
        labels = inputs['labels']

        outputs = model(**inputs)

        # MLM loss
        lm_loss = nn.CrossEntropyLoss()

        loss_mlm = lm_loss(outputs.view(-1, model.config.vocab_size), labels.view(-1))
        
        loss = loss_mlm
        
        return loss


config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name , config = config).to(device)

pretrain_dataset = PCLDatasetPretrain(tokenizer, train_df)

training_args = TrainingArguments(
        output_dir=temp_pretrain_path,
        learning_rate = 1e-6,
        logging_steps= 100,
        per_device_train_batch_size=4,
        per_device_eval_batch_size = 4,
        num_train_epochs = 2
                        )

trainer = Trainer(
        model=model,                         
        args=training_args,                 
        train_dataset=pretrain_dataset,                   
        data_collator=pretrain_dataset.collate_fn
    )
trainer.train()
trainer.save_model(model_pretrained_path)

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


config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_pretrained_path , config = config).to(device)


train_dataset = PCLDataset(tokenizer, train_df)
eval_dataset = PCLDataset(tokenizer, val_df)

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
        num_train_epochs = 4,
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

trainer.save_model(model_path)
tokenizer.save_pretrained(tokenizer_path)

train_df.to_pickle('train_df.pickle')
val_df.to_pickle('val_df.pickle')

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path , config = config).to(device)

train_df = pd.read_pickle('train_df.pickle')
val_df = pd.read_pickle('val_df.pickle')

train_dataset = PCLDataset(tokenizer, train_df)
eval_dataset = PCLDataset(tokenizer, val_df)

def predict_pcl(input, tokenizer, model): 
  model.eval()
  encodings = tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LEN)
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

validation_loader = DataLoader(eval_dataset)

preds, tot_labels = evaluate(model, tokenizer, validation_loader)
tot_labels = np.array(tot_labels)
preds = np.array(preds)
report = classification_report(tot_labels, preds, target_names=["Not PCL","PCL"], output_dict= True)
print(report)

print(report['accuracy'])
print(report['Not PCL']['f1-score'])
print(report['PCL']['f1-score'])

"""# Test set"""

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

# preds.shape
preds.shape

from collections import Counter
preds = preds.reshape(-1)
Counter(preds)

# helper function to save predictions to an output file
def labels2file(p, outf_path):
	with open(outf_path,'w') as outf:
		for pi in p:
			outf.write(','.join([str(k) for k in pi])+'\n')

labels2file([[k] for k in preds], 'task1.txt')
os.system("zip submission.zip task1.txt")

