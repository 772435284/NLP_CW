#imports
import os
from dpm_preprocessing import DPMProprocessed
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "microsoft/deberta-v2-xlarge"
assert model_name in ['roberta-base', 'bert-base-uncased', 'google/electra-small-discriminator',
                      "microsoft/deberta-v2-xlarge"]

model_path = f'./models/pcl_{model_name}_finetuned/model/'
tokenizer_path = f'./models/pcl_{model_name}_finetuned/tokenizer/'

#Constants
MAX_SEQ_LEN = 256
BATCH_SIZE = 9 #please update the batchsize if not enough memory, we trained on an nvidia v100 with 32BG of GPU memory

WORKING_ENV = 'SERVER'  #  Can be JONAS, SERVER
assert WORKING_ENV in ['JONAS', 'SERVER']

if WORKING_ENV == 'SERVER':
    temp_model_path = f'/hy-tmp/pcl/{model_name}/'
if WORKING_ENV == 'JONAS':
    temp_model_path = f'./experiment/pcl/{model_name}/'

#Dataset and training for training and evaluation
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

#helper method for evaluation
def predict_pcl(input, tokenizer, model, threshold=0.5):
    model.eval()
    encodings = tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=256)
    encodings = encodings.to(device)
    output = model(**encodings)
    logits = output.logits
    prob = nn.functional.softmax(logits, dim = 1)[:, 1].cpu()
    preds = np.array([int(prob > threshold)])
    return {'prediction': preds, 'confidence': prob}

#main function for evaluating the model with different thresholds for the PCL probability (please see report for more details)
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

#metrics used to monitor training
def compute_metric_eval(arg):
    logits, labels_gold = arg[0], arg[1]
    labels_pred = np.argmax(logits, axis=1)
    return {'f1_macro': f1_score(labels_gold, labels_pred, average='macro'),
            'pcl_f1':
                classification_report(labels_gold, labels_pred, target_names=["Not PCL", "PCL"], output_dict=True)[
                    'PCL']['f1-score']}  # more metrics can be added

#This function finds the best threshold to use for probability of being PCL (output of softmax) to the prediction
def find_best_threshold(tot_labels, confidences):
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
    return best_threshold


# initializing and loading pretrained model from hugginface
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)

dpm_pp = DPMProprocessed('.', 'task4_test.tsv')

#data has already been splitted and saved, for detail please refer to dpm_preprocessing.py
df_type = 'BACKTRANS'  #  Can be UNBALANCED, BACKTRANS, OVERSAMPLING
assert df_type in ['UNBALANCED', 'BACKTRANS', 'OVERSAMPLING']

if df_type == 'UNBALANCED':
    train_df_path = 'traindf.pickle'
    val_df_path = 'valdf.pickle'
if df_type == 'BACKTRANS':
    train_df_path = 'traindf_backtrans.pickle'
    val_df_path = 'valdf_backtrans.pickle'

#loading already prepared dataset
train_df = pd.read_pickle(train_df_path)
val_df = pd.read_pickle(val_df_path)

print("Training set length: ", len(train_df))
print("Validation set length: ", len(val_df))

train_dataset = PCLDataset(tokenizer, train_df)
eval_dataset = PCLDataset(tokenizer, val_df)

#Loading our custom trainer
training_args = TrainingArguments(
    output_dir=temp_model_path,
    learning_rate=5e-6,
    logging_steps=100,
    save_total_limit = 1,
    eval_steps=200,
    warmup_steps = 50,
    save_steps = 200,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
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

#Saving the best model
trainer.save_model(model_path)
tokenizer.save_pretrained(tokenizer_path)

#Loading the best model
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config).to(device)

train_dataset = PCLDataset(tokenizer, train_df)
eval_dataset = PCLDataset(tokenizer, val_df)

validation_loader = DataLoader(eval_dataset)

#evaluating the best model
preds, tot_labels, confidences = evaluate(model, tokenizer, validation_loader)
tot_labels = np.array(tot_labels)
preds = np.array(preds)
confidences = np.array(confidences)
report = classification_report(tot_labels, preds, target_names=["Not PCL", "PCL"], output_dict=True)
print(report)


"""# Test set"""
#Loading and saving competition results
test_df = dpm_pp.test_set_df
test_df['label'] = 0

test_dataset = PCLDataset(tokenizer, test_df)

test_loader = DataLoader(test_dataset)

preds, tot_labels, confidences = evaluate(model, tokenizer, test_loader, find_best_threshold(tot_labels, confidences))
tot_labels = np.array(tot_labels)
preds = np.array(preds)

# helper function to save predictions to an output file
def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi]) + '\n')

#preparing file for competition upload
labels2file([[k] for k in preds], 'task1.txt')
os.system("zip submission.zip task1.txt")
