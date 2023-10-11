import numpy as np
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer,  AutoModelForSequenceClassification,  TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import evaluate
import random
from utils import *

def create_train_text_finqa(instance):
  '''
  Create the train input for a FinQA BERT constraint module
  Args:
    instance (pd.Series) : a instance row of the FinQA metadata frame
  Returns:
    train_input (str) : the representation of the instance as input for BERT
  '''
  table = json_to_pandas(instance)
  table_description = get_table_description(table)
  question = instance['qa']['question'] 
  gold_text_idx = [int(g.split('_')[1])  for g in instance['qa']['gold_inds'] if 'text' in g]
  text = instance['pre_text'] + instance['post_text']
  gold_text = ''.join([text[p] for p in gold_text_idx])
  if gold_text != '':
     return 'Question: '+question + '\nContext: ' + table_description + 'Text: ' + gold_text        
  else :   #No gold text, then retrieve some text paragraphs randomly
    random_text_idx = random.choices([i for i in range(len(text))],k=10)
    random_text_idx.sort()
    random_text = ''.join([text[p] for p in random_text_idx])
    return 'Question: '+question + '\nContext: ' + table_description + 'Text: ' + random_text      

def create_test_text_finqa(instance,idx,text_filter):
  '''
  Create the test input for a FinQA BERT constraint module
  Args:
    instance (pd.Series) : a instance row of the FinQA metadata frame
    idx (int) : the index of the instance in the raw test set
    text_filter (list): the ids of the text paragraphs to use as text context
  Returns:
    train_input (pd.Series) : the representation of the instance as input for BERT
  '''
  table = json_to_pandas(instance)
  table_description = get_table_description(table)
  question = instance['qa']['question'] 
  text_idx = [int(i) for i in text_filter.iloc[:,idx].dropna()]
  text = instance['pre_text'] + instance['post_text']
  text = ''.join([text[p] for p in text_idx])
  if text != '':
     train_input = 'Question: '+question + '\nContext: ' + table_description + 'Text: ' + text     
  else : 
    train_input = 'Question: '+question + '\nContext: ' + table_description
  return train_input  
  
def create_hf_dataset_finqa(train_df,dev_df,test_df,raw_train,raw_dev,raw_test,text_filter_dev,text_filter_test,attribute='modality'): 
    dataset = {}
    dataset['train']  = [{'label':train_df.loc[i,attribute],'text':create_train_text_finqa(raw_train[i])} for i in range(len(raw_train))]
    dataset['dev']  = [{'label':dev_df.loc[i,attribute],'text':create_test_text_finqa(raw_dev[i],i,text_filter_dev)} for i in range(len(raw_dev))]
    dataset['test']  = [{'label':test_df.loc[i,attribute],'text':create_test_text_finqa(raw_test[i],i,text_filter_test)} for i in range(len(raw_test))]
    hf_dataset = DatasetDict()
    for k,v in dataset.items():
        hf_dataset[k] = Dataset.from_list(v)
    return hf_dataset

def create_train_text_tatqa(instance,raw_data):
  '''
  Create the train input for a TAT-QA BERT constraint module
  Args:
    instance (pd.Series) : a instance row of the TAT-QA metadata frame
    raw_data (list) : the raw TAT-QA dataset
  Returns:
    train_input (str) : the representation of the instance as input for BERT
  '''
  context = raw_data[instance.context_index] #Retrieve raw content
  table = tatqa_table_to_pandas(context)
  table_description = get_table_description(table)
  question = instance['question'] 
  gold_text_idx = [int(p)-1 for p in context['questions'][instance.instance_index]['rel_paragraphs']]
  gold_text = ''.join([context['paragraphs'][p]['text'] for p in gold_text_idx])
  if gold_text != '':
     return 'Question: '+question + '\nContext: ' + table_description + 'Text: ' + gold_text        
  else :   #No gold text (because the modality is table) : then retrieve some text paragraphs randomly
    random_text_idx = random.choices([i for i in range(len(context['paragraphs']))],k=2)
    random_text_idx.sort()
    text = ''.join([context['paragraphs'][p]['text'] for p in random_text_idx])
    train_input = 'Question: '+question + '\nContext: ' + table_description + 'Text: ' + text 
    return train_input


def create_test_text_tatqa(instance,raw_data,text_filter):
  '''
  Create the test input for a TAT-QA BERT constraint module
  Args:
    instance (pd.Series) : a instance row of the TAT-QA metadata frame
    raw_data (list) : the raw TAT-QA dataset
    text_filter (list): the ids of the text paragraphs to use as text context
  Returns:
    train_input (pd.Series) : the representation of the instance as input for BERT
  '''
  context = raw_data[instance.context_index] #Retrieve raw content
  table = tatqa_table_to_pandas(context)
  table_description = get_table_description(table)
  question = instance['question'] 
  text_idx = [int(i) for i in text_filter.dropna()]
  if len(context['paragraphs'])>=10:
    try:
      text = ''.join([context['paragraphs'][p]['text'] for p in text_idx])
    except:
      text = ''
      print('Error')
  else:
    text = ''.join([context['paragraphs'][p]['text'] for p in range(len(context['paragraphs']))])
  if text != '':
     return 'Question: '+question + '\nContext: ' + table_description + 'Text: ' + text        
  else : 
    return 'Question: '+question + '\nContext: ' + table_description
  

def create_hf_dataset_tatqa(train_df,dev_df,test_df,raw_train,raw_test,text_filter_dev,text_filter_test,attribute='modality'): 
    dataset = {}
    dataset['train']  = [{'label':train_df.loc[i,attribute],'text':create_train_text_tatqa(train_df.iloc[i,:],raw_train)} for i in range(train_df.shape[0])]
    dataset['dev']  = [{'label':dev_df.loc[j,attribute],'text':create_test_text_tatqa(dev_df.iloc[j,:],raw_train,text_filter_dev.iloc[:,j])} for j in range(dev_df.shape[0])]
    dataset['test']  = [{'label':test_df.loc[j,attribute],'text':create_test_text_tatqa(test_df.iloc[j,:],raw_test,text_filter_test.iloc[:,j])} for j in range(test_df.shape[0])]
    hf_dataset = DatasetDict()
    for k,v in dataset.items():
        hf_dataset[k] = Dataset.from_list(v)
    return hf_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluate.load('accuracy').compute(predictions=predictions, references=labels)


class BERT_constraint_module:
  def __init__(self,
               model='bert-base-cased',
               num_labels=3,
               output_dir="test_trainer",
               seed=42):
    self.tokenizer = AutoTokenizer.from_pretrained(model)
    self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=num_labels)
    self.training_args = TrainingArguments(output_dir=output_dir,seed=seed)
    self.trainer = None

  def tokenize_dataset(self,hf_dataset):
    def tokenize_function(examples):
      return self.tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = hf_dataset.map(tokenize_function,batched=True)
    return tokenized_datasets
  
  def fit(self,tokenized_dataset):
    self.trainer = Trainer(
        model=self.model,
        args=self.training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['dev'],
        compute_metrics=compute_metrics
    )
    self.trainer.train()

  def predict(self,tokenized_dataset,split='test'):
    pred_score = self.trainer.predict(tokenized_dataset[split])
    pred_label = [np.argmax(p) for p in pred_score.predictions]
    return pred_label