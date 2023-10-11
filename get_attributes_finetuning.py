#Show how to obtain attributes with ICL constraint modules or BERT constraint modules (after training a BERT model)
import argparse
import pandas as pd
from finetune_bert import *
from utils import load_file
from finetune_bert import BERT_constraint_module, create_hf_dataset_finqa, create_hf_dataset_tatqa


def get_fine_tuned_attributes(dataset='finqa',attribute='modality',num_labels=3,save_model=False,output_path=''):
#Load the data
    if dataset=='finqa':
        train = load_file('datasets/finqa/train.json')
        dev = load_file('datasets/finqa/dev.json')
        test = load_file('datasets/finqa/test.json')
        train_df = pd.read_csv('data_cache/finqa/metadata/finqa_train_df.csv')
        dev_df = pd.read_csv('data_cache/finqa/metadata/finqa_dev_df.csv')
        test_df = pd.read_csv('data_cache/finqa/metadata/finqa_test_df.csv')
        text_filter_dev_df = pd.read_csv('data_cache/finqa/text_retriever/retrieved_text_finqa_dev.csv')
        text_filter_test_df = pd.read_csv('data_cache/finqa/text_retriever/retrieved_text_finqa_test.csv')
        hf_dataset = create_hf_dataset_finqa(train_df,dev_df,test_df,train,dev,test,
                                    text_filter_dev_df,text_filter_test_df)
    
    else:
    #TAT-QA
        train = load_file('datasets/tatqa/train.json')
        test = load_file('datasets/tatqa/dev.json')
        train_df = pd.read_csv('data_cache/tatqa/metadata/tatqa_train_df.csv')
        dev_df = pd.read_csv('data_cache/tatqa/metadata/tatqa_dev_df.csv')
        test_df = pd.read_csv('data_cache/tatqa/metadata/tatqa_test_df.csv')
        text_filter_dev_df = pd.read_csv('data_cache/tatqa/text_retriever/retrieved_text_tatqa_dev.csv')
        text_filter_test_df = pd.read_csv('data_cache/tatqa/text_retriever/retrieved_text_tatqa_test.csv')
        hf_dataset = create_hf_dataset_tatqa(train_df,dev_df,test_df,train,test,
                                    text_filter_dev_df,text_filter_test_df,attribute)
        
    bert = BERT_constraint_module(num_labels=num_labels)
    tokenized_hf = bert.tokenize_dataset(hf_dataset)
    bert.fit(tokenized_hf)   
    #TO DO add script to load a trained BERT constraint module
    if save_model:
        bert.save(output_path)
    
    predicted_attributes = bert.predict(tokenized_hf,split='test')
    return predicted_attributes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path',type=str,default='data_cache/finqa/metadata/finqa_test_df.csv')
    parser.add_argument('--dataset',type=str,default='finqa')
    parser.add_argument('--attribute',type=str,default='modality')
    parser.add_argument('--num_labels',type=int,default=3,help='number of possible attribute values')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    test_df = pd.read_csv(args.metadata_path)
    predicted_attributes = get_fine_tuned_attributes(args.dataset,args.attribute,args.num_labels)
    test_df['predicted_modality'] = predicted_attributes
    test_df.to_csv(args.metadata_path)