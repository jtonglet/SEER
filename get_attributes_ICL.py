import argparse
import pandas as pd
from utils import load_file
from generate_prompt import get_test_prompt_modality_finqa, get_test_prompt_modality_tatqa, get_test_prompt_answer_type_tatqa
from codex_run import few_shot_prompting


def get_icl_attributes(dataset='finqa',attribute='modality',key='OPENAI_API_KEY',model='code-davinci-002',temperature=0):
    predicted_attributes=[]
    if dataset=='finqa':
        train = load_file('datasets/finqa/train.json')
        test = load_file('datasets/finqa/test.json')
        train_df = pd.read_csv('data_cache/finqa/metadata/finqa_train_df.csv')
        text_filter_df = pd.read_csv('data_cache/finqa/text_retriever/retrieved_text_finqa_test.csv')

        exemplars = [0,1,4]  #one exemplar per modality
        for idx in range(len(test)):
            instance = test[idx]
            text_filter = [int(t) for t in text_filter_df.iloc[:,idx].dropna().to_list()]
            prompt = get_test_prompt_modality_finqa(instance,train,train_df,exemplars,text_filter)
            predicted_attributes.append(few_shot_prompting(prompt,key=key,temperature=temperature,model=model)['choices'][0]['text'])
            print(predicted_attributes[-1])
        #Map to integer labels
        predicted_attributes = [0 if a=='table' else 1 if a=='text' else 2 for a in predicted_attributes]
    
    else:
    #TAT-QA
        train = load_file('datasets/tatqa/train.json')
        test = load_file('datasets/tatqa/dev.json')
        train_df = pd.read_csv('data_cache/tatqa/metadata/tatqa_train_df.csv')
        test_df = pd.read_csv('data_cache/tatqa/metadata/tatqa_test_df.csv')
        text_filter_df = pd.read_csv('data_cache/tatqa/text_retriever/retrieved_text_tatqa_test.csv')  
        if attribute =='modality':
            exemplars = [0,15,20]
            for idx in range(len(test_df)):
                context = test[test_df.loc[idx,'context_index']]
                instance = context['questions'][test_df.loc[idx,'instance_index']]
                text_filter = [int(t) for t in text_filter_df.iloc[:,idx].dropna().to_list()]
                prompt = get_test_prompt_modality_tatqa(instance,context,train,train_df,exemplars,text_filter)
                predicted_attributes.append(few_shot_prompting(prompt,key=key)['choices'][0]['text'])
            #Map to integer labels
            predicted_attributes = [0 if a=='table' else 1 if a=='text' else 2 for a in predicted_attributes]

        elif attribute=='answer_type':
            exemplars = [30, 31, 33, 34]
            for idx in range(len(test_df)):
                context = test[test_df.loc[idx,'context_index']]
                instance = context['questions'][test_df.loc[idx,'instance_index']]
                text_filter = [int(t) for t in text_filter_df.iloc[:,idx].dropna().to_list()]
                prompt = get_test_prompt_answer_type_tatqa(instance,context,train,train_df,exemplars,text_filter)
                predicted_attributes.append(few_shot_prompting(prompt,key=key)['choices'][0]['text'])
        else:
            print('Invalid attribute name. Available values are modality and answer_type')
            return None
        
    return predicted_attributes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path',type=str,default='data_cache/finqa/metadata/finqa_test_df.csv')
    parser.add_argument('--dataset',type=str,default='finqa')
    parser.add_argument('--attribute',type=str,default='modality')
    parser.add_argument('--key',type=str,default='OPENAI_API_KEY')
    parser.add_argument('--model', type=str, default='code-davinci-002') 
    parser.add_argument('--temperature', type=float, default=0.0)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    test_df = pd.read_csv(args.metadata_path)
    predicted_attributes = get_icl_attributes(args.dataset,args.attribute,args.key,args.model,args.temperature)
    test_df['predicted_modality'] = predicted_attributes
    test_df.to_csv(args.metadata_path)