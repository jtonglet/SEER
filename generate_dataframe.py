from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, GPT2Tokenizer
from generate_prompt import  get_prompt_instance_finqa, get_prompt_instance_tatqa
from utils import get_program_template, json_to_pandas, get_table_description, preprocess_text
import spacy
import pandas as pd
gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def create_question_dataframe_finqa(dataset,preprocess=True,ner_mask=True):
    '''
    Create a dataframe with questions, processed text, and equation  
    '''
    if preprocess:
        spacy_model = spacy.load("en_core_web_lg")
    if ner_mask:
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")
        bert_model = pipeline("ner", model=model, tokenizer=tokenizer)
    
    index = [i for i in range(len(dataset))]
    questions = [dataset[i]['qa']['question'] for i in range(len(dataset))]
    programs =  [dataset[i]['qa']['program'] for i in range(len(dataset))]
    answers =  [dataset[i]['qa']['exe_ans'] for i in range(len(dataset))]
    dataframe = pd.DataFrame({'index':index,'question':questions,'answer':answers,'program':programs})
    dataframe['program_template'] = dataframe['program'].apply(lambda row: get_program_template(row))

    table_desc = [get_table_description(json_to_pandas(dataset[i])) for i in range(len(dataset))]
    prompts = [get_prompt_instance_finqa(dataset[i]) for i in range(len(dataset))]
    dataframe['has_table'] = [1 if desc != 'No table available.' else 0 for desc in table_desc]
    dataframe['prompt_length'] = [len(p) for p in prompts]
    dataframe['token_prompt_length'] = [len(gpt2tokenizer(p)['input_ids']) for p in prompts]
    dataframe['use_table'] = [1 if 'table_query_0' in p else 0 for p in prompts]
    dataframe['use_text'] = [1 if 'text_variable_0' in p else 0 for p in prompts]


    dataframe['modality'] = dataframe.apply(lambda row : 0 if row['use_table']==1 and row['use_text'] ==0
                                            else 1 if row['use_table']==0 and row['use_text'] == 1 
                                            else 2,axis=1)
    dataframe['other'] = dataframe['modality'].apply(lambda row: 1 if row==3 else 0)  #For example questions that only require constants
    dataframe['hybrid'] = dataframe['modality'].apply(lambda row: 1 if row==2 else 0)
    dataframe['text_only'] = dataframe['modality'].apply(lambda row: 1 if row==1 else 0)
    dataframe['table_only'] = dataframe['modality'].apply(lambda row: 1 if row==0 else 0)
    if preprocess:
        dataframe['processed_question'] = dataframe['question'].apply(lambda row : preprocess_text(row,spacy_model,bert_model,ner_mask=ner_mask))
    return dataframe


def create_question_dataframe_tatqa(dataset,preprocess=True,ner_mask=True):
    '''
    Create a dataframe with questions, processed text, and equation  
    '''
    if preprocess:
        spacy_model = spacy.load("en_core_web_lg")
    if ner_mask:
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")
        bert_model = pipeline("ner", model=model, tokenizer=tokenizer)
   
    context_index = [i for i in range(len(dataset)) for _ in range(len(dataset[i]['questions']))]
    instance_index = [j for i in range(len(dataset)) for j in range(len(dataset[i]['questions']))]
  
    questions = [dataset[i]['questions'][j]['question'] for i in range(len(dataset)) for j in range(len(dataset[i]['questions']))]
    programs =  [dataset[i]['questions'][j]['derivation'] for i in range(len(dataset)) for j in range(len(dataset[i]['questions']))]
    dataframe = pd.DataFrame({'context_index':context_index,'instance_index':instance_index,'question':questions,'program':programs})
    prompts = [get_prompt_instance_tatqa(dataset[i]['questions'][j],dataset[i]) for i in range(len(dataset)) for j in range(len(dataset[i]['questions']))]
    dataframe['token_prompt_length'] = [len(gpt2tokenizer(p)['input_ids']) for p in prompts]
    dataframe['use_table'] = [1 if dataset[i]['questions'][j]['answer_from'] in ['table','table-text'] else 0 for i in range(len(dataset)) for j in range(len(dataset[i]['questions'])) ]
    dataframe['use_text'] = [1 if dataset[i]['questions'][j]['answer_from'] in ['text','table-text'] else 0 for i in range(len(dataset)) for j in range(len(dataset[i]['questions']))]

    dataframe['modality'] = dataframe.apply(lambda row : 0 if row['use_table']==1 and row['use_text'] ==0
                                            else 1 if row['use_table']==0 and row['use_text'] == 1 
                                            else 2,axis=1)
    dataframe['other'] = dataframe['modality'].apply(lambda row: 1 if row==3 else 0)  #For example questions that only require constants
    dataframe['hybrid'] = dataframe['modality'].apply(lambda row: 1 if row==2 else 0)
    dataframe['text_only'] = dataframe['modality'].apply(lambda row: 1 if row==1 else 0)
    dataframe['table_only'] = dataframe['modality'].apply(lambda row: 1 if row==0 else 0)
    dataframe['answer_type'] = [dataset[i]['questions'][j]['answer_type'] for i in range(len(dataset)) for j in range(len(dataset[i]['questions']))]
    dataframe['answer_type_int'] = dataframe['answer_type'].apply(lambda row :0 if row == 'span' else 1 if row == 'multi-span' else 2 if row =='arithmetic' else 3)
    dataframe['span'] = dataframe['answer_type'].apply(lambda row : 1 if row=='span' else 0)
    dataframe['multi-span'] = dataframe['answer_type'].apply(lambda row : 1 if row=='multi-span' else 0)
    dataframe['arithmetic'] = dataframe['answer_type'].apply(lambda row : 1 if row=='arithmetic' else 0)
    dataframe['count'] = dataframe['answer_type'].apply(lambda row : 1 if row=='count' else 0)

    if preprocess:
        dataframe['processed_question'] = dataframe['question'].apply(lambda row : preprocess_text(row,spacy_model,bert_model,ner_mask=ner_mask))
    return dataframe