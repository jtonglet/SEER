from utils import *
from transformers import GPT2Tokenizer
gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def get_prompt_instance_finqa(instance,split='train',retrieved_text_idx=[]):
    '''
    Generate the prompt of an instance of FinQA.
    Params:
        instance (dict) : a FinQA instance.
        split (str) : "train" or "test".
        retrieved_text_idx (int) : the list of index of text paragraphs from the context to include in the prompt
    '''
    #Format table 
    table = json_to_pandas(instance)
    #Linearize the table description as string
    table_description = get_table_description(table)  
    try:
        code = DSL_to_code_finqa(instance,table)
    except:
        code = ''
    prompt='Read the following text and table, and then write code to answer the question:\n'  
    if split=='train':
        #Train instance : include only the relevant text paragraphs.
        for k,v in instance['qa']['gold_inds'].items():
            if 'text' in k:
                prompt += v + '\n' 
    else:
        #Test instance : include the relevant text paragraphs according to the retriever
        context_text = instance['pre_text'] + instance['post_text']
        for i in retrieved_text_idx:
            prompt+= context_text[int(i)] + '\n'
    #Add the table description
    prompt += table_description
    #Add the question
    prompt+= 'Question: ' + instance['qa']['question'] + '?\n'
    if split=='train':
        #Train instance : provide the answer
        prompt+= 'Answer:\n#Python\n' + code
    else:
        prompt+= 'Answer:\n#Python\n'
    return prompt


def get_prompt_instance_tatqa(instance,context,split='train',retrieved_text_idx=[]):
    '''
    Generate the prompt of an instance of TAT-QA.
    Params:
        instance (dict) : a FinQA instance.
        split (str) : "train" or "test".
        retrieved_text_idx (int) : the list of index of text paragraphs from the context to include in the prompt
    '''
    table = tatqa_table_to_pandas(context)
    table_description = get_table_description(table)   
    try:
        code = DSL_to_code_tatqa(instance,table)
    except:
        code = ''
    prompt='Read the following text and table, and then write code to answer the question:\n'  
    if split=='train':
        for v in instance['rel_paragraphs']:
                prompt += context['paragraphs'][int(v)-1]['text'] + '\n' 
    else:
        context_text = context['paragraphs']
        for i in retrieved_text_idx:
            prompt+= context_text[int(i)]['text'] + '\n'
    #Add the table description
    prompt += table_description
    #Add the question
    prompt+= 'Question: ' + instance['question'] + '?\n'
    if split=='train':
        prompt+= 'Answer:\n#Python\n' + code
    else:
        prompt+= 'Answer:\n#Python\n'  
    return prompt



def get_prompt_modality_finqa(instance,split='train',retrieved_text_idx=[],modality='table'):
    '''
    Generate the prompt of an instance of FinQA for the constraint module "Modality Prediction".
    Params:
        instance (dict) : a FinQA instance.
        split (str) : "train" or "test".
        retrieved_text_idx (int) : the list of index of text paragraphs from the context to include in the prompt
        modality (str) : the ground truth modality, to be included for train instances.
    '''
    prompt = 'You are a financial analyst. Read the following document and question.\n'
    if split=='train':
        for k,v in instance['qa']['gold_inds'].items():
            if 'text' in k:
                prompt += v + '\n' 
    else:
        context_text = instance['pre_text'] + instance['post_text']
        for i in retrieved_text_idx:
            prompt+=  context_text[int(i)] + '\n'
    table = json_to_pandas(instance)
    table_description = get_table_description(table)
    prompt += table_description
    prompt+= 'Question: ' + instance['qa']['question'] + '?\n'
    prompt+= ('Do you need data from the table, the text paragraphs, or both (hybrid) to answer this question? Answer by one of the following : table, text, hybrid.')
    if split=='train':
        #add the answer
        prompt+= '\n' +modality + '\n' 
    return prompt


def get_prompt_modality_tatqa(instance,context,split='train',retrieved_text_idx=[],modality='table'):
    '''
    Generate the prompt of an instance of TAT-QA for the constraint module "Modality Prediction".
    Params:
        instance (dict) : a FinQA instance.
        split (str) : "train" or "test".
        retrieved_text_idx (int) : the list of index of text paragraphs from the context to include in the prompt
        modality (str) : the ground truth modality, to be included for train instances.
    '''
    prompt = 'You are a financial analyst. Read the following document and question.\n'
    if split=='train':
        for v in instance['rel_paragraphs']:
                prompt += context['paragraphs'][int(v)-1]['text'] + '\n' 
    else:
        context_text = context['paragraphs']
        for i in retrieved_text_idx:
            prompt+= context_text[int(i)]['text'] + '\n'
    table = tatqa_table_to_pandas(context)
    table_description = get_table_description(table)
    prompt += table_description
    prompt+= 'Question: ' + instance['question'] + '?\n'
    prompt+= ('Do you need data from the table, the text paragraphs, or both (hybrid) to answer this question? Answer by one of the following : table, text, hybrid.')
    if split=='train':
        prompt+= '\n' +modality + '\n' 
    return prompt


def get_prompt_answer_type_tatqa(instance,context,split='train',retrieved_text_idx=[],answer_type='span'):
    '''
    Generate the prompt of an instance of TAT-QA for the constraint module "Answer Type Prediction".
    Params:
        instance (dict) : a TAT-QA instance.
        split (str) : "train" or "test".
        retrieved_text_idx (int) : the list of index of text paragraphs from the context to include in the prompt
        modality (str) : the ground truth modality, to be included for train instances.
    '''
    prompt = 'You are a financial analyst. Read the following document and question.\n'
    if split=='train':
        for v in instance['rel_paragraphs']:
                prompt += context['paragraphs'][int(v)-1]['text'] + '\n' 
    else:
        context_text = context['paragraphs']
        for i in retrieved_text_idx:
            prompt+= context_text[int(i)]['text'] + '\n'
    table = tatqa_table_to_pandas(context)
    table_description = get_table_description(table)
    prompt += table_description
    prompt+= 'Question: ' + instance['question'] + '?\n'
    prompt+= 'Does this question require to extract spans from the document, to count, or to perform an arithmetic reasoning? Answer by one of the following : span, multi-span, count, arithmetic.'
    if split=='train':
        prompt+= '\n' +answer_type + '\n'
    return prompt


def get_test_prompt_finqa(instance,train,few_shot_idx=[],retrieved_text_idx=[]):
    '''
    Generate the complete prompt for a test instance, including few-shot exemplars.
    Params:
        instance (dict) : a FinQA instance.
        train (list) : the list object containing all train instances.
        few_shot_idx (list) : the train indexes of the few-shot exemplars
        retrieved_text_idx (int) : the list of index of text paragraphs from the context to include in the prompt
    '''
    prompt = ''
    #Add few shot instances
    for i in few_shot_idx[::-1]: 
        #Present the few_shot_idx in inverted order from least similar to most similar
        prompt+= get_prompt_instance_finqa(train[i]) + '\n'
    #Add test instance
    prompt += get_prompt_instance_finqa(instance,'test',retrieved_text_idx)
    return prompt  


def get_test_prompt_modality_finqa(instance,train,train_dataframe,few_shot_idx=[],retrieved_text_idx=[]):
    '''
    Generate the complete prompt for the modality prediction of a test instance, including few-shot exemplars.
    Params:
        instance (dict) : a FinQA instance.
        train (list) : the list object containing all train instances.
        train_dataframe (pandas.DataFrame) : the dataframe containing metadata information about the train instances.
        few_shot_idx (list) : the train indexes of the few-shot exemplars
        retrieved_text_idx (int) : the list of index of text paragraphs from the context to include in the prompt
    '''
    prompt = ''
    #Add few shot instances
    for i in few_shot_idx:  
        mapping = {0:'table',1:'text',2:'hybrid'}
        modality = mapping[train_dataframe.loc[i,'modality']]
        prompt+= get_prompt_modality_finqa(train[i],'train',modality=modality) + '\n'
    #Add test instance
    prompt += get_prompt_modality_finqa(instance,'test',retrieved_text_idx)
    return prompt  


def get_test_prompt_tatqa(instance,context,train,train_dataframe,few_shot_idx=[],retrieved_text_idx=[]):
    '''
    Generate the complete prompt for a test instance, including few-shot exemplars.
    Params:
        instance (dict) : a TAT-QA instance.
        context (dict) : the context of the TAT-QA instance.
        train (list) : the list object containing all train instances.
        few_shot_idx (list) : the train indexes of the few-shot exemplars
        retrieved_text_idx (int) : the list of index of text paragraphs from the context to include in the prompt
    '''
    prompt = ''
    #Add few shot instances
    for i in few_shot_idx[::-1]: 
        #Present the few_shot_idx in inverted order from least similar to most similar
        train_context = train[train_dataframe.loc[i,'context_index']]
        train_instance = train_context['questions'][train_dataframe.loc[i,'instance_index']]
        prompt+= get_prompt_instance_tatqa(train_instance,train_context) + '\n'
    #Add test instance
    prompt += get_prompt_instance_tatqa(instance,context,'test',retrieved_text_idx)
    return prompt  


def get_test_prompt_modality_tatqa(instance,context,train,train_dataframe,few_shot_idx=[],retrieved_text_idx=[]):
    '''
    Generate the complete prompt for the modality prediction of  a test instance, including few-shot exemplars.
    Params:
        instance (dict) : a TAT-QA instance.
        context (dict) : the context of the TAT-QA instance.
        train (list) : the list object containing all train instances.
        train_dataframe (pandas.DataFrame) : the dataframe containing metadata information about the train instances.
        few_shot_idx (list) : the train indexes of the few-shot exemplars
        retrieved_text_idx (int) : the list of index of text paragraphs from the context to include in the prompt
    '''
    prompt = ''
    #Add few shot instances
    for i in few_shot_idx[::-1]:  
        train_context = train[train_dataframe.loc[i,'context_index']]
        train_instance = train_context['questions'][train_dataframe.loc[i,'instance_index']]
        mapping = {0:'table',1:'text',2:'hybrid'}
        modality = mapping[train_dataframe.loc[i,'modality']]
        prompt+= get_prompt_modality_tatqa(train_instance,train_context,'train',modality=modality) + '\n'
    #Add test instance
    prompt += get_prompt_modality_tatqa(instance,context,'test',retrieved_text_idx)
    return prompt 


def get_test_prompt_answer_type_tatqa(instance,context,train,train_dataframe,few_shot_idx=[],retrieved_text_idx=[]):
    '''
    Generate the complete prompt for the  answer type prediction of a test instance, including few-shot exemplars.
    Params:
        instance (dict) : a TAT-QA instance.
        context (dict) : the context of the TAT-QA instance.
        train (list) : the list object containing all train instances.
        train_dataframe (pandas.DataFrame) : the dataframe containing metadata information about the train instances.
        few_shot_idx (list) : the train indexes of the few-shot exemplars
        retrieved_text_idx (int) : the list of index of text paragraphs from the context to include in the prompt
    '''
    prompt = ''
    #Add few shot instances
    for i in few_shot_idx[::-1]:
        train_context = train[train_dataframe.loc[i,'context_index']]
        train_instance = train_context['questions'][train_dataframe.loc[i,'instance_index']]
        answer_type = train_dataframe.loc[i,'answer_type']
        prompt+= get_prompt_answer_type_tatqa(train_instance,train_context,'train',answer_type=answer_type) + '\n'
    #Add test instance
    prompt += get_prompt_answer_type_tatqa(instance,context,'test',retrieved_text_idx)
    return prompt  

def get_test_messages_finqa(instance,train,few_shot_idx=[],retrieved_text_idx=[],query=False):
    '''
    Generate a test prompt using the OpenAI Chat API syntax.
    Generates a message with the following structure : messages = [{'role':'user','content':prompt}] 
    '''
    messages = []
    for i in few_shot_idx[::-1]: 
        prompt = get_prompt_instance_finqa(train[i],query=query)
        messages.append({'role':'user','content':prompt.split('#Python\n')[0]+'#Python\n'})
        messages.append({'role':'assistant','content':prompt.split('#Python\n')[1]})
    #Add test instance
    prompt = get_prompt_instance_finqa(instance,'test',retrieved_text_idx,query=query)
    messages.append({'role':'user','content':prompt.split('#Python\n')[0]+'#Python\n'})
    return messages

def get_test_messages_tatqa(instance,context,train,train_dataframe,few_shot_idx=[],retrieved_text_idx=[],query=False):
    '''
    Generate a test prompt using the OpenAI Chat API syntax.
    Generates a message with the following structure : messages = [{'role':'user','content':prompt}] 
    '''
    messages = []
    for i in few_shot_idx[::-1]:  
        train_context = train[train_dataframe.loc[i,'context_index']]
        train_instance = train_context['questions'][train_dataframe.loc[i,'instance_index']]
        prompt= get_prompt_instance_tatqa(train_instance,train_context,query=False) + '\n'
        messages.append({'role':'user','content':prompt.split('#Python\n')[0]+'#Python\n'})
        messages.append({'role':'assistant','content':prompt.split('#Python\n')[1]})
    #Add test instance
    prompt = get_prompt_instance_tatqa(instance,context,'test',retrieved_text_idx,query=False)
    messages.append({'role':'user','content':prompt.split('#Python\n')[0]+'#Python\n'})
    return messages


def get_max_prompt_length_finqa(i,raw_dataset,text_filter_df,max_code_length,max_model_length=4096): 
    #4096 tokens for Codex
    tf = [int(t) for t in text_filter_df.iloc[:,i].dropna().to_list()]
    prompt = get_test_prompt_finqa(raw_dataset[i],'',[],tf)
    inputs = gpt2tokenizer(prompt)
    prompt_length = len(inputs['input_ids'])
    return max_model_length - max_code_length - prompt_length

def get_max_prompt_length_tatqa(i,test,test_dataframe,train,train_dataframe,text_filter_df,max_code_length,max_model_length=4096): 
    #4096 tokens for Codex
    context = test[test_dataframe.loc[i,'context_index']]
    instance = context['questions'][test_dataframe.loc[i,'instance_index']]
    tf = [int(t) for t in text_filter_df.iloc[:,i].dropna().to_list()]
    prompt = get_test_prompt_tatqa(instance,context,train,train_dataframe,[],tf)
    inputs = gpt2tokenizer(prompt)
    prompt_length = len(inputs['input_ids'])
    return max_model_length - max_code_length - prompt_length


def remove_invalid_scripts_finqa(dataset):
    '''
    Remove train instances that have not been correctly processed and do not constitute correct examples for FinQA dataset
    '''
    #Filter out train examples that have not been correctly converted to DSL
    wrong_program_templates =  ['add(X_0,X_1),add(X_2,#0),add(#1,constant),divide(#2,constant)',
                       'add(X_0,X_1),add(#0,X_2),add(#1,constant),divide(#2,constant)',
                       'divide(X_0,X_1),divide(#0,X_1)',
                       'add(X_0,X_1),add(#0,constant),divide(#1,constant)'
                      ]
    prompt = [get_prompt_instance_finqa(dataset[i]) for i in range(len(dataset))]
    wrong_programs = [i for i in range(len(dataset)) if get_program_template(dataset[i]['qa']['program']) in wrong_program_templates ]
    table_golds = [i for i in range(len(dataset)) if 'table_' in [g[:-1] for g in dataset[i]['qa']['gold_inds'].keys()]]
    no_text_golds = [i for i in range(len(dataset)) if not 'text_' in [g[:-1] for g in dataset[i]['qa']['gold_inds'].keys()]]  #Examples that do not use text
    #Problem needs table but no query provided
    filtered_instances = [p for p in table_golds if not 'table_query_0' in prompt[p]] 
    #The problem is table only but variables are instantiated
    filtered_instances += [p for p in range(len(prompt)) if 'text_variable_' in prompt[p] and p in no_text_golds and p not in filtered_instances]
    #An error occured in the prompt generation
    filtered_instances += [p for p in range(len(prompt)) if not 'ans' in prompt[p] and not p in filtered_instances]
    #Remove filtered insances and wrong programs 
    instances_to_keep = [p for p in range(len(prompt)) if p not in filtered_instances+wrong_programs]
    return instances_to_keep


def remove_invalid_scripts_tatqa(dataset,dataframe):
    '''
    Remove train instances that have not been correctly processed and do not constitute correct examples
    '''
    #Filter out train examples that have not been correctly converted to DSL
    prompt = [get_prompt_instance_tatqa(dataset[dataframe.loc[i,'context_index']]['questions'][dataframe.loc[i,'instance_index']],dataset[dataframe.loc[i,'context_index']]) for i in range(len(dataframe))]
    table_golds = [i for i in range(len(dataframe)) if dataframe.loc[i,'modality'] in ['table','table_text']]
    no_text_golds = [i for i in range(len(dataframe)) if not dataframe.loc[i,'modality'] in ['text','table_text']]  #Examples that do not use text
    #Problem needs table but no query provided
    filtered_instances = [p for p in table_golds if not 'table_query_0' in prompt[p]] 
    #The problem is table only but variables are instantiated
    filtered_instances += [p for p in range(len(prompt)) if 'text_variable_' in prompt[p] and p in no_text_golds and p not in filtered_instances]
    #An error occured in the prompt generation, 'ans' has not been generated
    filtered_instances += [p for p in range(len(prompt)) if not 'ans' in prompt[p] and not p in filtered_instances]
    instances_to_keep = [p for p in range(len(prompt)) if p not in filtered_instances]
    return instances_to_keep