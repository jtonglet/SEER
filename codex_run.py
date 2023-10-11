import json
import time
import os
import openai
from utils import create_python_script
from evaluation import evaluate_script, execute_code, evaluate_script_tatqa


def few_shot_prompting(prompt,model='code-davinci-002', 
                       max_tokens = 256,
                       temperature=0.0,  
                       n=1, 
                       stop=['\n\n','Read the context','Read the following','Modality'],
                       key="OPENAI_API_KEY",
                       sleep=5):
    '''
    Generate code output given a prompt using the OpenAI Code Completion API (code-davinci-002).
    This API is now deprecated and only accessible through the Research Access Program : 
    See  https://platform.openai.com/docs/guides/code) for a complete guide on OpenAI API Chat endpoints.
    Params:
        messages (list) : list of messages. Each message is a dictionary   
        max_tokens (int) : the maximum token length of the output.
        temperature (float) : the temperature of the LLM, higher value results in less deterministic behavior.
        n (int) :  number of responses to generate.
        key (str) : API key identifier. Refers to the name of an environment variable on the user's device.
        sleep (float) : waiting time between 2 API calls.
    '''   
    #Retrieve key from environment variables
    openai.api_key = os.getenv(key)
    response = False
    start_time = time.time()
    while not response:
        try:
            response = openai.Completion.create(model=model,prompt=prompt,
                                                 max_tokens=max_tokens,temperature=temperature,n=n,
                                                 stop=stop)
        except Exception:
            time.sleep(sleep)  
        if time.time() -start_time >= 300:
            response={}
            response['correct'] = 'Time out error.'
            break
    response['time_taken'] = time.time() - start_time
    return response


def few_shot_chat_prompting(messages,model='gpt-3.5-turbo',
                            max_tokens = 256, 
                            temperature=0.0, 
                            n=1,  
                            key="OPENAI_API_KEY",
                            sleep=5):
    '''
    Generate code output given a prompt using the OpenAI Chat API (gpt-3.5-turbo model).
    Following the deprecation of the CODEX API, we propose an alternative in the form of the CHAT API (gpt-3.5-turbo)
    See  (https://platform.openai.com/docs/guides/gpt/chat-completions-api) for a complete guide on OpenAI API Chat endpoints.
    Params:
        messages (list) : list of messages. Each message is a dictionary   
        max_tokens (int) : the maximum token length of the output.
        temperature (float) : the temperature of the LLM, higher value results in less deterministic behavior.
        n (int) : number of responses to generate
        key (str) : API key identifier. Refers to the name of an environment variable on the user's device.
        sleep (float) : waiting time between 2 API calls.
    '''
    #Retrieve key from environment variables
    openai.api_key = os.getenv(key)
    response = False
    start_time = time.time()
    while not response:
        try:
            response = openai.ChatCompletion.create(model=model,messages=messages,
                                                 max_tokens=max_tokens,temperature=temperature,n=n)
        except Exception:
            time.sleep(sleep)
        if time.time() -start_time >= 180:
            response={}
            response['correct'] = 'Time out error.'
            break
    response['time_taken'] = time.time() - start_time
    time.sleep(sleep)  #Cooldown between each API call
    return response


def save_result_finqa(idx,dataset,dataset_path,result,output_path="output/finqa/predictions/run1/",selection=[],print_eval=False):
    result['question'] = dataset[idx]['qa']['question']
    result['gold_answer'] = dataset[idx]['qa']['exe_ans']
    result['code'] = result['choices'][0]['text']
    script = create_python_script(result['code'],idx,dataset_path)
    try:
        result['prediction'] = execute_code(script)
    except:
        print(script)
    result['correct'] =  evaluate_script(dataset[idx],script)  
    if print_eval:
        print('Instance %s is %s'%(idx,result['correct']))
    result['few_shot'] = selection
    #Serializing
    json_object = json.dumps(result,indent = 4)
    #Json output
    with open(output_path+"sample%s.json"%idx, "w") as outfile:
        outfile.write(json_object)

def save_result_tatqa(idx,dataset,dataframe,result,output_path="output/tatqa/predictions/run1/",selection=[],print_eval=False):
    context = dataset[dataframe.loc[idx,'context_index']]
    instance = context['questions'][dataframe.loc[idx,'instance_index']]
    result['question'] = instance['question']
    result['gold_answer'] = instance['answer']
    result['code'] = result['choices'][0]['text']
    script = create_python_script(result['code'])
    try:
        result['prediction'] = execute_code(script)
    except:
        print(script)
    result['correct'] =  evaluate_script_tatqa(instance,script)  
    if print_eval:
        print('Instance %s : EM = %s  , F1 = %s'%(idx,result['correct'][0],result['correct'][1]))
    result['few_shot'] = selection
    #Serializing
    json_object = json.dumps(result,indent = 4)
    #Json output
    with open(output_path+"sample%s.json"%idx, "w") as outfile:
        outfile.write(json_object)