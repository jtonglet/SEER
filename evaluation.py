import pandas as pd
import re
from utils import *
from tatqa_metric import get_metrics   
from evaluate_finqa import equal_program

################
#     FinQA    #
################

def compute_accuracy_finqa(dataset,path,n_runs=3):
    '''
    Compute the average execution accuracy over n_runs. Returns a dictionary with runs as keys and a tuple containing the EA and PA as values.
    Args:
        dataset (list) : raw dataset of FinQA
        path (str) : the path to the directory where the predictions are stored
        n_runs (int) : the number of "run" folders in the predictions directory
    Returns:
        accuracy_dict (dict) : a dictionary with model runs as keys and (EA,PA) tuples as values
    '''
    accuracy_dict = {}
    for i in range(1,n_runs+1):
        run_results = [load_file(path+'/run%s/'%(i)+'sample%s.json'%(r)) for r in range(len(dataset))]
        EA = sum([1 if result['correct']==True else 0 for result in run_results])/len(dataset)  
        PA = get_program_accuracy(dataset,run_results)/len(dataset)
        accuracy_dict['run%s'%(i)] = (EA,PA)
    return accuracy_dict


def map_string(s):
    '''
    Convert  the python output of the LLM to  the FinQA DSL language to compare program accuracy
    '''
    operators_dict = {'-':'subtract','/':'divide','+':'add','*':'multiply','>':'greater','<':'smaller','**':'exp'}
    pattern = r"(table_query_\d+|text_variable_\d+) = (-?\d+\.\d+|\d+)"
    variables = re.findall(pattern, s)
    var_dict = {name: float(val) for name, val in variables}
    for name, val in var_dict.items():
        s = s.replace(name, str(val))
    s = s.split('#Write')[1]
    steps = s.split('\n')[1:]
    for o in range(len(steps)):
            steps[o] = steps[o].split('=')[1]
            if len(steps[o].split(' ')) >=4:
                operator = steps[o].split(' ')[2]
                operator = operators_dict[operator.replace(' ','')]
                operand_1 = steps[o].split(' ')[1]
                if 'step' in operand_1:
                    operand_1 = '#'+operand_1[-1]
                elif float((operand_1)).is_integer():
                    operand_1 = str(int(float(operand_1)))
                else:
                    pass
                operand_2 = steps[o].split(' ')[3] 
                if 'step' in operand_2:
                    operand_2 = '#'+operand_2[-1]
                elif float((operand_2)).is_integer():
                    operand_2 = str(int(float(operand_2)))
                else:
                    pass
                steps[o] =operator+'('+ operand_1   + ', ' + operand_2 + ')'
            else:
                pass
    new_s = ', '.join(steps)  
    if new_s[-9:] == ',  step_0':
        new_s = new_s[:-9]   
    return new_s


def gold_program_eval(gold_program):
    '''
    Make edits to the gold program to make it more generalizable
    '''
    #Convert to correct format by removing the const_ in front of the constants values
    pattern = r'const_([\d.]+)'
    gold_program =  re.sub(pattern, lambda x: x.group(1).lower(), gold_program)
    #Remove 0 at the end of floating numbers
    decimal_pattern = re.compile(r'(\.\d*?)0+\b')
    gold_program = decimal_pattern.sub(r'\1', gold_program)
    gold_program =  re.sub(r'(?<=\d)\.(?=\s|\)|,|#\d)', '', gold_program)
    #Convert percentages 
    percentages = re.findall(r'\d+\.\d+%', gold_program) + re.findall(r'\d+\.%', gold_program) + re.findall(r'\d+%', gold_program)
    decimal_vals = [str(float(p.strip('%')) / 100) for p in percentages]
    for p in range(len(percentages)):
        gold_program = gold_program.replace(percentages[p],decimal_vals[p])
    return gold_program


def get_program_accuracy(dataset,predictions):
    '''
    Compute the program accuracy for a dataset and the corresponding predictions.
    '''
    correct = 0
    for i in range(len(dataset)):
        if predictions[i]['correct']==True: #No point in evaluating if incorrect
            predicted_code = predictions[i]['choices'][0]['text']
            table = json_to_pandas(dataset[i])
            gold_code = DSL_to_code_finqa(dataset[i],table,query=False)
            if  predicted_code.replace('\n','') == gold_code.replace('\n','') :
                #If the code is identical stop here
                        correct +=1
            else:
                try:
                    pred_template = map_string(predicted_code)
                    gold_template = gold_program_eval(dataset[i]['qa']['program'])
                    if  pred_template == gold_template :
                        correct +=1
                    elif equal_program(pred_template,gold_template):  
                        #Test equivalence
                        correct +=1
                    else:  
                        pass
                except :
                    pass
    return correct


def get_answer(instance):
    #Extract answer from instance template
    #Convert yes/no answers to 1/0
    try:
        exe_ans = instance['qa']['exe_ans']
    except:
        exe_ans = instance['qa']['answer']
    if exe_ans=='yes':
        exe_ans=1
    if exe_ans=='no':
        exe_ans=0
    return exe_ans


def execute_code(script):
    results={}
    exec(script,results) 
    exec_output = results['output']
    return exec_output

        
def evaluate_answer(exec_output,correct_answer,threshold=0.001,percentage=True):
    '''
    Evaluate if an execution output yields the correct answer.
    Args:
        exec_output (str/int) : the output of the code execution
        correct_answer (str/int) : the correct answer 
        threshold (float) : the tolerance threshold of deviation between the predicted and correct answer
        percentage (bool) : whether to apply tolerance for percentages or not
    '''
    if type(exec_output)==str:
        if exec_output == correct_answer: 
            return True 
        elif exec_output in correct_answer:
            return True
        else:
            return 'Span to evaluate further'  
    else: 
        if abs(exec_output-correct_answer) <= threshold :
            return True
        #Percentage case 
        elif percentage and (abs((exec_output/100)-correct_answer) <= threshold or abs((exec_output*100)-correct_answer) <=threshold): 
            return True
        else:
            return False



           
def evaluate_script(instance,script,threshold=0.001):
    '''
    Evaluate if the output script of a LLM results in the correct answer after execution.
    '''
    correct_answer = get_answer(instance)
    try:
        exec_output = execute_code(script)
        evaluation = evaluate_answer(exec_output,correct_answer,threshold)
        return evaluation
    except Exception as e : 
        #If the code did not execute, try to replace the last variable by 'ans' and the last statement by 'return ans' 
        edited_script = script
        modifications = [ans_edit]
        for m in modifications:
            try:
                edited_script = m(script)
                exec_output = execute_code(edited_script)
                evaluation = evaluate_answer(exec_output,correct_answer,threshold)
                return evaluation
            except:
                pass 
        return 'Error: '+str(e)   


#####################
#       TATQA       #
#####################


def compute_metrics_tatqa(dataframe,
                          dataset,
                          path,
                          n_runs=(1,3)):
    '''
    Compute the EM and F1 over n_runs
    Args:
        dataframe (pandas.DataFrame) : the dataframe containing the TAT-QA metadata
        dataset (list) : the raw dataset
        path (str) : the path to the directory containing the predictions
        n_runs (tuple) : the starting index and last of index of the runs to evaluate
    Returns:
        accuracy_dict (dict) : a dictionary with model runs as keys and (EM,F1) tuples as values
    '''
    accuracy_dict = {}
    for i in range(n_runs[0],n_runs[1]+1):
        EM, F1  = compute_metrics_tatqa_dataset(dataframe,dataset,path+'/run%s/'%(i))
        accuracy_dict['run%s'%(i)] = (sum(EM)/len(dataframe),sum(F1)/len(dataframe))
    return accuracy_dict


def evaluate_script_tatqa(instance,script):
    correct_answer = instance['answer']
    correct_answer = preprocess_ground_truth(correct_answer)
    try:
        pred_scale = script.split('scale=')[1].split('\nans=')[0].replace(' ','')
    except:
        pred_scale = ''
    try:
        exec_output = execute_code(script)
        evaluation = evaluate_tatqa(exec_output,correct_answer,pred_scale)
        return evaluation
    except Exception as e :
        return str(e), ''     


def process_numeric_answer(answer,scale):
    #Convert float to percentages if predicted scale is percent
    if scale == 'percent':           
        answer *= 100
    #Apply rounding after the second decimal
    if len(str(answer).rsplit('.')[-1]) >2: 
        answer = round(answer,2)
    #Remove trailing 0s
    if '.0'  in str(answer).replace(' ','')[-2:] :
        answer= int(float(answer))
    return answer

    
def evaluate_tatqa(exec_output,answer,pred_scale):
    '''
    Takes an execution output and a TAT-QA instance and returns Exact Match and Numeracy Focused F1.
    '''
    #Case 1 : direct match
    if str(exec_output)==answer:
        EM, F1 =1, 1
    #Case 2 : no direct match
    else:
        elements_to_remove = [",","$","£","'","million","billion","years","%","m","`","(",")"]
        #The exec output is a string
        if type(exec_output)==str and exec_output !='':
            #Case when it is a list represented as a string that needs to be processed
            if  exec_output[0]=='[':
                exec_output = exec_output[1:-1].split("', '")
                for e in elements_to_remove:
                    exec_output = [s.replace(e,'')  for s in exec_output]
                for j in range(len(exec_output)):
                    #Convert floats in the list
                    if is_float(exec_output[j]):
                        exec_output[j] = process_numeric_answer(float(exec_output[j]),pred_scale)
            else:
                #Normal string
                for e in elements_to_remove:
                    exec_output = exec_output.replace(e,'')
                if is_float(exec_output):
                        exec_output = str(process_numeric_answer(float(exec_output),pred_scale))
        #The exec output is a list
        if type(exec_output)==list:
            for j in range(len(exec_output)):
                if is_float(exec_output[j]):
                    exec_output[j] = str(process_numeric_answer(float(exec_output[j]),pred_scale))
                if pred_scale == 'percent' :
                    exec_output[j] *= 100 
                if type(exec_output[j]) != str:
                    exec_output[j]=str(exec_output[j])
            if type(answer)==list:
                #Merge the execution output if all answers are concatenated in the ground truth answer
                if len(exec_output) > 1 and len(answer)==1:
                    exec_output = ' '.join(exec_output)
        #The execution output is a float
        if type(exec_output)==float:  #The instance is a true float
             #Multiply percentages to the correct scale
            exec_output = process_numeric_answer(exec_output,pred_scale)
        if type(exec_output)!= list:
            exec_output = str(exec_output)
        if type(answer)!= list:
            answer = str(answer)
        EM, F1 =  get_metrics(answer,exec_output)  
        
    return EM, F1


def preprocess_ground_truth(answer):
    #Provide some slack to the predictions of the LLM which did not learn some of the dataset syntax 
    elements_to_remove = [",","$","£","'","million","billion","years","%","m","`"]
    if type(answer)==str:
        for e in elements_to_remove:
            answer = answer.replace(e,'')
    if type(answer)==list:
        for j in range(len(answer)):
            for e in elements_to_remove:
                answer[j] = answer[j].replace(e,'')
            if '.0'  in str(answer[j]).replace(' ','')[-2:] and is_float(answer):
                #Remove trailing 0
                answer[j]= str(int(float(answer[j])))
    if type(answer)!=list:
        answer= str(answer)
    return answer


def compute_metrics_tatqa_dataset(dataframe,raw_dataset,path=''):
    '''
    Predictions is a list of predicted output.
    '''
    EM_total = []
    F1_total = []
    for i in dataframe.index.to_list():
        context = raw_dataset[dataframe.loc[i,'context_index']]
        instance = context['questions'][dataframe.loc[i,'instance_index']] 
        pred = load_file(path+'sample%s.json'%(i))
        code = pred['choices'][0]['text']
        script = create_python_script(code)
        results={}
        if 'scale =' in code:
            pred_scale = code.split('scale =')[1].replace('"','').replace(' ','')
        else:
            pred_scale = ''
        try:
            exec(script,results)  
            exec_output = results['output']
        except:
            exec_output = ''
        ground_truth = preprocess_ground_truth(instance['answer'])  
        EM, F1 = evaluate_tatqa(exec_output,ground_truth,pred_scale)
        EM_total += [EM]
        F1_total += [F1]
    return EM_total, F1_total