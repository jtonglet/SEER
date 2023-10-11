import pandas as pd
import json
import re
import textwrap
from sentence_transformers import SentenceTransformer
from sentence_transformers import  util
import spacy
import torch
from tqdm import tqdm

def load_file(path,encoding='utf-8'):
    #Load json file from path
    if '.jsonl' in path:
        with open(path, 'r', encoding=encoding) as f:
            data = [json.loads(line) for line in f]
    else:
        file = open(path,encoding=encoding)
        data = json.load(file)
    return data


def save_file(data,path):
    #Save json file
    with open(path, 'w') as f:
        json.dump(data, f)


def json_to_pandas(instance):
    '''
    Converts a raw table of a FinQA instance to a Pandas table with cleaned cells and headers structure.
    '''
    table_json = instance['table']
    #Verify if the table has a header
    if len(table_json[0]) == 2 : 
        if re.sub("[$\-\s\(\).]","",table_json[0][1]).isdigit():
            header = False
        else:
            header = True
    elif len(table_json) ==1: 
        header=False
    else:
        #The first row is a header
        header = True
    df =  pd.DataFrame(table_json)
    #Set header
    if header:
        df.columns = df.iloc[0]
        df = df[1:]
    else :
        if df.shape[1]==2:
            df.columns = ['Index','Values']
        else:
            df.columns = ['Col_'+str(i) for i in range(df.shape[1])]
    #Check if no columns have the same name
    if len(df.columns.unique()) < df.shape[1]:
        new_col = [v for v in df.columns]
        count_dict = {}
        for i in range(len(df.columns)):
            if df.columns[i] in df.columns.unique():
                if df.columns[i] in count_dict.keys():
                    count_dict[df.columns[i]] += 1
                else: 
                    count_dict[df.columns[i]] = 1
            if count_dict[df.columns[i]] >=2:
                new_col[i] = df.columns[i] + '_' + str(count_dict[df.columns[i]])
        df.columns = new_col
    #Set Index as first column
    df = df.set_index(df.iloc[:,0])
    #Check if no rows have the same name
    df = df.iloc[:,1:]
    if len(df.index.unique()) < df.shape[0]:
        new_index = [v for v in df.index]
        count_dict = {}
        for i in range(len(df.index)):
            if df.index[i] in df.index.unique():
                if df.index[i] in count_dict.keys():
                    count_dict[df.index[i]] += 1
                else: 
                    count_dict[df.index[i]] = 1
            if count_dict[df.index[i]] >=2:
                new_index[i] = df.index[i] + '_' + str(count_dict[df.index[i]])
        df.index = new_index
    #Process numeric values in table cells to a standardized format
    numeric_df = df.applymap(table_string_to_numeric_finqa)
    return numeric_df


def tatqa_table_to_pandas(context):
    '''
    Converts a raw table of a TAT-QA context to a Pandas table with cleaned cells and headers structure.
    '''
    raw_table = pd.DataFrame(context['table']['table'])
    try:
        # Identify the number of header levels
        index_name = []
        row_to_drop = []
        num_levels=1
        for r in range(raw_table.shape[0]):
            if len([c for c in raw_table.iloc[r,:] if c!=''])==1:
                num_levels=r+2
            else:
                break
        if num_levels==1:   
            for r in range(len(raw_table.iloc[:,0])-1):
                if raw_table.iloc[r,0]=='' or (raw_table.iloc[r,0]!='' and raw_table.iloc[r+1,0]==''):
                    num_levels=r+2
                else:
                    break
        for i in range(num_levels):
            if len([cell for cell in raw_table.iloc[i,:] if cell != ''])==1: 
                index_name.append([cell for cell in raw_table.iloc[i,:] if cell != ''][0])
                row_to_drop.append(i)
            else:
                for j in range(1,len(raw_table.iloc[i,:])):
                    if raw_table.iloc[i,j]=='':
                        raw_table.iloc[i,j]= raw_table.iloc[i,j-1]
        header_lists = [[] for _ in range(num_levels)]
        for i in range(num_levels):
            if i not in row_to_drop:
                for j in range(len(raw_table.columns)):
                    header_lists[i].append(raw_table.iloc[i,j])
                
        header_lists = [l for l in header_lists if len(l)!=0]
        multi_index = pd.MultiIndex.from_arrays(header_lists)
        df = raw_table.iloc[num_levels:,:]
        df.columns = df.iloc[0,:]
        df.index = df.iloc[:,0]
        df.index.name = ' '.join(index_name)
        df.columns = multi_index
        df.columns = ['_'.join(col) for col in df.columns.values]
        df = df.iloc[:,1:]
        #Check for columns with same names
        if len(df.columns.unique()) < df.shape[1]:
            new_col = [v for v in df.columns]
            count_dict = {}
            for i in range(len(df.columns)):
                if df.columns[i] in df.columns.unique():
                    if df.columns[i] in count_dict.keys():
                        count_dict[df.columns[i]] += 1
                    else: 
                        count_dict[df.columns[i]] = 1
                if count_dict[df.columns[i]] >=2:
                    new_col[i] = df.columns[i] + '_' + str(count_dict[df.columns[i]])
            df.columns = new_col
        #Check for index with same names
        if len(df.index.unique()) < df.shape[0]:
            new_index = [v for v in df.index]
            count_dict = {}
            for i in range(len(df.index)):
                if df.index[i] in df.index.unique():
                    if df.index[i] in count_dict.keys():
                        count_dict[df.index[i]] += 1
                    else: 
                        count_dict[df.index[i]] = 1
                if count_dict[df.index[i]] >=2:
                    new_index[i] = df.index[i] + '_' + str(count_dict[df.index[i]])
            df.index = new_index
        #Process numeric values in table cells to a standardized format
        numeric_df = df.applymap(table_string_to_numeric_tatqa)
    except :
        numeric_df = raw_table
    return numeric_df


def table_string_to_numeric_finqa(cell):
    '''
    Process string values in table cells.
    '''
    try:
        if not type(cell)==int and not type(cell)==float:
            if cell[0]=='$':
                #Remove commas from numbers representing money amounts
                cell = cell.replace(',','')
            cleaned_cell = re.sub("[<$\s]", "", cell).split('(')[0]
            #Remove some noisy character values
            for w in ['months','years','pt','million','billion','x']:
                cleaned_cell = cleaned_cell.replace(w,'')
            if cleaned_cell in ['','-','—','NM','n/a','nm','***','na','none','*','--']: 
                #Standardize the representation of None
                return None
            if ''.join([c for c in cleaned_cell if c not in ['%','$','-']]).replace('.','',1).isdigit() and '-' not in cleaned_cell[1:]:  
                if cleaned_cell[-1]=='%': 
                    #Convert percentages
                    return float(cleaned_cell[:-1])/100
                else:
                    return float(cleaned_cell)
            else: 
                return cell
        else:
            return cell
    except:
        return cell


def table_string_to_numeric_tatqa(cell):
    '''
    Process string values in table cells.
    '''
    try:
        if not type(cell)==int and not type(cell)==float:
            cell = cell.replace(',','')
            cleaned_cell = re.sub("[<$\s]", "", cell)
            for w in ['months','years','pt','million','billion','x','bps']:
                cleaned_cell = cleaned_cell.replace(w,'')
            if cleaned_cell in ['','-','—','NM','n/a','nm','***','na','none','*','--']: 
                #Various symbols used by tables to represent NaN values
                return None
            if ''.join([c for c in cleaned_cell if c not in ['(',')','%','$','-']]).replace('.','',1).isdigit() and '-' not in cleaned_cell[1:]:  
                if ')' in cleaned_cell and '%' in cleaned_cell:
                    return -float(cleaned_cell[1:-2])/100
                elif cleaned_cell[-1]=='%': 
                    return float(cleaned_cell[:-1])/100
                elif cleaned_cell[0]=='(':
                    return -float(cleaned_cell[1:-1])
                else:
                    return float(cleaned_cell)
            else: 
                return cell
        else:
            return cell
    except:
        return cell


def numerical_values_to_retrieve_finqa(instance):
    #Extract numerical values from the program of a FinQA instance
    program = instance['qa']['program']
    if  'table_' in program and program.count('(')==1 :
        values = []
    else:
        values =  re.sub("[^0-9.,\-\_#]","",program).split(',')
        values = [float(v) for v in values if '#' not in v and '_' not in v and v!='']
    return values


def numerical_values_to_retrieve_tatqa(instance): 
    #Extract numerical values from the program of a TAT-QA instance.
    program = instance['derivation'].replace(',','') 
    match = re.search(r'\(([\d,]+)\)', program)
    if match:
        program = program.replace(match.group(0),'(-'+match.group(0)[1:-1]+')')
    values =  re.sub("[^0-9.]"," ",program).split(' ')
    values = [float(v.replace(' ','')) for v in values if v not in ['','.']]
    return values


def row_values_to_retrieve_finqa(instance):
    #Extract column names from the program of FinQA instances (for table operators).
    program = instance['qa']['program']
    step = program.split('),')
    table_step = [s for s in step if  'table_' in s]
    row = [ s.split('(')[1].split(',')[0] for s in table_step]
    return row


def gold_inds_to_variables_finqa(instance,table):
    gold_inds = instance['qa']['gold_inds']
    queries = {}
    #Get the list of values to retrieve from text or table
    program_values = numerical_values_to_retrieve_finqa(instance)
    program_values += row_values_to_retrieve_finqa(instance)
    for ind in gold_inds:
        type_ind, pos_ind = ind.split('_')
        if not 'Values' in table.columns:
            pos_ind = int(pos_ind) - 1
        else:
            pos_ind = int(pos_ind)
        if type_ind=='table':
            #Table gold ind
            for v in program_values:
                if v == table.index[pos_ind]: 
                    #table operator, retrieve a whole row
                    queries[v] = 'table.loc["' + v + '",:]'
                if v in table.iloc[pos_ind].to_list()  and v not in queries.keys(): 
                    #retrieve a specific cell
                    queries[v]=v
                if v in [cell*100 for cell in table.iloc[pos_ind].to_list() if cell is not None]  and v not in queries.keys(): 
                    #the value is a percentage
                    queries[v] = v/100
    return queries


def gold_inds_to_variables_tatqa(instance,table): 
    #Get the list of values to retrieve from text or table
    program_values = numerical_values_to_retrieve_tatqa(instance)
    queries={}
    for v in program_values:
        if v in table.values in table.values  and v not in queries.keys():   
            queries[v] = v
        if v/100 in table.values in table.values  and v not in queries.keys():           
            queries[v] = v/100
    return queries


def DSL_to_code_finqa(instance,table):
    '''
    Convert the DSL numerical program of FinQA to python code.
    '''
    program = instance['qa']['program'].split('),')
    op_count = 0
    var_count = 0
    query_count = 0
    code_query = "#Define Pandas queries based on available row and column names\n" 
    code_var = "#Define variables based on available values in the text\n"
    code_op= "#Write a reasoning program \n" 
    query_var_dict = {}
    queries = gold_inds_to_variables_finqa(instance,table)
    #Assign queries to variables
    for _,v in queries.items():
        query_var_dict[v] = "table_query_" + str(query_count) 
        code_query += "table_query_" + str(query_count) + ' = ' + str(v) + '\n'
        query_count +=1
    #Generate the code step by step
    for s in range(len(program)):
        if 'table_' in program[s]:
            #Table operator
            step_operator = get_operator(program[s].split('(')[0].replace(' ',''))
            step_operand_1 = query_var_dict[queries[program[s].split('(')[1].split(',')[0]]]
            if s==len(program)-1:
                code_op += 'ans = ' + step_operand_1 +step_operator + " \n"
            else:
                code_op += 'step_' + str(op_count) + ' = ' + step_operand_1 +step_operator + " \n"
        else:
            step = re.sub("[\(\),]"," ",program[s]).split()
            if 'const_' in step[1]:
                step_operand_1 = step[1].split('_')[1]
                if step_operand_1 == 'm1':  
                    #Special case where the constant m1 in DSL represents -1
                    step_operand_1 = '-1'
            elif '#' in step[1]:
                step_operand_1 = 'step_' +  step[1].replace('#','')
            else: 
                if '%' in step[1]:
                    value = float(step[1].replace('%',''))/100
                else:
                    value = float(step[1])
                if value in queries.keys():
                    step_operand_1 = query_var_dict[queries[value]]
                else: 
                    code_var += 'text_variable_' + str(var_count) + ' = ' + str(value) + '\n'
                    step_operand_1 = 'text_variable_' + str(var_count)
                    var_count +=1
            if 'const_' in step[2]:
                step_operand_2 = step[2].split('_')[1]
                if step_operand_2 == 'm1':
                    step_operand_2 = '-1'
            elif '#' in step[2]:
                step_operand_2 = 'step_' +  step[2].replace('#','')
            elif step[2] =='none':
                step_operand_2 = 'average'
            else: 
                if '%' in step[2]:
                    value = float(step[2].replace('%',''))/100
                else:
                    value = float(step[2])
                if value in queries.keys(): 
                    step_operand_2 = query_var_dict[queries[value]]
                else: 
                    code_var += 'text_variable_' + str(var_count) + ' = ' + str(value) + '\n'
                    step_operand_2 = 'text_variable_' + str(var_count)
                    var_count +=1             
        #Define operator
            step_operator = get_operator(step[0])
            if s==len(program)-1: 
                #Last step
                code_op += 'ans = '   + step_operand_1  + step_operator + step_operand_2 + '\n'
            else:
                code_op += 'step_' + str(op_count) + ' = '   + step_operand_1  + step_operator + step_operand_2 + '\n'       
        op_count +=1
    #remove unused sections of the code
    if query_count==0:
        code_query = ''
    if var_count==0:
        code_var= ''  
    return code_query + code_var +code_op    


def DSL_to_code_tatqa(instance,table): 
    '''
    Convert the DSL numerical program of TAT-QA to python code.
    '''
    program = instance['derivation']
    match = re.search(r'\(([\d,]+)\)', program)
    if match:
        program = program.replace(match.group(0),'(-'+match.group(0)[1:-1]+')')
    var_count = 0
    query_count = 0
    code_query = "#Define Pandas queries based on available row and column names\n" 
    code_var = "#Define variables based on available values in the text\n"
    code_op= "#Write a reasoning program \n" 
    query_var_dict = {}
    if not instance['answer_type']=='count':
        #Retrieve all numerical values
        queries = gold_inds_to_variables_tatqa(instance,table)  
    else:
        queries = {}
    #Assign queries to variables
    for _,v in queries.items():
        query_var_dict[v] = "table_query_" + str(query_count) 
        code_query += "table_query_" + str(query_count) + ' = ' + str(v) + '\n'
        query_count +=1
    if instance['answer_type'] == 'arithmetic':
        program = ''.join([c for c in program if c not in [',', ',','$','%']])
        program=program.replace('[','(').replace(']',')')
        operands = [o.replace(' ','') for o in re.split('[-/*+()]',program) if o.replace(' ','') not in ['']]
        operands = list(set(operands))
        operands.sort(key=lambda s:len(s))
        operands = operands[::-1]
        for o in range(len(operands)) :
            if type(operands[o])==str:
                if 'thousand' in operands[o]:
                    operands[o] =  float(operands[o].split('t')[0])
                elif 'million' in operands[o]:
                    operands[o] = 1e6 * float(operands[o].split('m')[0])
                else:
                    pass
        for o in range(len(operands)): 
            if table_string_to_numeric_tatqa(operands[o]) in queries.keys():
                program = program.replace(operands[o],str(query_var_dict[queries[table_string_to_numeric_tatqa(operands[o])]]))
            elif table_string_to_numeric_tatqa(operands[o]) >=4:
                code_var += 'text_variable_' + str(var_count) + ' = ' + str(table_string_to_numeric_tatqa(operands[o])) + '\n'
                text_variable = 'text_variable_' + str(var_count)
                var_count +=1
                program = program.replace(operands[o],text_variable)
            else:
                pass
        code_op += 'ans = '   + program + '\n'
        code_op += 'scale = "'+instance['scale']+'"\n'
        if query_count==0:
            code_query = ''
        if var_count==0:
            code_var= ''  
        return code_query + code_var +code_op  
    else: 
        #Span, MultiSpan or Count program
        comment = ''
        if instance['derivation']!='':
            comment = '#' + instance['derivation'] +'\n'
        answer = instance['answer']
        if type(answer)==list and len(answer)==1:
            answer = table_string_to_numeric_tatqa(answer[0])
            if answer in queries.keys(): 
                answer = queries[answer]  
        if query_count==0:
            code_query=''
        return code_query + code_op +comment + 'ans = "' + str(answer) + '"\n'


def get_operator(operator_DSL):
    #Map an operator in DSL to the corresponding operator in Python
    d = {'add':' + ','subtract':' - ','multiply':' * ','divide':' / ','exp':'**','greater':' > ',
    'table_max':'.max()','table_min':'.min()','table_average':'.mean()','table_sum':'.sum()'}
    return d[operator_DSL]

            
def get_table_description(table):  
    #Represent a Pandas table as a string
    description = "Table: \n"
    index_array = table.index.fillna('').values.astype(str).tolist()
    column_names =  [table.index.name] + table.columns.fillna('').values.astype(str).tolist()
    if type(column_names[0])!= str:
        column_names[0] = ''
    table_array = table.to_numpy()
    # Join array elements with delimiter
    table_string = ' | '.join(column_names) + '\n' + '\n'.join([f"{index} | {' | '.join(row.astype(str))}" for index, row in zip(index_array, table_array)])
    return description + table_string + '\n'


def create_python_script(code,i=0,dataset=''):
    '''
    Generate an executable script from the generated code fragment.
    '''
    s= ''''''
    if 'table.' in code:
        s+= '#Import packages\nfrom utils import load_file, json_to_pandas\n\n' 
    s+= 'def solve():\n'
    if 'table.' in code:
        s+= '   #Load table\n'
        s+= '   table = json_to_pandas(instance)\n'
    code = textwrap.indent(code, 3 * ' ')
    s+=code  + '\n'
    s+= '   return ans\n'
    if 'table.' in code:
        s+= '\ninstance = load_file("'+dataset+'")['+str(i)+']\n'
    s+= 'output = solve()\n'
    return s


def ans_edit(script):
    #Try to replace the last step by ans
    last_step = script.split('\n')[-5].split('=')[0].replace(' ','')
    edited_script = script
    if last_step != 'ans':
        edited_script =  edited_script.replace(last_step,'ans') 
    return edited_script


def values_column(script,tab,*kwargs):
    #Replace the chosen column name by the single column available
    # Define the regex pattern
    edited_script=script
    if len(tab.columns)==1: #single  column dataframe
        pattern = r'table\.loc\[(\".*?\"\s*,\s*)\".*?\"\]'
        col_name= tab.columns[0]
        replacement = r'table.loc[\1"%s"]'%col_name
        edited_script = re.sub(pattern, replacement, script)
    return edited_script

def is_float(s):
    #Evaluate if a string s is a float
    try:
        float(s)
        return True
    except:
        return False
    
def BERT_NER_tagging(text,bert_model):
    ner_results = bert_model(text)
    entity_dict = {}
    for e in range(len(ner_results)): 
        if ner_results[e]['entity'][0]=='B': 
            #It marks the beginning of entity
            entity_type = ner_results[e]['entity'][2:]
            if e==len(ner_results)-1:
                entity_text = text[ner_results[-1]['start']:ner_results[-1]['end']]
                entity_dict[entity_text] = entity_type
            else:
                start = ner_results[e]['start']
                for f in range(e+1,len(ner_results)):
                    if ner_results[f]['entity'][0]=='B':
                        end = ner_results[f-1]['end'] 
                        entity_text = text[start:end]
                        entity_dict[entity_text] = entity_type
                        break
                    if f == len(ner_results)-1: 
                        end = ner_results[f]['end']
                        entity_text = text[start:end]
                        entity_dict[entity_text] = entity_type
    return entity_dict


def preprocess_text(text,
                    spacy_model,
                    bert_model=None,
                    ner_mask=True, 
                    ner_tags = ['ORG','GPE','LOC','NORP','PERSON','DATE','MONEY','CARDINAL','PERCENT']):
    '''
    Convert a question text by lowering, removing digits, punctuation, and optionnaly replacing entities by their NER tags.
    '''
    text = text.lower() 
    tokens = spacy_model(text) 
    processed_text = ' '.join([token.text for token in tokens if not token.is_punct])
    if ner_mask:
        bert_entities = BERT_NER_tagging(processed_text,bert_model)  
        #First apply the spacy mask
        for ent in tokens.ents:
            if ent.label_ in ner_tags:
                processed_text = processed_text.replace(ent.text,ent.label_)
        #Then the NER BERT mask
        for k,v in bert_entities.items():
            processed_text = processed_text.replace(k,v)       
    return processed_text


def get_program_template(program,normalize_const=True):
    tokens = program.replace('(',' ( ').replace(')',' ) ').replace(',',' , ').split(' ')
    var_idx = {}
    count = 0
    for t in range(len(tokens)):
        if tokens[t].replace('.','').replace('-','').replace('%','').isdigit():
            if tokens[t] not in var_idx.keys():
                var_idx[tokens[t]] = count
                count += 1
            tokens[t] = 'X_' + str(var_idx[tokens[t]])
        if 'table_' in tokens[t-2]:  #Table operator
            if tokens[t] not in var_idx.keys():
                var_idx[tokens[t]] = count
                count +=1
            tokens[t] = 'X_' + str(var_idx[tokens[t]])
            count_table = t + 1
            while ')' not in tokens[count_table]:
                tokens[count_table] = ''
                count_table += 1
        if 'const_' in tokens[t] and normalize_const:
            tokens[t] = 'constant'    
    template = ''.join([t for t in tokens])        
    return template


def get_sentence_embeddings(corpus,embedding_model,progress_bar=False): #all-MiniLM-L12-v2 , all-mpnet-base-v2 
    embedder = SentenceTransformer(embedding_model)
    corpus_embeddings = embedder.encode(corpus,show_progress_bar=progress_bar)
    return corpus_embeddings


def retrieve_top_k_text_facts_finqa(data,k=10):
    spacy_model = spacy.load("en_core_web_lg") #Requires to first install en_core_web_lg
    top_results = pd.DataFrame()
    query_embeddings = get_sentence_embeddings([data[i]['qa']['question'] for i in range(len(data))],'all-MiniLM-L6-v2')  
    for i in tqdm(range(len(query_embeddings))):
        context = get_context_corpus_finqa(data,i,spacy_model)
        context_embeddings = get_sentence_embeddings(context,'all-MiniLM-L6-v2')
        cos_scores = util.cos_sim(query_embeddings[i], context_embeddings)[0]
        query_results =  torch.topk(cos_scores, k=min(len(context),k)).indices.tolist()
        if k > len(context):
            query_results += [None for _ in range(k-len(context))]
        top_results[i] = query_results
    return top_results 


def retrieve_top_k_text_facts_tatqa(data,dataframe,k=10):
    spacy_model = spacy.load("en_core_web_lg")
    top_results = pd.DataFrame()
    query_embeddings = get_sentence_embeddings([dataframe.loc[i,'question'] for i in range(len(dataframe))],'all-MiniLM-L6-v2')  
    for i in tqdm(range(len(query_embeddings))):
            j = dataframe.loc[i,'context_index']
            context = get_context_corpus_tatqa(data,j,spacy_model)
            context_embeddings = get_sentence_embeddings(context,'all-MiniLM-L6-v2')
            cos_scores = util.cos_sim(query_embeddings[i], context_embeddings)[0]
            query_results =  torch.topk(cos_scores, k=min(len(context),k)).indices.tolist()
            if k > len(context):
                query_results += [None for _ in range(k-len(context))]
            top_results['-'.join([str(i),str(j)])] = query_results
    return top_results 

def get_context_corpus_finqa(data,idx, spacy_model):
    '''
    Preprocess the context text and concatenate the pre and post parts.
    '''
    pre_text = data[idx]['pre_text']
    post_text = data[idx]['post_text']
    context_corpus = [preprocess_text(t,spacy_model,ner_mask=False) for t in pre_text]  #No NER masking needed here
    context_corpus += [preprocess_text(t,spacy_model,ner_mask=False) for t in post_text]
    return context_corpus

def get_context_corpus_tatqa(data,idx, spacy_model):
    context_corpus = [preprocess_text(p['text'],spacy_model,ner_mask=False) for p in data[idx]['paragraphs']]
    return context_corpus