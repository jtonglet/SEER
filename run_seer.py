import argparse
import pandas as pd 
from codex_run import few_shot_prompting, save_result_finqa, save_result_tatqa
from generate_prompt import *
from seer import SEER 
from utils import load_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='datasets/finqa/train.json')
    parser.add_argument('--test_path', type=str, default='datasets/finqa/test.json')
    parser.add_argument('--train_metadata_path',type=str,default='data_cache/finqa/metadata/finqa_train_df.csv')
    parser.add_argument('--test_metadata_path',type=str,default='data_cache/finqa/metadata/finqa_test_df.csv')
    parser.add_argument('--text_retriever',type=str,default='data_cache/finqa/text_retriever/retrieved_text_finqa_test.csv')
    parser.add_argument('--similarity_matrix',type=str,default='data_cache/finqa/similarity_matrices/finqa_test_sim.txt')
    parser.add_argument('--max_output_length',type=int,default=309,help='Maximum length allocated to the output in tokens')
    parser.add_argument('--max_model_length',type=int,default=4096,help='Maximum token capacity (input+output) opf the LLM')
    parser.add_argument('--remove_invalid_train',type=bool,default=True,help='if True, remove train instances that were not converted correctly to python code')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--output_path',type=str,default="output/finqa/predictions/run1/")
    #SEER parameters
    parser.add_argument('--alpha',type=float,default=0.75)
    parser.add_argument('--beta',type=float,default=0)
    parser.add_argument('--modules',type=list,default=['modality'],choices=[['modality'],['modality','answer_type']])
    parser.add_argument('--n_exemplars', type=int, default=4, help='Number of n-shot training examples.') 
    parser.add_argument('--k',type=int,default=200)
    #CODEX parameters
    parser.add_argument('--key',type=str,default='OPENAI_API_KEY')
    parser.add_argument('--model', type=str, default='code-davinci-002') 
    parser.add_argument('--temperature', type=float, default=0.0)
    return parser.parse_args()



if __name__=='__main__':

    args = parse_args()
    #Load the data
    train = load_file(args.train_path)
    test = load_file(args.test_path)
    #Dataframes
    train_df = pd.read_csv(args.train_metadata_path)
    test_df = pd.read_csv(args.test_metadata_path)
    text_filter_df = pd.read_csv(args.text_retriever)
    similarity = pd.read_table(args.similarity_matrix,header=None,sep=' ') 
    if args.remove_invalid_train:
        if 'finqa' in args.train_path:
            valid_idx = remove_invalid_scripts_finqa(train)
        else:
            valid_idx =remove_invalid_scripts_tatqa(train,train_df)
    #Initialize SEER
    seer = SEER(k=args.k,M=args.n_exemplars,alpha=args.alpha,
                beta=args.beta,
                modules=args.modules)
    for i in range(len(test)):
        if 'finqa' in args.train_path:
            max_length = get_max_prompt_length_finqa(i,test,text_filter_df,args.max_output_length,args.max_model_length)
            selection = seer.get_few_shot_exemplars(i,similarity,train_df,test_df,valid_idx,max_length)
            text_filter = [int(t) for t in text_filter_df.iloc[:,i].dropna().to_list()]
            prompt = get_test_prompt_finqa(test[i],train,selection,text_filter)
            result = few_shot_prompting(prompt,key=args.key,model=args.model,temperature=args.temperature)
            save_result_finqa(i,test,args.test_path,result,
                        args.output_path,selection,print_eval=True)
        else:
            max_length = get_max_prompt_length_tatqa(i,test,test_df,train, train_df, text_filter_df,args.max_output_length,args.max_model_length)
            context = test[test_df.loc[i,'context_index']]
            instance = context['questions'][test_df.loc[i,'instance_index']]
            selection = seer.get_few_shot_exemplars(i,similarity,train_df,test_df,valid_idx,max_length)
            text_filter = [int(t) for t in text_filter_df.iloc[:,i].dropna().to_list()]
            prompt = get_test_prompt_tatqa(instance,context,train,train_df,selection,text_filter)
            result = few_shot_prompting(prompt,key=args.key,model=args.model,temperature=args.temperature)
            save_result_tatqa(i,test,test_df,result,
                            args.output_path,selection,print_eval=True)