import numpy as np
from constraint import Problem, MinConflictsSolver
from seer import *
from utils import *

SEED =42 

def get_candidates_fixed_set(n_candidates=20,num_exemplars=4,dataset_size=4000):
    #Independent of the test instance index
    np.random.seed(SEED)
    candidate_fixed_set = [list(np.random.randint(1, dataset_size, num_exemplars)) for _ in range(n_candidates)]
    return candidate_fixed_set


def get_random_set(instance_idx,valid_idx,num_exemplars=4): 
    instance_seed = SEED + instance_idx  
    #Make a unique instance seed based on global SEED + instance index
    np.random.seed(instance_seed) 
    few_shot_idx = np.random.choice(valid_idx,num_exemplars,replace=False)  
    return list(few_shot_idx)


def get_kate_set(instance_idx,similarity_matrix,valid_train_idx,num_exemplars):
    kate = SEER(k=num_exemplars)  
    #We only use the get KNN method of SEER
    few_shot_idx = kate.get_KNN(instance_idx,similarity_matrix,num_exemplars,valid_train_idx) 
    return few_shot_idx

def CSP(instance_idx,train_dataframe,test_dataframe,valid_train_idx,num_candidates=20,num_exemplar=4,alpha=0.5,beta=0.25,max_length=4096):
    modality = test_dataframe.loc[instance_idx,'modality']
    try:
        ans_type = test_dataframe.loc[instance_idx,'answer_type']
    except:
        ans_type = None
    #Initialize CSP instance
    problem = Problem(MinConflictsSolver())
    #Sample n candidates from the train set
    candidate_idx = np.random.choice(valid_train_idx,num_candidates,replace=False)
    # Define variables
    for i in candidate_idx:
        problem.addVariable(f"X_{i}", [0, 1])  #ILP
    def sum_X(*X):
        #Max M exemplars
        return sum(X) <= num_exemplar 
    def sum_length_X(*X):
        #Capacity constraint
        return sum([train_dataframe.loc[i,'token_prompt_length']*X[i] for i in range(len(X))]) <= max_length
    #Modality diversity
    def sum_table_only_X(*X):
        return sum([train_dataframe.loc[i,'table_only']*X[i] for i in range(len(X))]) >= num_exemplar *alpha
    def sum_text_only_X(*X):
        return sum([train_dataframe.loc[i,'text_only']*X[i] for i in range(len(X))]) >= num_exemplar * alpha
    def sum_has_table_X(*X):
        return sum([train_dataframe.loc[i,'use_table']*X[i] for i in range(len(X))]) >= num_exemplar * beta
    def sum_has_text_X(*X):
        return sum([train_dataframe.loc[i,'use_text']*X[i] for i in range(len(X))]) >= num_exemplar * beta
    def sum_is_hybrid_X(*X):
        return sum([train_dataframe.loc[i,'hybrid']*X[i] for i in range(len(X))]) >= num_exemplar * beta
    #Ans type diversity
    def sum_ans_type_X(*X):
        return sum([train_dataframe.loc[i,ans_type]*X[i] for i in range(len(X))]) >= num_exemplar * alpha
    def sum_other_ans_type_X(*X):
        other_answer_type = ['span','multi-span','arithmetic','count']
        other_answer_type.remove(ans_type)
        return (sum([train_dataframe.loc[i,other_answer_type[0]]*X[i] for i in range(len(X))]) 
                + sum([train_dataframe.loc[i,other_answer_type[1]]*X[i] for i in range(len(X))]) 
                + sum([train_dataframe.loc[i,other_answer_type[2]]*X[i] for i in range(len(X))]) >= num_exemplar * beta)
    #Add constraints to the problem
    problem.addConstraint(sum_X)
    problem.addConstraint(sum_length_X)
    if modality ==0:
        problem.addConstraint(sum_table_only_X)
        problem.addConstraint(sum_has_text_X)
    if modality ==1:
        problem.addConstraint(sum_text_only_X)
        problem.addConstraint(sum_has_table_X)
    if modality:
        problem.addConstraint(sum_has_table_X)
        problem.addConstraint(sum_has_text_X)
        problem.addConstraint(sum_is_hybrid_X)
    if ans_type in ['span','multi-span','arithmetic','count']:
        problem.addConstraint(sum_ans_type_X)
        problem.addConstraint(sum_other_ans_type_X)
    # Find all solutions
    solution = problem.getSolution()  
    #Returns one solution found to the problem
    selection = [int(k.split('_')[1]) for k,v in solution.items() if v==1 ]

    return selection
    