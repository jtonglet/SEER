import numpy as np
import pandas  as pd
from sentence_transformers import util
from utils import *
from tqdm import tqdm
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, GUROBI_CMD


def compute_similarity_matrix(train_questions,  
                              test_questions, 
                              embedding_model='all-MiniLM-L6-v2',
                              progress_bar=False,
                              save=True,
                              output_path='output/similarity_matrix.txt'):
    '''
    Generate  a similarity matrix between train and test instances based on the cosine similarity of their sentence embeddings
    Params:
        train_questions (list) : list of train set questions.
        test_questions (list) : list of test set questions.
        embedding_model (str) : the name of the chosen SBERT embedding model.
        progress_bar (bool) : if True, prints a progress bar while the embeddings are loading.
        save (bool) : if True, saves the similarity matrix at the provided output_path.
        output_path (str) : path to destination for saved file.
    '''
    train_questions = train_questions.to_list() if type(train_questions) != list else train_questions
    test_questions = test_questions.to_list() if type(test_questions) != list else test_questions
    train_embeddings = get_sentence_embeddings(train_questions,embedding_model,progress_bar)
    test_embeddings = get_sentence_embeddings(train_questions,embedding_model,progress_bar)
    similarities = pd.DataFrame()
    #Compute cosinus similarity between the embeddings
    for t in tqdm(range(len(test_embeddings))):
        similarities[t] = [round(util.cos_sim(train_embeddings[i],test_embeddings[t]).item(),5) for i in range(len(train_questions))]
    if save:
        np.savetxt(output_path,similarities.values)
    return similarities


class SEER:
    '''
    The SEER algorithm.
    Attributes:
        k (int) : the number of nearest neighbor to filter
        M (int) : the maximum number of exemplars to select
        alpha (float) : the share of exemplars that should possess the attribute of the test instance
        beta (float) : the share of exemplars that should not possess the attribute of the test instance
        modules (list) : list of constraint modules to include
    '''
    def __init__(self,
                 k=200,
                 M=4,
                 alpha=0.5,
                 beta=0.25,
                 modules=['modality','answer_type']):  
        self.k = k
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.modules = modules


    def get_KNN(self,test_idx,similarity_matrix,train_filter=[]):
        '''
        Retrieves the nearest neighbor of a test instance given a similarity matrix.
        '''
        instance_similarity = similarity_matrix[test_idx]
        candidates = [idx for idx in np.argsort(instance_similarity) if idx in train_filter][-self.k:][::-1]  #First is the best one 
        return candidates
    
    def get_knapsack(self,
                     test_idx,
                     candidates_idx,
                     train_dataframe,
                     test_dataframe,
                     similarity_matrix,
                     max_prompt_length,
                     name='seer knapsack problem'):
        problem = LpProblem(name,LpMaximize)
        candidates = LpVariable.dicts("instance",candidates_idx,0,cat=LpBinary)
        #Add objective
        problem += lpSum([similarity_matrix[test_idx][i]*candidates[i] for i in candidates_idx])
        #Add capacity constraints
        problem += lpSum([train_dataframe.loc[i,'token_prompt_length']*candidates[i] for i in candidates_idx]) <= max_prompt_length
        problem += lpSum([candidates[i] for i in candidates_idx]) <= self.M
        #Add diversity constraints
        if 'modality' in self.modules:
            modality= test_dataframe.loc[test_idx,'predicted_modality']
            if modality==0:
                problem += lpSum([train_dataframe.loc[i,'table_only']*candidates[i]for i in candidates_idx]) >= self.M * self.alpha
                problem += lpSum([train_dataframe.loc[i,'use_text']*candidates[i]for i in candidates_idx]) >=self.M * self.beta
            elif modality==1:
                problem += lpSum([train_dataframe.loc[i,'use_table']*candidates[i]for i in candidates_idx]) >= self.M * self.beta
                problem += lpSum([train_dataframe.loc[i,'text_only']*candidates[i]for i in candidates_idx]) >= self.M * self.alpha
            else:
                problem += lpSum([train_dataframe.loc[i,'table_only']*candidates[i]for i in candidates_idx]) >= self.M * self.beta
                problem += lpSum([train_dataframe.loc[i,'text_only']*candidates[i]for i in candidates_idx]) >= self.M * self.beta
                problem += lpSum([train_dataframe.loc[i,'hybrid']*candidates[i]for i in candidates_idx]) >= min(self.M * self.beta,train_dataframe.loc[candidates_idx,'hybrid'].sum()) 
        if 'answer_type' in self.modules:
            answer_type = test_dataframe.loc[test_idx,'predicted_answer_type']
            problem += lpSum([train_dataframe.loc[i,answer_type]*candidates[i]for i in candidates_idx]) >= self.M * self.alpha
            other_answer_type = ['span','multi-span','arithmetic','count']
            other_answer_type.remove(answer_type)
            problem +=  lpSum([train_dataframe.loc[i,other_answer_type[0]]*candidates[i] +
                       train_dataframe.loc[i,other_answer_type[1]]*candidates[i] +
                       train_dataframe.loc[i,other_answer_type[2]]*candidates[i] for i in candidates_idx]) >= self.M * self.beta      
        return problem
    

    def solve_knapsack(self,problem,timelimit=5.0):
        solver = GUROBI_CMD(timeLimit=timelimit) 
        problem.solve(solver)
        try:
            solution = dict(zip([int(v.name.split('_')[1]) for v in problem.variables() ],[v.varValue for v in problem.variables() ]))
        except:
            solution = dict(zip([int(v.name.split('_')[1]) for v in problem.variables()[1:]],[v.varValue for v in problem.variables()[1:]]))
        return solution
    
    def get_few_shot_exemplars(self,test_idx,similarity_matrix,train_dataframe,test_dataframe,train_filter,L):
        candidates = self.get_KNN(test_idx,similarity_matrix,train_filter)
        problem = self.get_knapsack(test_idx,candidates,train_dataframe,test_dataframe,similarity_matrix,L)
        solution = self.solve_knapsack(problem)
        selection = [k for k,v in solution.items() if v==1]
        few_shot_selection = [idx for idx in np.argsort(similarity_matrix[test_idx]) if idx in selection][::-1]
        return few_shot_selection