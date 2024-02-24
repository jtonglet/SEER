#First script preprocessing
from utils import load_file, retrieve_top_k_text_facts_finqa, retrieve_top_k_text_facts_tatqa
from generate_dataframe import create_question_dataframe_finqa, create_question_dataframe_tatqa
from seer import compute_similarity_matrix

if __name__=='__main__':
    #Load datasets
    #FinQA
    finqa_train = load_file('datasets/finqa/train.json')
    finqa_dev = load_file('datasets/finqa/dev.json')
    finqa_test = load_file('datasets/finqa/test.json')
    #TAT-QA
    tatqa_train = load_file('datasets/tatqa/train.json')
    tatqa_test = load_file('datasets/tatqa/dev.json')
    #New dev split from TAT-QA train
    ctx_idx_dev = [1, 4, 6, 13, 14, 23, 30, 39, 43, 51, 54, 61, 64, 65, 88, 93, 96, 102, 103, 110, 114, 117, 118, 119, 120,
                124, 130, 131, 135, 138, 141, 142, 145, 146, 154, 161, 163, 175, 178, 186, 189, 191, 193, 198, 200, 201,
                206, 209, 217, 223, 224, 228, 229, 234, 247, 255, 257, 262, 270, 283, 285, 287, 292, 313, 317, 318, 322, 
                323, 326, 327, 330, 333, 334, 337, 338, 340, 350, 365, 375, 388, 389, 392, 393, 407, 411, 429, 432, 433,
                435, 437, 438, 440, 445, 447, 449, 451, 457, 460, 466, 468, 469, 471, 476, 484, 487, 490, 493, 497, 501, 
                505, 507, 509, 511, 514, 538, 539, 541, 542, 543, 546, 548, 552, 563, 569, 570, 584, 592, 600, 601, 607, 
                611, 629, 638, 642, 644, 646, 663, 664, 676, 689, 692, 694, 696, 704, 725, 727, 735, 740, 741, 743, 747, 
                758, 764, 765, 775, 776, 777, 778, 781, 788, 799, 810, 817, 821, 824, 832, 833, 841, 859, 864, 865, 866,
                867, 877, 882, 890, 897, 907, 918, 919, 924, 928, 929, 931, 939, 940, 946, 947, 956, 958, 968, 973, 976, 
                985, 994, 995, 996, 1000, 1010, 1022, 1025, 1029, 1034, 1039, 1043, 1052, 1059, 1080, 1083, 1086, 1087, 
                1090, 1093, 1098, 1099, 1103, 1104, 1107, 1116, 1125, 1130, 1133, 1134, 1140, 1149, 1150, 1154, 1158, 1159, 
                1161, 1167, 1168, 1182, 1186, 1188, 1195, 1197, 1206, 1209, 1213, 1220, 1221, 1232, 1236, 1244, 1245, 1247,
                1256, 1265, 1266, 1272, 1276, 1282, 1283, 1287, 1291, 1293, 1309, 1316, 1319, 1326, 1327, 1330, 1333, 1334, 
                1338, 1341, 1345, 1346, 1350, 1352, 1354, 1355, 1358, 1359, 1360, 1362, 1365]
    #1. Create dataframes
    #FinQA
    finqa_train_df = create_question_dataframe_finqa(finqa_train,preprocess=True,ner_mask=True)
    finqa_dev_df = create_question_dataframe_finqa(finqa_dev,preprocess=True,ner_mask=True)
    finqa_test_df = create_question_dataframe_finqa(finqa_test,preprocess=True,ner_mask=True)
    finqa_train_df.to_csv('data_cache/finqa/metadata/finqa_train_df.csv',index=False)
    finqa_dev_df.to_csv('data_cache/finqa/metadata/finqa_dev_df.csv',index=False)
    finqa_test_df.to_csv('data_cache/finqa/metadata/finqa_test_df.csv',index=False)
    #TAT-QA
    tatqa_train_df = create_question_dataframe_tatqa(tatqa_train,preprocess=True,ner_mask=True)
    tatqa_train_df['dev_split'] = tatqa_train_df['context_index'].apply(lambda row : True if row in ctx_idx_dev else False)
    tatqa_dev_df = tatqa_train_df[tatqa_train_df.dev_split==True].reset_index(drop=True)
    tatqa_train_df = tatqa_train_df[tatqa_train_df.dev_split==False].reset_index(drop=True)
    tatqa_test_df = create_question_dataframe_tatqa(tatqa_test,preprocess=True,ner_mask=True)
    tatqa_train_df.to_csv('data_cache/tatqa/metadata/tatqa_train_df.csv',index=False)
    tatqa_dev_df.to_csv('data_cache/tatqa/metadata/tatqa_dev_df.csv',index=False)
    tatqa_test_df.to_csv('data_cache/tatqa/metadata/tatqa_test_df.csv',index=False)

    #2. Apply text retriever
    #FinQA
    retrieved_text_finqa_dev = retrieve_top_k_text_facts_finqa(finqa_test,k=10)
    retrieved_text_finqa_test = retrieve_top_k_text_facts_finqa(finqa_test,k=10)
    retrieved_text_finqa_dev.to_csv('data_cache/finqa/text_retriever/retrieved_text_finqa_dev.csv',index=False)
    retrieved_text_finqa_test.to_csv('data_cache/finqa/text_retriever/retrieved_text_finqa_test.csv',index=False)
    #TAT-QA
    retrieved_text_tatqa_dev = retrieve_top_k_text_facts_tatqa(tatqa_train,tatqa_dev_df,k=10) 
    retrieved_text_tatqa_test = retrieve_top_k_text_facts_tatqa(tatqa_test,tatqa_test_df,k=10) 
    retrieved_text_tatqa_dev.to_csv('data_cache/tatqa/text_retriever/retrieved_text_tatqa_dev.csv',index=False)
    retrieved_text_tatqa_test.to_csv('data_cache/tatqa/text_retriever/retrieved_text_tatqa_test.csv',index=False)

    #3. Compute similarity embeddings
    if not 'similarity_matrices' in 'data_cache/finqa/':
        os.mkdir('data_cache/finqa/similarity_matrices')
    if not 'similarity_matrices' in 'data_cache/tatqa/':
        os.mkdir('data_cache/tatqa/similarity_matrices')  
    #FinQA
    finqa_dev_sim = compute_similarity_matrix(finqa_train_df['question'],finqa_dev_df['question'],
                                          'all-mpnet-base-v2',True,True,
                                          'data_cache/finqa/similarity_matrices/finqa_dev_sim.txt')
    finqa_test_sim = compute_similarity_matrix(finqa_train_df['question'],finqa_test_df['question'],
                                          'all-mpnet-base-v2',True,True,
                                          'data_cache/finqa/similarity_matrices/finqa_test_sim.txt')
    #TAT-QA
    tatqa_dev_sim = compute_similarity_matrix(tatqa_train_df['question'],tatqa_dev_df['question'],
                                          'all-mpnet-base-v2',True,True,
                                          'data_cache/tatqa/similarity_matrices/tatqa_dev_sim.txt')
    tatqa_test_sim = compute_similarity_matrix(tatqa_train_df['question'],tatqa_test_df['question'],
                                          'all-mpnet-base-v2',True,True,
                                          'data_cache/tatqa/similarity_matrices/tatqa_test_sim.txt')
