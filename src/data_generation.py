import sys
sys.path.append('../src')

import qa
import pickle

benchmarks = ["ARC", "OpinionQA", "GlobalOpinionQA", "SIQA"]
factual = ["MMLU-Pro", "GPQA-Diamond", "ARC"]
opinion = ["OpinionQA", "GlobalOpinionQA", "SIQA"]

for benchmark in benchmarks:

    with open(f'../data/{benchmark}/sample_results/org.pkl', 'rb') as f:
        res_org = pickle.load(f)

    input_feat_list = []
    for disagree_size in range(1, 11): # row
        input_row = []
        for agree_size in range(1, 11): # column
            
            eval_feat = {
                'type': 'grp_count',
                'agree_size': agree_size,
                'disagree_size': disagree_size,
                'disagree_type': '2nd',
                'q_type': 'factual' if benchmark in factual else 'opinion',
                'order': 'random'
            }
            input_row.append(eval_feat)
        input_feat_list.append(input_row)

    results, accuracy = qa.qa_eval_matrix(res_org, input_feat_list)

    with open(f'../data/{benchmark}/sample_results/grp_count_10.pkl', 'wb') as f:
        pickle.dump(results, f)

