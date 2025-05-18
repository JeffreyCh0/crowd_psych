import sys
sys.path.append('../src')

import qa
import pickle

benchmarks = ["MMLU-Pro", "GPQA-Diamond", "ARC", "OpinionQA", "GlobalOpinionQA", "SIQA"]
factual = ["MMLU-Pro", "GPQA-Diamond", "ARC"]
opinion = ["OpinionQA", "GlobalOpinionQA", "SIQA"]

for benchmark in benchmarks:

    with open(f'../data/{benchmark}/sample_results/org_reason.pkl', 'rb') as f:
        org_reason = pickle.load(f)

    input_feat_list = []
    for disagree_size in range(1, 11): # row
        input_row = []
        for agree_size in range(1, 11): # column

            eval_feat = {
                'type': 'grp_disc',
                'agree_size': agree_size,
                'disagree_size': disagree_size,
                'disagree_type': '2nd',
                'q_type': 'factual' if benchmark in factual else 'opinion',
                'order': 'random',
                'use_reason': True
            }
            input_row.append(eval_feat)
        input_feat_list.append(input_row)

    results, accuracy = qa.qa_eval_matrix(org_reason, input_feat_list)

    with open(f'../data/{benchmark}/sample_results/grp_reason_10.pkl', 'wb') as f:
        pickle.dump(results, f)