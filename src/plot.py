import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

def heatmap(accuracy, row_labels, col_labels, x_label, y_label, title, vlimit):
    # Define the data
    accuracy = np.array(accuracy)

    # Create DataFrame
    df = pd.DataFrame(accuracy, index=row_labels, columns=col_labels)

    plt.figure(figsize=(8, 6))

    if vlimit:
        vmin, vmax = vlimit
        sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=vmin, vmax=vmax)
    else:
        sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_from_file(PATH, metric, title = None, vlimit = None):
    results = pickle.load(open(PATH, 'rb'))
    results_data = results['data']
    results_metadata = results['metadata']
    q_type = results_metadata['q_type']
    d_type = results_metadata['type']
    order = results_metadata['order']

    if d_type == "grp_count" or d_type == "grp_disc" or d_type == "grp_list" or d_type == "grp_ratio":
        row_labels = [str(x) for x in range(11)]
        col_labels = [str(x) for x in range(11)]
        x_label = "# of Agree"
        y_label = "# of Disagree"
    elif d_type == "group_ratio_old":
        row_labels = [4,12,50,100,1000]
        col_labels = ['0%', '25%', '50%', '75%', '100%']
        x_label = "% of Agree"
        y_label = "# of Peer"
        title = "Heatmap of Agreement Ratio"
    else:
        raise ValueError(f"Unknown peer type: {d_type}")

    if not title:
        if order == "ad":
            title = f"metric={metric}, q_type={q_type}, peer_type={d_type}, order=agree first"
        elif order == "da":
            title = f"metric={metric}, q_type={q_type}, peer_type={d_type}, order=disagree first"
        
    accuracy = []
    for row in results_data:
        row_acc = []
        for eles in row:
            if metric == "accuracy":
                row_acc.append(sum([ele['r'] == ele['answer'] for ele in eles])/len(eles))
            elif metric == "consistency":
                row_acc.append(sum([ele['r'] == ele['r^org'] for ele in eles])/len(eles))
        accuracy.append(row_acc)
    accuracy = np.array(accuracy)

    heatmap(accuracy, row_labels, col_labels, x_label, y_label, title, vlimit)