import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

def heatmap(accuracy, row_labels, col_labels, x_label, y_label, title, vlimit, cbar = True):
    # Define the data
    accuracy = np.array(accuracy)

    # Create DataFrame
    df = pd.DataFrame(accuracy, index=row_labels, columns=col_labels)

    plt.figure(figsize=(8, 6))

    if vlimit:
        vmin, vmax = vlimit
        sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=vmin, vmax=vmax, cbar = cbar)
    else:
        sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar = cbar)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_from_file(PATH, metric, title = None, vlimit = None, cbar = True):
    results = pickle.load(open(PATH, 'rb'))
    results_data = results['data']
    results_metadata = results['metadata']
    q_type = results_metadata['q_type']
    d_type = results_metadata['type']
    order = results_metadata['order']

    if d_type == "grp_count" or d_type == "grp_disc" or d_type == "grp_list" or d_type == "grp_ratio":
        row_labels = [str(x) for x in range(1,6)]
        col_labels = [str(x) for x in range(1,6)]
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
            elif metric == "flip_rate":
                row_acc.append(sum([ele['r'] != ele['r^org'] for ele in eles])/len(eles))
        accuracy.append(row_acc)
    accuracy = np.array(accuracy)

    return heatmap(accuracy, row_labels, col_labels, x_label, y_label, title, vlimit, cbar)

def plot_from_list(list_PATH, metric, list_title, vlimit):
    num_plots = len(list_PATH)
    fig, axs = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8), squeeze=False)

    for fig_idx, path_ele in enumerate(list_PATH):
        cbar = True if fig_idx == num_plots - 1 else False


        # Load data
        if type(path_ele) == str:
            with open(path_ele, 'rb') as f:
                results = pickle.load(f)
            results_data = results['data']
        elif type(path_ele) == list:
            temp_results = []
            for PATH in path_ele:
                with open(PATH, 'rb') as f:
                    results_data = pickle.load(f)['data']
                temp_results.append(results_data)

            results_data = [[[] for _ in range(len(temp_results[0][0]))] for _ in range(len(temp_results[0]))]

            for target_data in temp_results:
                for r_idx, row in enumerate(target_data):
                    for c_idx, col in enumerate(row):
                        for ele in col:
                            results_data[r_idx][c_idx].append(ele)
            
        # Set up labels
        row_labels = [str(x) for x in range(1,6)]
        col_labels = [str(x) for x in range(1,6)]
        x_label = "# of Agree"
        y_label = "# of Disagree"

        # Title
        subplot_title = list_title[fig_idx] if list_title else None

        # Compute accuracy matrix
        accuracy = []
        for row in results_data:
            row_acc = []
            for eles in row:
                if metric == "accuracy":
                    row_acc.append(sum([ele['r'] == ele['answer'] for ele in eles]) / len(eles))
                elif metric == "consistency":
                    row_acc.append(sum([ele['r'] == ele['r^org'] for ele in eles]) / len(eles))
                elif metric == "flip_rate":
                    row_acc.append(sum([ele['r'] != ele['r^org'] for ele in eles]) / len(eles))
            accuracy.append(row_acc)

        # Plot on the specific axis
        df = pd.DataFrame(accuracy, index=row_labels, columns=col_labels)
        ax = axs[0][fig_idx]
        sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
                    vmin=vlimit[0] if vlimit else None, vmax=vlimit[1] if vlimit else None,
                    cbar=False, ax=ax, annot_kws={"size": 20})

        ax.set_title(subplot_title, fontsize=20)
        ax.set_xlabel(x_label, fontsize=18)
        ax.set_ylabel(y_label, fontsize=18)

    plt.tight_layout()
    plt.show()

def plot_from_list_data(list_data, metric, list_title, vlimit):
    num_plots = len(list_data)
    fig, axs = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8), squeeze=False)

    for fig_idx, results_data in enumerate(list_data):

        # Set up labels
        row_labels = [str(x) for x in range(1,6)]
        col_labels = [str(x) for x in range(1,6)]
        x_label = "# of Agree"
        y_label = "# of Disagree"

        # Title
        subplot_title = list_title[fig_idx] if list_title else None

        # Compute accuracy matrix
        accuracy = []
        for row in results_data:
            row_acc = []
            for eles in row:
                if metric == "accuracy":
                    row_acc.append(sum([ele['r'] == ele['answer'] for ele in eles]) / len(eles))
                elif metric == "consistency":
                    row_acc.append(sum([ele['r'] == ele['r^org'] for ele in eles]) / len(eles))
                elif metric == "flip_rate":
                    row_acc.append(sum([ele['r'] != ele['r^org'] for ele in eles]) / len(eles))
            accuracy.append(row_acc)

        # Plot on the specific axis
        df = pd.DataFrame(accuracy, index=row_labels, columns=col_labels)
        ax = axs[0][fig_idx]
        sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
                    vmin=vlimit[0] if vlimit else None, vmax=vlimit[1] if vlimit else None,
                    cbar=False, ax=ax, annot_kws={"size": 20})

        ax.set_title(subplot_title, fontsize=20)
        ax.set_xlabel(x_label, fontsize=18)
        ax.set_ylabel(y_label, fontsize=18)

    plt.tight_layout()
    plt.show()