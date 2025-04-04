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

    vmin, vmax = vlimit

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=vmin, vmax=vmax)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_from_file(PATH, q_type):
    results = pickle.load(open(PATH, 'rb'))
    accuracy = []
    for row in results:
        row_acc = []
        for eles in row:
            if q_type == "factual":
                row_acc.append(sum([ele['r'] == ele['answer'] for ele in eles])/len(eles))
            elif q_type == "opinion":
                row_acc.append(sum([ele['r'] == ele['r^org'] for ele in eles])/len(eles))
        accuracy.append(row_acc)
    accuracy = np.array(accuracy)

    heatmap(accuracy)