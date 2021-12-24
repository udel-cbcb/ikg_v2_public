from loguru import logger
import matplotlib.pyplot as plt
from numpy import column_stack
import pandas as pd
from scipy.sparse.construct import random
from sklearn.manifold import TSNE
import plotly.express as px

def plot_pr_curve(pr_curve_data,pr_baseline):
    fig, ax = plt.subplots()
    
    no_skill = pr_baseline
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    fig.legend(loc='upper right',
               prop={'size': 6}
            )

    precision = pr_curve_data[0]
    recall = pr_curve_data[1]

    ax.plot(recall, precision, linestyle='--', label="AU-PR")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    fig.legend(loc='upper right',
               prop={'size': 6}
            )

    return fig

def plot_roc_curve(roc_curve_data,label):
    
    # plot curve
    fig, ax = plt.subplots()

    fps = roc_curve_data[0]
    tpr = roc_curve_data[1]

    ax.plot(fps, tpr, linestyle='--', label=label)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    fig.legend(loc='upper right',
               prop={'size': 6}
            )

    # plot baseline
    ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    fig.legend(loc='upper right',
               prop={'size': 6}
    )

    return fig

def plot_embeddings(embeddings,nodes,labels_df):
    
    # transform
    """tSNE = TSNE(n_components=2,random_state=10)
    data = tSNE.fit_transform(embeddings)"""

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    data = pca.fit_transform(embeddings)

    #print("before merge")
    #print(labels_df.head())

    embeddings_df = pd.DataFrame(data=data,columns=["dim_1","dim_2"])
    embeddings_df["node"] = nodes
    embeddings_df["label"] = labels_df["label"].to_list()

    #print("after merge")
    #print(embeddings_df[["node","label"]])
    
    # plot
    fig = px.scatter(embeddings_df,x="dim_1", y="dim_2",color="label")

    fig.update_layout(
        autosize=False,
        width=500,
        height=400,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
    )
    
    #fig.write_image(path)

    return fig

def plot_and_save_embeddings(embeddings,nodes,labels_df,path):
    
    # transform
    tSNE = TSNE(n_components=2,random_state=10)
    data = tSNE.fit_transform(embeddings)

    #from sklearn.decomposition import PCA
    #pca = PCA(n_components=2)
    #data = pca.fit_transform(embeddings)
    
    embeddings_df = pd.DataFrame(data=data,columns=["dim_1","dim_2"])
    embeddings_df["node"] = nodes
    embeddings_df["label"] = labels_df["label"]
    
    # plot
    fig = px.scatter(embeddings_df,x="dim_1", y="dim_2",color="label")

    fig.update_layout(
        autosize=False,
        width=500,
        height=400,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
    )
    
    fig.write_image(path)

    return fig

import io 
from PIL import Image
import numpy as np

def fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


