import seaborn as sns
from kmeans import KMeans
import pandas as pd
from random import random

N_CLUSTER = 3
N_POINTS = 1000
N_DIMENS = 2

def generateRandomPoints(n_dimens=5):
    return tuple(random() for _ in range(n_dimens))

if __name__ == "__main__":
    points = [generateRandomPoints(N_DIMENS) for i in range(N_POINTS)]
    kmeans = KMeans(N_CLUSTER).fit(points)
    predictions = kmeans.predict(points)
    df = pd.DataFrame(points)
    df["preds"] = predictions
    df=df.rename(columns={0: "x", 1: "y"})
    centroids = kmeans.centroids
    sns.set_palette("pastel")
    sns_plot= sns.scatterplot(x="x",y="y",data=df,hue="preds")
    fig = sns_plot.get_figure()
    fig.savefig("output.png",dpi=250)

