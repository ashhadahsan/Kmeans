from kmeans import KMeans
from random import random
import pandas as pd

N_CLUSTERS = 3
N_POINTS = 1000
N_DIMS = 5

def generateRandomPoints(n_dims=5):
    return tuple(random() for _ in range(n_dims))
def main():
    points = list(generateRandomPoints(N_DIMS) for i in range(N_POINTS))
    kmeans = KMeans(N_CLUSTERS).fit(points)
    predictions = kmeans.predict(points)
    df = pd.DataFrame(points)
    df["preds"] = predictions
    assert all(isinstance(pred, int) and 0 <= pred < N_CLUSTERS for pred in predictions)
    print("Assert")
if __name__ == "__main__":
    main()
