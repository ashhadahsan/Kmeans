from typing import Tuple, List, Dict
from collections import defaultdict, Counter
from random import shuffle
from operator import itemgetter
Instance = Tuple[float, ...]
Cluster = List[Instance]
Centroids = Cluster
def computeCentroids(classification: Dict[int, Cluster]) -> Centroids:
    idx_and_centroids = [(cls_idx, computeCenter(group)) for cls_idx, group in classification.items()]
    idx_and_centroids.sort(key=itemgetter(0))
    return list(centroid for _, centroid in idx_and_centroids)
def computeEuc(p1: Instance, p2: Instance) -> float:
    return sum((x1 - x2) ** 2 for (x1, x2) in zip(p1, p2))
def randomClassify(k: int, Instances: Cluster) -> Dict[int, Cluster]:
    lengthind=range(len(Instances))
    all_idxs = list(lengthind)
    shuffle(all_idxs)
    base_class = 0
    classification: Dict[int, Cluster] = defaultdict(list)
    for idx in all_idxs:
        classification[base_class].append(Instances[idx])
        base_class = (base_class + 1) % k
    return classification
def computeCenter(Instances: Cluster) -> Instance:
    Instances_it = iter(Instances)
    Instance = next(Instances_it)
    if Instance is None:
        raise TypeError("Empty list found")
    totalInstances = list(Instance)
    for Instance in Instances_it:
        for coordinate, value in enumerate(Instance):
            totalInstances[coordinate] += value
    answer= tuple(vals/len(Instances) for vals in totalInstances)
    return answer
def classify(centroids: Centroids, Instances: Cluster) -> Dict[int, Cluster]:
    classification: Dict[int, Cluster] = defaultdict(list)
    for Instance in Instances:
        idx = findNearestCentroidIDX(centroids, Instance=Instance)
        classification[idx].append(Instance)
    return classification
def findNearestCentroidIDX(centroids: Centroids, Instance: Instance) -> int:
    closest_idx_centroid = min(enumerate(centroids), key=lambda idx_centroid: computeEuc(idx_centroid[1], Instance))
    return closest_idx_centroid[0]
def checkforGrouping(k: int, classification1: Dict[int, Cluster], classification2: Dict[int, Cluster]) -> bool:
    for cls_idx in range(k):
        item1 = classification1[cls_idx]
        item2 = classification2[cls_idx]
        if Counter(item1) != Counter(item2):
            return False
    return True
class KMeansModel:
    def __init__(self, centroids: Centroids):
        self.centroids = centroids
    def predict(self, X: Cluster) -> List[int]:
        if self.centroids is None:
            raise ValueError("Fit the data to the model first")
        return [findNearestCentroidIDX(self.centroids, Instance) for Instance in X]
class KMeans:
    def __init__(self, n_cluster: int, max_iter=500):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
    def fit(self, X):
        groups = randomClassify(self.n_cluster, X)
        centroids = computeCentroids(groups)
        groups_new = classify(centroids, X)
        n_tries_left = self.max_iter
        while (not checkforGrouping(self.n_cluster, groups, groups_new)and n_tries_left > 0):
            n_tries_left -= 1
            groups = groups_new
            centroids = computeCentroids(groups)
            groups_new = classify(centroids, X)
        if n_tries_left == 0:
            print("Failed to converge!! WARN")
        return KMeansModel(centroids)
    def fit_predict(self, X):
        kmeans = self.fit(X)
        return (kmeans.centroids, kmeans.predict(X))
