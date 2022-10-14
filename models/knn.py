import numpy as np
from collections import defaultdict

class KNeighborsClassifier(object):
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _distance(self, data1, data2):
        return sum(abs(data1 - data2))

    def _compute_weights(self, distances):
        if self.weights == 'uniform':
            return [(1, y) for d, y in distances]
        elif self.weights == 'distance':
            matches = [(1, y) for d, y in distances if d == 0]
            return matches if matches else [(1/d, y) for d, y in distances]
        raise ValueError("weights not recognized: should be 'uniform' or 'distance'")

    def _predict_one(self, test):
        distances = sorted((self._distance(x, test), y) for x, y in zip(self.X, self.y))
        weights = self._compute_weights(distances[:self.n_neighbors])
        weights_by_class = defaultdict(list)
        for d, c in weights:
            weights_by_class[c].append(d)
        return max((sum(val), key) for key, val in weights_by_class.items())[1]

    def predict(self, X):
        return [self._predict_one(i) for i in X]

X_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y_train = np.array([1,1,1,0,0,0])
neighbor = KNeighborsClassifier(n_neighbors=3)
neighbor.fit(X_train, y_train)
X_test = np.array([[1, 0], [-2, -2]])
print(neighbor.predict(X_test))

X = np.array([[1, 1], [4, 4], [5, 5]])
y = np.array([1,0,0])
# neighbor = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(X, y)
neighbor = KNeighborsClassifier(weights='distance')
print(neighbor._compute_weights([(0, 1),(0, 1),(3, 0),(0, 0)]))

neighbor = KNeighborsClassifier()
print(neighbor._compute_weights(np.array([(1, 0), (2, 0), (3, 1)])))