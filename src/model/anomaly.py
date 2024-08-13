from sklearn import svm


class AnomalyEstimation:
    def __init__(self):
        self.model = svm.OneClassSVM()

    def fit(self, normal_data):
        self.model.fit(normal_data)
