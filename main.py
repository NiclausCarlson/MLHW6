import math

import numpy
from sklearn.tree import DecisionTreeClassifier

import Utils


def weightsError(predicted, classes, weight):
    error = 0
    index = 0
    for p in predicted:
        if p != classes[index]:
            error += weight[index]
        index += 1
    return error


class AdaBoost:  # конкретный мега классификатор
    def __init__(self, treeQuantity, fileName):
        self.treeQuantity = treeQuantity
        self.fileSystem = Utils.FileSystem(fileName)
        self.classifiers = []
        self.TREE_DEPTH = 4

    class Classifier:
        def __init__(self, classifier, coef):
            self.classifier = classifier
            self.coefs = coef

        def classify(self, obj):
            return self.coefs * self.classifier.predict(obj)

    def setClassifiers(self):
        objects, classes = self.fileSystem.getData()
        weights = [1 / len(classes) for _ in range(len(classes))]
        for t in range(self.treeQuantity):
            curClassifier = DecisionTreeClassifier(max_depth=self.TREE_DEPTH)
            curClassifier.fit(objects, classes, sample_weight=weights)
            predicted = curClassifier.predict(objects)
            error = weightsError(predicted, classes, weights)
            coef = 1 / 2 * math.log1p((1 - error) / error)
            self.classifiers.append(self.Classifier(curClassifier, coef))
            for i in range(len(weights)):
                weights[i] = weights[i] * math.exp(-coef * classes[i] * predicted[i])
            Z = sum(weights)
            for i in range(len(weights)):
                weights[i] = weights[i] / Z

    def classify(self, obj):
        return numpy.sign(sum([self.classifiers[i].classify(obj) for i in range(len(self.classifiers))]))


def getBackgroundPoints():
    pass


def getAccuracy():
    pass


def main():
    treeQuantity = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    fileNames = ['chips', 'geyser']
    plotter = Utils.Plotter()
    for name in fileNames:
        accuracy = []
        for tQ in treeQuantity:
            curClassifier = AdaBoost(tQ, name)
            # сгенерировать background
            # нанести точки
            # accuracy.append...
            # так 9 раз
        # print graph от accuracy и treeQuantity
