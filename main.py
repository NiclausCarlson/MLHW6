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


class AdaBoost:
    def __init__(self, treeQuantity, objects, classes):
        self.treeQuantity = treeQuantity
        self.classifiers = []
        self.TREE_DEPTH = 4
        self.objects = objects
        self.classes = classes

    class Classifier:
        def __init__(self, classifier, coef):
            self.classifier = classifier
            self.coefs = coef

        def classify(self, obj):
            return self.coefs * self.classifier.predict(obj)

    def setClassifiers(self):
        weights = [1 / len(self.classes) for _ in range(len(self.classes))]
        for t in range(self.treeQuantity):
            curClassifier = DecisionTreeClassifier(max_depth=self.TREE_DEPTH)
            curClassifier.fit(self.objects, self.classes, sample_weight=weights)
            predicted = curClassifier.predict(self.objects)
            error = weightsError(predicted, self.classes, weights)
            coef = 1 / 2 * math.log1p((1 - error) / error)
            self.classifiers.append(self.Classifier(curClassifier, coef))
            for i in range(len(weights)):
                weights[i] = weights[i] * math.exp(-coef * self.classes[i] * predicted[i])
            Z = sum(weights)
            for i in range(len(weights)):
                weights[i] = weights[i] / Z

    def classify(self, obj):
        return numpy.sign(sum([self.classifiers[i].classify(obj) for i in range(len(self.classifiers))]))

    def getAccuracy(self):
        def getClass(cls):
            if cls == 'P':
                return 1
            else:
                return -1

        counter = 0
        for i in range(len(self.classes)):
            predicted = numpy.sign(self.classify(self.objects[i]))
            if predicted == getClass(self.classes[i]):
                counter += 1
        return 100 * counter / len(self.classes)

    def getMinMax(self):
        xMin, xMax, yMin, yMax = self.objects[0][0], self.objects[0][0], self.objects[0][1], self.objects[0][1]
        for obj in self.objects:
            if obj[0] < xMin:
                xMin = obj[0]
            if obj[0] > xMax:
                xMax = obj[0]
            if obj[1] < xMin:
                yMin = obj[1]
            if obj[1] > xMax:
                yMax = obj[1]
        return xMin, xMax, yMin, yMax


def main():
    treeQuantity = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    fileNames = ['chips', 'geyser']
    plotter = Utils.Plotter()
    for name in fileNames:
        fileSystem = Utils.FileSystem(name)
        objects, classes = fileSystem.getData()
        accuracy = []
        index = 1
        for tQ in treeQuantity:
            curClassifier = AdaBoost(tQ, objects, classes)
            plotter.drawAda(name + ' ' + str(index), name + '/', curClassifier)
            accuracy.append(curClassifier.getAccuracy())
            index += 1
        plotter.drawAccuracy(name + ' accuracy', name + '/', treeQuantity, accuracy)


main()
