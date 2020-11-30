import os

from numpy.ma import arange
import matplotlib.pyplot as plt
import csv


class FileSystem:
    def __init__(self, fileName):
        self.fileName = fileName

    def getData(self):
        path = 'data/' + self.fileName + '.csv'
        objects, classes = [], []
        reader = csv.reader(open(path, 'r'))
        for row in reader:
            if row[0] == 'x':
                continue
            tmp = [float(row[0]), float(row[1])]
            objects.append(tmp)
            classes.append(row[2])
        return objects, classes


def getBackgroundPoints(minX, maxX, minY, maxY, step=0.05):
    return [x for x in arange(minX - 7 * step, maxX + 7 * step, step)], \
           [y for y in arange(minY - 7 * step, maxY + 7 * step, step)]


def createDir(path):
    try:
        os.mkdir(path)
    except OSError:
        print('Already exist')


class Plotter:

    def __init__(self):
        self.pointSize = 30
        self.backgroundSize = self.pointSize * 4
        self.positiveBackgroundColor = 'green'
        self.negativeBackgroundColor = 'red'
        self.positivePointColor = 'blue'
        self.negativePointColor = 'yellow'
        self.accuracyColor = 'blue'
        self.fontSize = 10

    def drawAda(self, name, path, classifier):
        xMin, xMax, yMin, yMax = classifier.getMinMax()
        x, y = getBackgroundPoints(xMin, xMax, yMin, yMax)
        xPositive, yPositive, xNegative, yNegative = [], [], [], []
        for _x in x:
            for _y in y:
                predicted = classifier.classify([[_x, _y]])
                if predicted == 1:
                    xPositive.append(_x)
                    yPositive.append(_y)
                else:
                    xNegative.append(_x)
                    yNegative.append(_y)

        positivePointsX, positivePointsY, negativePointsX, negativePointsY = [], [], [], []
        for i in range(len(classifier.classes)):
            if classifier.classes[i] == 1:
                positivePointsX.append(classifier.objects[i][0])
                positivePointsY.append(classifier.objects[i][1])
            else:
                negativePointsX.append(classifier.objects[i][0])
                negativePointsY.append(classifier.objects[i][1])

        fig, ax = plt.subplots()
        ax.scatter(xPositive, yPositive, color=self.positiveBackgroundColor, s=self.backgroundSize, label='P-класс')
        ax.scatter(xNegative, yNegative, color=self.negativeBackgroundColor, s=self.backgroundSize, label='N-класс')
        ax.scatter(positivePointsX, positivePointsY, color=self.positivePointColor, s=self.pointSize,
                   label="Точка класса P")
        ax.scatter(negativePointsX, negativePointsY, color=self.negativePointColor, s=self.pointSize,
                   label="Точка класса N")

        ax.set(title=name,
               xlabel='x-class',
               ylabel='y-class')
        ax.legend(fontsize=self.fontSize)
        plt.show()
        createDir(path)
        fig.savefig(path + name)

    def drawAccuracy(self, name, path, steps, accuracyes):
        fig, ax = plt.subplots()
        ax.plot(steps, accuracyes, color=self.accuracyColor)
        ax.set(title=name,
               xlabel='steps',
               ylabel='accuracy')
        plt.show()
        createDir(path)
        fig.savefig(path + name)
