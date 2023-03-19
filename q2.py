from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']
idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


def testAccuracyPercentage(n, k):
    accuratePreds = 0
    trainingImages = train[:n]
    trainingLabels = train_labels[:n]
    testLabels = test_labels.astype(int)
    for i in range(len(test)):
        prediction = kNearestNeighbors(trainingImages, trainingLabels, test[i], k)
        accuratePreds += (prediction == testLabels[i])
    return accuratePreds / len(test) * 100


def kNearestNeighbors(trainingImages, trainingLabels, queryImage, k):
    nearestLabels = findNeighborLabels(trainingImages, trainingLabels, queryImage, k).astype(int)
    # Find the mode by counting each label's frequency and selecting the argmax
    prediction = np.bincount(nearestLabels).argmax()
    return prediction


def findNeighborLabels(trainingImages, trainingLabels, queryImage, k):
    nearestIndices = findNearestNeighborIndices(trainingImages, queryImage, k)
    return trainingLabels[nearestIndices]


def findNearestNeighborIndices(trainingImages, queryImage, k):
    distances = computeDistances(trainingImages, queryImage)
    return np.argsort(distances)[:k]


def computeDistances(trainingImages, queryImage):
    distances = [computeL2Distance(queryImage, trainingImage) for trainingImage in trainingImages]
    return distances


def computeL2Distance(source, target):
    return np.linalg.norm(source - target, ord=2)


def runFirstTest():
    print(testAccuracyPercentage(1000, 10))


runFirstTest()
