import csv
import random
import math
import operator
 
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    
    # le o dataset
    with open(filename) as csvfile:

        lines = csv.reader(csvfile)
        dataset = list(lines)
        
        for x in range(len(dataset)-1):
        
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
 
 
# distancia euclidiana entre dois pontos
def euclideanDistance(data1, data2, length):
    
    distance = 0
    for x in range(length):
        distance += pow((data1[x] - data2[x]), 2)
    
    return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
    
    distances = []
    length = len(testInstance)-1
    
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    
    distances.sort(key=operator.itemgetter(1))
    
    neighbors = []
    
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
 
def getResponse(neighbors):
    
    classVotes = {}
    
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    
    return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
    correct = 0
    
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    
    return (correct/float(len(testSet))) * 100.0
    
def main():
    
    trainingSet=[]
    testSet=[]
    #treinar o dataset
    split = 0.80 
    
    loadDataset('iris.csv', split, trainingSet, testSet)

    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))

    predictions=[]
    k = 3 # k is the parameter
    
    for x in range(len(testSet)):
        
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    
    accuracy = getAccuracy(testSet, predictions)
    
    print('Accuracy: ' + repr(accuracy) + '%')
    
main()
