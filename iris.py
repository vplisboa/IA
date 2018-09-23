import csv
import random
import math
import operator
 
def carregaDataset(nomeArquivo, divisor, conjuntoTreinamento=[] , conjuntoTeste=[]):
    
    # le o dataset
    with open(nomeArquivo) as arquivocsv:

        linhas = csv.reader(arquivocsv)
        dataset = list(linhas)
        
        for x in range(len(dataset)):
        
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            
            if random.random() < divisor:
                conjuntoTreinamento.append(dataset[x])
            else:
                conjuntoTeste.append(dataset[x])
 
 
# distancia euclidiana entre dois pontos
def distanciaEuclidiana(entrada1, entrada2, tamanho):
    
    distancia = 0
    for x in range(tamanho):
        distancia += pow((entrada1[x] - entrada2[x]), 2)
    
    return math.sqrt(distancia)
 
def recuperarVizinhos(conjuntoTreinamento, instanciaTeste, k):
    
    distancias = []
    tamanho = len(instanciaTeste)-1
    
    for x in range(len(conjuntoTreinamento)):
        dist = distanciaEuclidiana(instanciaTeste, conjuntoTreinamento[x], tamanho)
        distancias.append((conjuntoTreinamento[x], dist))
    
    distancias.sort(key=operator.itemgetter(1))
    
    vizinhos = []
    
    for x in range(k):
        vizinhos.append(distancias[x][0])
    return vizinhos
 
def obterResposta(vizinhos):
    
    votosNaClasse = {}
    
    for x in range(len(vizinhos)):
        resposta = vizinhos[x][-1]
        if resposta in votosNaClasse:
            votosNaClasse[resposta] += 1
        else:
            votosNaClasse[resposta] = 1
    
    votosOrdenados = sorted(votosNaClasse.iteritems(), key=operator.itemgetter(1), reverse=True)
    
    return votosOrdenados[0][0]
 
def obterPrecisao(conjuntoTeste, predicoes):
    previsoesCorretas = 0
    
    for x in range(len(conjuntoTeste)):
        if conjuntoTeste[x][-1] == predicoes[x]:
            previsoesCorretas += 1
    
    return (previsoesCorretas/float(len(conjuntoTeste))) * 100.0
    
def main():
    
    conjuntoTreinamento=[]
    conjuntoTeste=[]
    #treinar o dataset
    divisor = 0.80 
    
    carregaDataset('iris.csv', divisor, conjuntoTreinamento, conjuntoTeste)

    print ('Conjunto de treinamento: ' + repr(len(conjuntoTreinamento)))
    print ('Conjunto de teste: ' + repr(len(conjuntoTeste)))

    predicoes=[]
    k = 3 # k eh o parametero
    
    for x in range(len(conjuntoTeste)):
        
        vizinhos = recuperarVizinhos(conjuntoTreinamento, conjuntoTeste[x], k)
        resultado = obterResposta(vizinhos)
        predicoes.append(resultado)
        
        print('> Previsao=' + repr(resultado) + ', Esperado=' + repr(conjuntoTeste[x][-1]))
    
    precisao = obterPrecisao(conjuntoTeste, predicoes)
    
    print('Precisao: ' + repr(precisao) + '%')
    
main()
