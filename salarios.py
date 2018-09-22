import numpy as np
import pandas as pd
from random import seed,randrange,shuffle

data = pd.read_csv("salarios.csv",delimiter=";")

# funcao sample embaralha as linhas do csv
data = data.sample(frac=1)

sex = data['sx'].values
rank = data['rk'].values
years = data['yr'].values
formacao = data['dg'].values
yearsF = data['yd'].values
salary = data['sl'].values #y

m  = len(sex[:43])
x0 = np.ones(m)
x = np.array([x0,sex[:43],rank[:43],years[:43],formacao[:43],yearsF[:43]]).T

#valores iniciais
b = np.array([0,0,0,0,0,0])
y = np.array(salary[:43])
alpha = 0.0001

def calcula_custo(x,y,b):
	m = len(y)
	j = np.sum((x.dot(b) - y) ** 2) / (2*m)
	return j

def gradiente(x,y,b,alpha,iteracoes):
	historico_custo = [0]*iteracoes
	m = len(y)
	custo = 0
	for iteracao in range (iteracoes):
		#valor hipotetico
		h = x.dot(b)
		#diferenca entre hipotese e valor real
		perda = h - y
		#calculo do gradiente
		gradiente = x.T.dot(perda) / m
		#altera valor de b usando gradiente
		b = b - alpha * gradiente
		#novo custo
		custo = calcula_custo(x,y,b)
		historico_custo[iteracao] = custo

	return b, historico_custo

novoB, custo = gradiente(x,y,b,alpha,100000)

def rmse(y,y_pred):
	rmsee = np.sqrt(sum((y - y_pred) **2) / len(y))
	return rmsee
	
def r2_score(y,y_pred):
	y_medio = np.mean(y)
	ss_tot = sum((y - y_medio) ** 2)
	ss_res = sum((y - y_pred) ** 2)
	r2 = 1 - (ss_res / ss_tot)
	
	return r2

y_pred = x.dot(novoB)

def prever(novoB,sex,years,rank,formacao,yearsF,salary):
	m = len(sex)
	acertos = 0
	for i in range(0,m-1):
		equacao = novoB[0] + novoB[1]*sex[i] + novoB[2]*years[i] + novoB[3]*rank[i] + novoB[4] * formacao[i] + novoB[5] * yearsF[i]
		print("encontrado: " + str(equacao) + " \nesperado: " + str(salary[i]))
		if equacao == salary[i]:
			acertos+= 1
			
	return acertos
#print(data.head())
#print(novoB)
#print(rmse(y,y_pred))
#print("avaliacao utilizando r^2:  " + str(r2_score(y,y_pred)))
print("Avalicao utilizando funcao para prever resultados")
print("A funcao substitui as variaveis na equacao hipotese e compara com o valor real")
print("Total de acertos: "+str(prever(novoB,sex[43:],years[43:],rank[43:],formacao[43:],yearsF[43:],salary[43:])))
