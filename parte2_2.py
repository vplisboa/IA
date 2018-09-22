import numpy as np
import pandas as pd

data = pd.read_csv("test.csv",delimiter=";")

sex = data['sx'].values
rank = data['rk'].values
years = data['yr'].values
formacao = data['dg'].values
yearsF = data['yd'].values
salary = data['sl'].values #y

m  = len(sex)
x0 = np.ones(m)
x = np.array([x0,sex,rank,years,formacao,yearsF]).T

#valores iniciais
b = np.array([0,0,0,0,0,0])
y = np.array(salary)
alpha = 0.0001

def custo_function(x,y,b):
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
		custo = custo_function(x,y,b)
		historico_custo[iteracao] = custo

	return b, historico_custo

novoB, custo = gradiente(x,y,b,alpha,1000)

def rmse(y,y_pred):
	rmsee = np.sqrt(sum((y - y_pred) **2) / len(y))
	return rmsee
	
def r2_score(y,y_pred):
	y_medio = np.mean(y)
	ss_tot = sum((y - y_medio) **2)
	ss_res = sum((y - y_pred) **2)
	r2 = 1 - (ss_res / ss_tot)
	
	return r2

y_pred = x.dot(novoB)

print(novoB)
print(rmse(y,y_pred))
print(r2_score(y,y_pred))
