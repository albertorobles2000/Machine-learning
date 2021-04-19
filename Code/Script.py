# -*- coding: utf-8 -*-
"""
TRABAJO 1.
Nombre Estudiante: Alberto Robles Hernandez
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')


"""
Funcion E(u,v)
"""
def E(u,v):
    return ((u**3)*np.e**(v-2)-2*(v**2)*np.e**(-u))**2

"""
Derivada parcial de E(u,v) respecto a u
"""
def dEu(u,v):
    value = 2*((u**3)*np.e**(v-2)-2*(v**2)*np.e**(-u))*(3*(u**2)*np.e**(v-2)+2*v**2*np.e**(-u))
    return value

"""
Derivada parcial de E(u,v) respecto a v
"""
def dEv(u,v):
    value = 2*((u**3)*np.e**(v-2)-2*(v**2)*np.e**(-u))*(u**3*np.e**(v-2)-4*v*np.e**(-u))
    return value

"""
Gradiente de E(u,v)
"""
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])


"""
Algoritmo del Gradiente descendente sobre E(u,v)
--- initial_point: Punto donde vamos a iniciar la busqueda
--- numMaxIter: Numero maximo de iteraciones
--- errorToGet: Error minimo que buscamos
--- learning_rate: Tasa de aprendizaja
"""
def gradient_descent(initial_point, numMaxIter, errorToGet, learning_rate):
    w = initial_point
    errorActual = E(w[0],w[1])
    iterations=0
    while (iterations < numMaxIter and errorToGet <= errorActual):
        #Actualizamos el valor de W
        incrementoW = - learning_rate * gradE(w[0],w[1])
        w = w + incrementoW
        errorActual = E(w[0],w[1])
        iterations+=1
    return w, iterations


eta = 0.1
maxIter = 100000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent(initial_point, maxIter, error2get, eta)

print ("Gradiente descendente sobre E(u,v)")
print ('Punto inicial: [',initial_point[0],',',initial_point[1],']')
print ('Learning rate: ', eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: ('+str(w[0])+','+ str(w[1])+')')
print ('Valor obtenido: ', E(w[0], w[1]))


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print('Ejercicio 2\n')
"""
Funcion F(x,y)
"""
def F(x,y):
    return ((x+2)**2+2*(y-2)**2+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))

"""
Derivada parcial de F(x,y) respecto a x
"""
def dFx(x,y):
    value = 2*(2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)+x+2)
    return value

"""
Derivada parcial de F(x,y) respecto a y
"""
def dFy(x,y):
    value = 4*(np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)+y-2)
    return value

"""
Gradiente de F(x,y)
"""
def gradF(x,y):
    return np.array([dFx(x,y), dFy(x,y)])

"""
Algoritmo del Gradiente descendente sobre F(x,y)
--- w: Punto donde vamos a iniciar la busqueda
--- numMaxIter: Numero maximo de iteraciones
--- errorToGet: Error minimo que buscamos
--- learning_rate: Tasa de aprendizaja
"""
def gradient_descent(w, numMaxIter, errorToGet, learning_rate):

    valoresF = np.ndarray((numMaxIter+1))
    valoresF[0] = F(w[0],w[1])
    iterations=0
    while (iterations < numMaxIter and errorToGet<F(w[0],w[1])):
        incrementoW = - learning_rate * gradF(w[0],w[1])
        w = w + incrementoW
        iterations+=1
        valoresF[iterations] = F(w[0],w[1])   
    return w, iterations, valoresF


print("Dibujando F(x,y)")

x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = F(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
ax.set(title='Función F(x,y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x,y)')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

"""
A continuacion vamos a aplicar el gradiente descendente sobre F(x,y) con 
un learning rate de 0.01
"""
eta = 0.01
maxIter = 50
error2get = -1000
initial_point = np.array([-1.0,1.0])


w, it, valoresF = gradient_descent(initial_point, maxIter, error2get, eta)

print ('Gradiente descendente sobre F(x,y) con learningRate=0.01')
print ('Punto inicial: [',initial_point[0],',',initial_point[1],']')
print ('Learning rate: ', eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: ('+str(w[0])+','+ str(w[1])+')')
print ('Valor obtenido: ', F(w[0], w[1]))

iteraciones = np.arange(it+1)
fig = plt.figure()
plt.plot(iteraciones, valoresF, color="green",linestyle="--",label="learningRate=0.01")

plt.title(label="LearningRate=0.01")
plt.xlabel("Iteraciones")
plt.ylabel("F(x,y)")
plt.grid()
plt.legend(framealpha=0.5)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

"""
A continuacion vamos a aplicar el gradiente descendente sobre F(x,y) con 
un learning rate de 0.1
"""
eta = 0.1
w, it, valoresF = gradient_descent(initial_point, maxIter, error2get, eta)


print ('Gradiente descendente sobre F(x,y) con learningRate=0.1')
print ('Punto inicial: [',initial_point[0],',',initial_point[1],']')
print ('Learning rate: ', eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: ('+str(w[0])+','+ str(w[1])+')')
print ('Valor obtenido: ', F(w[0], w[1]))

fig = plt.figure()
plt.plot(iteraciones, valoresF, color="green",linestyle="--",label="learningRate=0.1")
plt.title(label="LearningRate=0.1")
plt.xlabel("Iteraciones")
plt.ylabel("F(x,y)")
plt.grid()
plt.legend(framealpha=0.5)
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
print("Distintos puntos de inicio learningRate=0.01")


"""
Vamos a calcular el gradiente desendende con 
distintos puntos de inicio y con un learningRate = 0.01
"""

eta = 0.01
maxIter = 50
error2get = -1000
initial_points = np.array([[-0.5,-0.5],[1,1],[2.1,-2.1],[-3,3],[-2,2]])
colors = ['green', 'blue', 'red', 'orange', 'black']


for i in range(5):

    w, it, valoresF = gradient_descent(initial_points[i], maxIter, error2get, eta)
    plt.plot(iteraciones, valoresF, color=colors[i],linestyle="--",label=str(initial_points[i]))
    print ("\n------------------------------------------")
    print (i, '-->Punto Inicial(',str(initial_points[i]),'):')
    print ('Valor minimo:',F(w[0],w[1]))
    print ('Coordenadas obtenidas: ('+str(w[0])+','+ str(w[1])+')')
    

plt.xlabel("Iteraciones")
plt.ylabel("Value of F(x,y)")
plt.title(label="Comparación de puntos de inicio, learningRate=0.01")
plt.grid()
plt.legend(framealpha=0.5)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print("Distintos puntos de inicio learningRate=0.02")


"""
Vamos a calcular el gradiente desendende con 
distintos puntos de inicio y con un learningRate = 0.02
"""

eta = 0.02
maxIter = 50
error2get = -1000
initial_points = np.array([[-0.5,-0.5],[1,1],[2.1,-2.1],[-3,3],[-2,2]])
colors = ['green', 'blue', 'red', 'orange', 'black']



for i in range(5):

    w, it, valoresF = gradient_descent(initial_points[i], maxIter, error2get, eta)
    plt.plot(iteraciones, valoresF, color=colors[i],linestyle="--",
                        label=str(initial_points[i]))
    print ("\n------------------------------------------")
    print (i, '-->Punto Inicial(',str(initial_points[i]),'):')
    print ('Valor minimo:',F(w[0],w[1]))
    print ('Coordenadas obtenidas: ('+str(w[0])+','+ str(w[1])+')')
    

plt.xlabel("Iteraciones")
plt.ylabel("Value of F(x,y)")
plt.title(label="Comparación de puntos de inicio, learningRate=0.02")
plt.grid()
plt.legend(framealpha=0.5)
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

"""
Funcion para calcular el error cuadratico medio
"""
def Err(x,y,w):
    sumatoria = np.sum((np.matmul(x,w)-y)**2,axis=0)
    return sumatoria/y.shape[0]
    
"""
Funcion para calcular el gradiente de la funcion de error en un punto
"""
def gradErr(x,y,w):
    return (2/y.shape[0])*np.matmul(x.T,np.matmul(x,w)-y)

"""
Funcion que nos divide en un array en bachs de un tamanio dado
"""
def getBachs(indices,tamanioBachs=32):
    for i in range(0, indices.shape[0], tamanioBachs):
        yield indices[i:i+tamanioBachs]


"""
Algoritmo del gradiente descendente estocastico
"""
def sgd(x, y, w, numMaxIter, learning_rate, tamanioBachs=32):
    iterations=0
    while(iterations<numMaxIter):
        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)#Barajamos los indices
        mis_baches = list(getBachs(indices,tamanioBachs))#Separamos en bachs
        for mi_bach in mis_baches:
            #Aplicamos a W el descenso de gradiente
            incrementoW = - learning_rate * gradErr(x[mi_bach[:]],y[mi_bach[:]],w)
            w = w + incrementoW
        iterations+=1
    return w


#Creamos el vector de pesos
w = np.array([1,2,3])
# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

numMaxIter = 1000
learning_rate = 0.01
tamanioBachs = 32
w = sgd(x, y, w, numMaxIter, learning_rate, tamanioBachs)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
print ("W: ", w)



print("\nCreando scatter plot de la muestra, con SGD\n")

"""
Creamos ScatterPlot con los valores de la muestra, asi como nuestra funcion h
obtenido, tras ajustar los pesos mediante el SGD a la muestra
"""
plt.scatter(x[y==1,1],x[y==1,2], c='blue', marker='.',label='5')
plt.scatter(x[y==-1,1],x[y==-1,2], c='red', marker='.',label='1')  
xDraw = np.linspace(np.amin(x[:,1]),np.amax(x[:,1]), 100)
yDraw = np.linspace(np.amin(x[:,2]),np.amax(x[:,2]), 100)
X, Y = np.meshgrid(xDraw,yDraw)
F = w[0]+w[1]*X+w[2]*Y
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('SGD')  
plt.title(label='ScatterPlor SGD Muestra')
plt.xlabel('Nivel de Gris')
plt.ylabel('Simetria')
plt.legend()
plt.show()



print("\nCreando scatter plot de la población, con SGD\n")

"""
Creamos ScatterPlot con los valores de la poblacion, asi como nuestra funcion h
obtenido, tras ajustar los pesos mediante el SGD a la muestra
"""
plt.scatter(x_test[y_test==1,1],x_test[y_test==1,2], c='blue', marker='.',label='5')
plt.scatter(x_test[y_test==-1,1],x_test[y_test==-1,2], c='red', marker='.',label='1')    
xDraw = np.linspace(np.amin(x_test[:,1]),np.amax(x_test[:,1]), 100)
yDraw = np.linspace(np.amin(x_test[:,2]),np.amax(x_test[:,2]), 100)
X, Y = np.meshgrid(xDraw,yDraw)
F = w[0]+w[1]*X+w[2]*Y
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('SGD')
plt.title(label='ScatterPlor SGD Poblacion')
plt.xlabel('Nivel de Gris')
plt.ylabel('Simetria')
plt.legend()
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")

# Pseudoinversa
def pseudoinverse(X):
    pseudoinversa = np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T)
    return pseudoinversa

def Regress_Lin(x,y):
    pseudoinversa = pseudoinverse(x)
    w = np.matmul(pseudoinversa,y)
    return w


w = Regress_Lin(x,y)
print ('Bondad del resultado para pseudo inversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
print ("W: ", w)

print("\nCreando scatter plot de la muestra, con PseudoInversa\n")

"""
Creamos ScatterPlot con los valores de la muestra, asi como nuestra funcion h
obtenido, tras ajustar los pesos mediante la pseudoInversa a la muestra
"""
plt.scatter(x[y==1,1],x[y==1,2], c='blue', marker='.',label='5')
plt.scatter(x[y==-1,1],x[y==-1,2], c='red', marker='.',label='1')    
xDraw = np.linspace(np.amin(x[:,1]),np.amax(x[:,1]), 100)
yDraw = np.linspace(np.amin(x[:,2]),np.amax(x[:,2]), 100)
X, Y = np.meshgrid(xDraw,yDraw)
F = w[0]+w[1]*X+w[2]*Y
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('PseudoInversa')  
plt.title(label='ScatterPlor PseudoInversa Muestra')
plt.xlabel('Nivel de Gris')
plt.ylabel('Simetria')
plt.legend()
plt.show()


print("\nCreando scatter plot de la poblacion, con PseudoInversa\n")

"""
Creamos ScatterPlot con los valores de la poblacion, asi como nuestra funcion h
obtenido, tras ajustar los pesos mediante la pseudoInversa a la muestra
"""
plt.scatter(x_test[y_test==1,1],x_test[y_test==1,2], c='blue', marker='.',label='5')
plt.scatter(x_test[y_test==-1,1],x_test[y_test==-1,2], c='red', marker='.',label='1')    
xDraw = np.linspace(np.amin(x_test[:,1]),np.amax(x_test[:,1]), 100)
yDraw = np.linspace(np.amin(x_test[:,2]),np.amax(x_test[:,2]), 100)
X, Y = np.meshgrid(xDraw,yDraw)
F = w[0]+w[1]*X+w[2]*Y
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('PseudoInversa')
plt.title(label='ScatterPlor PseudoInversa Poblacion')
plt.xlabel('Nivel de Gris')
plt.ylabel('Simetria')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1
"""
Función que va a etiquetar nuestros datos
"""
def f(x1, x2):
    return sign((x1-0.2)**2+x2**2-0.6)


print("Creando scatter plot de valores de entrenamientos")


"""
Dibujamos los datos sin etiquetar en un scatter plot
"""
N = 1000
size = 1
d = 2
x_train = simula_unif(N, d, size)

plt.scatter(x_train[:,0],x_train[:,1], c='blue', marker='.')
plt.title(label='ScatterPlor valores entrenamiento')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

"""
Funcion que nos devuelve un array con las etiquetas de las instancias
"""
def getLabels(x):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = f(x[i,0],x[i,1])
    return y

"""
Funcion que aniade un porcentaje de ruido a nuestras etiquetas
Cambia los 1 por -1 y viceversa
"""
def getNoisyLabels(y,porcentaje=10):
    noisy_y = np.copy(y)
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    a_modificar = (porcentaje*y.shape[0])//100
    for i in range(a_modificar):
        noisy_y[indices[i]] = y[indices[i]]*-1
    return noisy_y
        

input("\n--- Pulsar tecla para continuar ---\n")
print("Creando scatter plot de valores de entrenamiento etquetados y con 10% de ruido")


"""
Dibujamos los datos etiquetados y con un 10% de ruido en un scatter plot
"""
y_train = getLabels(x_train)
y_train_noisy = getNoisyLabels(y_train)

plt.scatter(x_train[y_train_noisy==1,0],x_train[y_train_noisy==1,1], c='blue', marker='.', label='1')
plt.scatter(x_train[y_train_noisy==-1,0],x_train[y_train_noisy==-1,1], c='red', marker='.', label='-1')
plt.title(label='ScatterPlor valores etiquetados con 10% de ruido')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left',framealpha=10)
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

#Aniadimos una columa de unos
nuevaColumna = np.ones(x_train.shape[0])
nuevaColumna= nuevaColumna.reshape((-1,1))
x_train = np.append(nuevaColumna, x_train, axis=1)

w = np.array([1,2,3])
# Lectura de los datos de entrenamiento
numMaxIter = 50
learning_rate = 0.01
tamanioBachs = 32

w = sgd(x_train, y_train_noisy, w, numMaxIter, learning_rate, tamanioBachs)


plt.scatter(x_train[y_train_noisy==1,1],x_train[y_train_noisy==1,2], c='blue', marker='.',label='1')
plt.scatter(x_train[y_train_noisy==-1,1],x_train[y_train_noisy==-1,2], c='red', marker='.',label='-1')
x = np.linspace(-1.0, 1.0, 100)
y = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x,y)
F = w[0]+w[1]*X+w[2]*Y
plt.contour(X,Y,F,[0])

plt.title(label='ScatterPlor SGD')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim((-1, 1)) 
plt.ylim((-1, 1))
plt.legend(loc='upper left',framealpha=10)
plt.show()


print ('Bondad del resultado para grad. descendente estocastico:')
print ("Ein: ", Err(x_train,y_train_noisy,w))
print ("W: ", w) 

input("\n--- Pulsar tecla para continuar ---\n")
print("Vamos a iterar 1000 veces")

ErrIn = 0
ErrOut = 0
N = 1000
size = 1
d = 2
numMaxIter = 50
learning_rate = 0.01
tamanioBachs = 32
w = np.array([1,2,3])
for i in range(1000):
    x_train = simula_unif(N, d, size)
    y_train_noisy = getNoisyLabels(getLabels(x_train))
    nuevaColumna = np.ones(x_train.shape[0]).reshape((-1,1))
    x_train = np.append(nuevaColumna, x_train, axis=1)
    
    x_test = simula_unif(N, d, size)
    y_test_noisy = getNoisyLabels(getLabels(x_test))
    x_test = np.append(nuevaColumna, x_test, axis=1)
    
    w = sgd(x_train, y_train_noisy, w, numMaxIter, learning_rate, tamanioBachs)
    
    ErrIn += Err(x_train,y_train_noisy,w)
    ErrOut += Err(x_test,y_test_noisy,w)
    if(i%100==0):
        print(str(i)+'/1000')
    

ErrIn = ErrIn/N
ErrOut = ErrOut/N
print ('Errores medios tras 1000 iteraciones:\n')
print ("Ein medio: ", ErrIn)
print ("Eout medio: ", ErrOut)


input("\n--- Pulsar tecla para continuar ---\n")

"""
Funcion que prepara el vector de caracteristicas aniadiendole
unos, y las combinaciones no lineales
"""
def getVectorDeCaracteristicasNoLineal(x):
    #[1,x1,x2,x1*x2,x1^2,x2^2]
    #Aniadimos unos
    primeraColumna = np.ones(x.shape[0])
    primeraColumna= primeraColumna.reshape((-1,1))
    x_train = np.append(primeraColumna, x, axis=1)
    #Aniadimos x1*x2
    cuartaColumna = x[:,0]*x[:,1]
    cuartaColumna= cuartaColumna.reshape((-1,1))
    x_train = np.append(x_train, cuartaColumna, axis=1)
    #Aniadimos x1^2
    quintaColumna = x[:,0]**2
    quintaColumna= quintaColumna.reshape((-1,1))
    x_train = np.append(x_train, quintaColumna, axis=1)
    #Aniadimos x2^2
    sextaColumna = x[:,1]**2
    sextaColumna= sextaColumna.reshape((-1,1))
    x_train = np.append(x_train, sextaColumna, axis=1)
       
    return x_train

N = 1000
size = 1
d = 2
x_train = simula_unif(N, d, size)
y_train = getLabels(x_train)
y_train_noisy = getNoisyLabels(y_train)
x_train = getVectorDeCaracteristicasNoLineal(x_train)



w = np.array([1,2,3,4,5,6])
numMaxIter = 50
learning_rate = 0.01
tamanioBachs = 32

w = sgd(x_train, y_train_noisy, w, numMaxIter, learning_rate, tamanioBachs)
plt.scatter(x_train[y_train_noisy==1,1],x_train[y_train_noisy==1,2], c='blue', marker='.',label='1')
plt.scatter(x_train[y_train_noisy==-1,1],x_train[y_train_noisy==-1,2], c='red', marker='.',label='-1')
x = np.linspace(-1.0, 1.0, 100)
y = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x,y)
F = w[0]+w[1]*X+w[2]*Y+w[3]*X*Y+w[4]*X**2+w[5]*Y**2
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('SGD')
plt.title(label='ScatterPlor SGD Caract. no lineales')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim((-1, 1)) 
plt.ylim((-1, 1))


plt.legend(loc='upper left',framealpha=10)
plt.show()

print ('Bondad del resultado para grad. descendente estocastico sobre\n'+
       'caracteristicas no lineales:\n')
print ("Ein: ", Err(x_train,y_train_noisy,w))
print ("W: ", w)



input("\n--- Pulsar tecla para continuar ---\n")


ErrIn = 0
ErrOut = 0
N = 1000
size = 1
d = 2
numMaxIter = 50
learning_rate = 0.01
tamanioBachs = 32
w = np.array([1,2,3,4,5,6])
for i in range(1000):
    x_train = simula_unif(N, d, size)
    y_train = getLabels(x_train)
    y_train_noisy = getNoisyLabels(y_train)
    x_train = getVectorDeCaracteristicasNoLineal(x_train)
    
    x_test = simula_unif(N, d, size)
    y_test = getLabels(x_test)
    y_test_noisy = getNoisyLabels(y_test)
    x_test = getVectorDeCaracteristicasNoLineal(x_test)
    
    w = sgd(x_train, y_train_noisy, w, numMaxIter, learning_rate, tamanioBachs)
    
    ErrIn += Err(x_train,y_train_noisy,w)
    ErrOut += Err(x_test,y_test_noisy,w)
    if(i%100==0):
        print(str(i)+'/1000')
    

ErrIn = ErrIn/N
ErrOut = ErrOut/N
print ('Errores medios tras 1000 iteraciones:\n')
print ("Ein medio: ", ErrIn)
print ("Eout medio: ", ErrOut)


input("\n--- Pulsar tecla para continuar ---\n")
print ('METODO DE NEWTON\n')


def F(x,y):
    return ((x+2)**2+2*(y-2)**2+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))
#Segunda derivada parcial primero respecto a x luego denuevo respecto a x
def dFxx(x,y):
    value = 2-8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    return value
#Segunda derivada parcial primero respecto a x luego denuevo respecto a y
def dFxy(x,y):
    value = 8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
    return value
#Segunda derivada parcial primero respecto a y luego denuevo respecto a x
def dFyx(x,y):
    value = 8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
    return value
#Segunda derivada parcial primero respecto a y luego denuevo respecto a y
def dFyy(x,y):
    value = 4-8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    return value

"""
Obtenemos la matriz hessiana en el punto (x,y)
"""
def Hessian(x,y):
    hessiana = np.ndarray(shape=(2,2),dtype="float64")
    hessiana[0,0] = dFxx(x,y)
    hessiana[0,1] = dFxy(x,y)
    hessiana[1,0] = dFyx(x,y)
    hessiana[1,1] = dFyy(x,y)
    return hessiana

def NewtonsMethod(w, numMaxIter, learning_rate):
    iterations = 0
    valoresF = np.ndarray((numMaxIter+1))
    valoresF[0] = F(w[0],w[1])
    while(iterations<numMaxIter):
        a = np.linalg.inv(Hessian(w[0],w[1]))
        b = gradF(w[0],w[1])
        incrementoW = -learning_rate*np.matmul(a,b)
        w = w + incrementoW
        iterations+=1
        valoresF[iterations] = F(w[0],w[1])
    return w, valoresF

eta = 0.01
maxIter = 1500
initial_point = np.array([-1.0,1.0])


w, valoresF = NewtonsMethod(initial_point, maxIter, eta)

print ('Newtons method sobre F(x,y) con learningRate=0.01')
print ('Punto inicial: [',initial_point[0],',',initial_point[1],']')
print ('Learning rate: ', eta)
print ('Coordenadas obtenidas: ('+str(w[0])+','+ str(w[1])+')')
print ('Valor obtenido: ', F(w[0], w[1]))

iteraciones = np.arange(maxIter+1)
plt.plot(iteraciones, valoresF, color='blue',linestyle="--",label="[-1.0,1.0]")
    
plt.xlabel("Iteraciones")
plt.ylabel("Value of F(x,y)")
plt.title(label="Newton's method learningRate=0.01")
plt.grid()
plt.legend(framealpha=0.5)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


eta = 0.1
maxIter = 1500
initial_point = np.array([-1.0,1.0])
w, valoresF = NewtonsMethod(initial_point, maxIter, eta)

print ('Newtons method sobre F(x,y) con learningRate=0.1')
print ('Punto inicial: [',initial_point[0],',',initial_point[1],']')
print ('Learning rate: ', eta)
print ('Coordenadas obtenidas: ('+str(w[0])+','+ str(w[1])+')')
print ('Valor obtenido: ', F(w[0], w[1]))

iteraciones = np.arange(maxIter+1)
plt.plot(iteraciones, valoresF, color='blue',linestyle="--",label="[-1.0,1.0]")
    
plt.xlabel("Iteraciones")
plt.ylabel("Value of F(x,y)")
plt.title(label="Newton's method learningRate=0.1")
plt.grid()
plt.legend(framealpha=0.5)
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
print ("Aplicando el metodo de Newton con distintos puntos de inicio")
"""
Acontinuacion vamos a comparar como afectan los distintos puntos de inicio al
metodo de newton
"""
eta = 0.01
maxIter = 6000
initial_points = np.array([[-0.5,-0.5],[1,1],[2.1,-2.1],[-3,3],[-2,2]])
colors = ['green', 'blue', 'red', 'orange', 'black']

iteraciones = np.arange(maxIter+1)
for i in range(5):

    w,valoresF = NewtonsMethod(initial_points[i], maxIter, eta)
    plt.plot(iteraciones, valoresF, color=colors[i],linestyle="--",
                        label=str(initial_points[i]))
    print ("\n------------------------------------------")
    print (i, '-->Punto Inicial(',str(initial_points[i]),'):')
    print ('Valor minimo:',F(w[0],w[1]))
    print ('Coordenadas obtenidas: ('+str(w[0])+','+ str(w[1])+')')
    

plt.xlabel("Iteraciones")
plt.ylabel("Value of F(x,y)")
plt.title(label="Newton's method Comparación puntos de inicio")
plt.grid()
plt.legend(framealpha=0.5)
#plt.xlim((0, 2000)) 
#plt.ylim((-10, 100))
plt.show()
