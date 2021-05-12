# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Alberto Robles Hernández
"""
#importamos el modulo numpy
import numpy as np
#importamos el modulo matplotlib
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)

#Simula una distribución uniforme de "N" puntos en "dim" dimensiones
#sobre los intervalor "rango"
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

#Simula una distribución gaussiana de "N" puntos en "dim" dimensiones
#con sigma[eje]
def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

#Simula una recta de forma aleatoria de la forma y=ax+b
def simula_recta(intervalo):
    #Genera dos puntos de forma aleatoria y calcula la recta entre ambos
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

#####################################################################################
# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
#####################################################################################

#Creamos una nube de 50 puntos con una distribución uniforme
#en dos dimensiones sobre el rango [-50,50] 

x = simula_unif(50, 2, [-50,50])
#Pintamos los puntos generados mediante un scatter plot
plt.scatter(x[:,0],x[:,1], c='blue', marker='.')
#Asignamos un titulo al grafico
plt.title(label='ScatterPlor distribucion Uniforme')
#Asignamos un nombre al eje x
plt.xlabel('x')
#Asignamos un nombre al eje y
plt.ylabel('y')
plt.show()

print("Generamos 50 puntos con distribucion uniforme")
input("\n--- Pulsar tecla para continuar ---\n")

######################################################

#Creamos una nube de 50 puntos con una distribución gaussiana
#en dos dimensiones asignando a x sigma=5 a y sigma=7 

x = simula_gaus(50, 2, np.array([5,7]))
#Pintamos los puntos generados mediante un scatter plot
plt.scatter(x[:,0],x[:,1], c='blue', marker='.')
#Asignamos un titulo al grafico
plt.title(label='ScatterPlor distribucion Gaussiana')
#Asignamos un nombre al eje x
plt.xlabel('x')
#Asignamos un nombre al eje y
plt.ylabel('y')
plt.show()

print("Generamos 50 puntos con distribucion Gaussiana")
input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
    if x >= 0: #Si x es mayor o igual que 0 devolvemos 1, si no -1
        return 1
    return -1

#funcion que devuelve la etiqueta de un punto (x,y) dada una funcion y=xa+b
def f(x, y, a, b):
	return signo(y - a*x - b)


#####################################################################################
# EJERCICIO 1.2.a: Dibujar una gráfica con una nube de puntos y una recta aleatoria
# que crea el etiquetado
#####################################################################################

#Generamos otra nube de puntos con una distribución uniforme
x = simula_unif(100, 2, [-50,50])
#Pintamos los puntos generados mediante un scatter plot
plt.scatter(x[:,0],x[:,1], c='blue', marker='.')
#Asignamos un titulo al grafico
plt.title(label='ScatterPlor de la muestra sin etiquetar')
#Asignamos un nombre al eje x
plt.xlabel('x')
#Asignamos un nombre al eje y
plt.ylabel('y')
plt.show()
print("Generamos una muestra de 100 puntos con una distribucion uniforme "+
      "sin etiquetar")
input("\n--- Pulsar tecla para continuar ---\n")



#Generamos una recta la cual va a separar las dos clases
a,b = simula_recta([-50,50])
#Acontinuacion vamos a generar las etiquetas del dataset anterior
#Creamos un vector de etiquetas
y = np.ndarray(shape=(x.shape[0]),dtype=np.float64)
#Comprobamos la posicion de cada uno de los puntos de x respecto a la recta
#y=xa+b y le asignamos un 1 o un -1
for i,instancia in enumerate(x):
    y[i] = f(instancia[0],instancia[1],a,b)
   
#Pintamos la funcion y=xa+b
#Generamos un vector de elementos en el eje x
funcionFx = np.linspace(-50,50,50)
#Para cada elemento del eje x calculamos la y
funcionFy = a*funcionFx+b
#Pintamos la funcion y=xa+b
plt.plot(funcionFx,funcionFy, c='orange', label=str(round(a,2))+'x+'+str(round(b,2)))    
#Pintamos los puntos etiquetados como 1 de color azul
plt.scatter(x[y==1,0],x[y==1,1], c='blue', marker='.',label='1')
#Pintamos los puntos etiquetados como -1 de color rojo
plt.scatter(x[y==-1,0],x[y==-1,1], c='red', marker='.',label='-1')
#Asignamos un titulo al grafico
plt.title(label='ScatterPlor etiquetado y separado por f')
#Asignamos un nombre al eje x
plt.xlabel('x')
#Asignamos un nombre al eje y
plt.ylabel('y')
plt.legend()
plt.show()

print("Generamos una recta aleatoria y etiquetamos los puntos \n"+
      "en funcion de su posicion respecto a la recta")

input("\n--- Pulsar tecla para continuar ---\n")

#####################################################################################
# # 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido
#####################################################################################

#Funcion que aniade un porcentaje de ruido a ambas clases 
def aniadeRuideACadaClase(y,porcentaje=10):
        #Copiamos el vector en otro para no modificar el vector original
        yRuido = np.copy(y)
        #Buscamos los indices de las etiquetas que son iguales a 1 y los 
        #guardamos en el verctor IndicesDeUno
        IndicesDeUno=np.where(y == 1)
        #Barajamos el vector de indices
        np.random.shuffle(IndicesDeUno)
        #Calculamos el numero de etiquetas que debemos cambiar a -1
        elementosACambiar = int((porcentaje*IndicesDeUno[0].shape[0])/100)
        print("Numero de unos: "+str(IndicesDeUno[0].shape[0]))
        print("Numero de cambios: "+str(elementosACambiar))
        #iteramos sobre elementosACambiar elementos y cambiamos su etiqueta de
        #1 a -1
        for i in range(elementosACambiar):
            yRuido[IndicesDeUno[0][i]]=-1
        
        #Buscamos los indices de las etiquetas que son iguales a -1 y los 
        #guardamos en el verctor IndicesDeUno
        #IMPORTANTE BUSCARLOS EN Y, NO EN YRUIDO, YA QUE ESTOS YA ESTAN 
        #MODIFICADOS
        IndicesDeMenosUno=np.where(y == -1)
        #Barajamos el vector de indices
        np.random.shuffle(IndicesDeMenosUno)
        #Calculamos el numero de etiquetas que debemos cambiar a 1
        elementosACambiar = int((porcentaje*IndicesDeMenosUno[0].shape[0])/100)
        print("Numero de menos unos: "+str(IndicesDeMenosUno[0].shape[0]))
        print("Numero de cambios: "+str(elementosACambiar))
        #iteramos sobre elementosACambiar elementos y cambiamos su etiqueta de
        #-1 a 1
        for i in range(elementosACambiar):
            yRuido[IndicesDeMenosUno[0][i]]=1
        
        #Devolvemos el vector de etiquetas con ruido
        return yRuido
    
print("Aniadimos ruido a la muestra\n")
#Aniadimos un 10% de ruido a cada una de las clases de y 
yRuido = aniadeRuideACadaClase(y,10) 
#Pintamos la funcion y=ax+b   
plt.plot(funcionFx,funcionFy, c='orange', label=str(round(a,2))+'x+'+str(round(b,2)))   
#Pintamos las etiquetas con ruido que tienen valor=1 de color azul 
plt.scatter(x[yRuido==1,0],x[yRuido==1,1], c='blue', marker='.',label='1')
#Pintamos las etiquetas con ruido que tienen valor=-1 de color rojo 
plt.scatter(x[yRuido==-1,0],x[yRuido==-1,1], c='red', marker='.',label='-1')
plt.title(label='ScatterPlor etiquetado, separado por f y con 10% de ruido')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#Funcion que genera una matriz de confusion para una muestra dada por x e y
#y una recta de la forma f(x)=xa+b
def confusionMatrix(a,b,x,y):
    #Unos clasificados como unos
    positivasClasificadaBien=0
    #Unos clasificados como menos unos
    positivasClasificadaMal=0
    #Menos unos clasificados como menos unos
    negativasClasificadaBien=0
    #Menos unos clasificados como unos
    negativasClasificadaMal=0
    for caracteristicas, etiqueta in  zip(x,y):
        #Calculamos como clasifica la funcion al punto (caracteristicas[0], caracteristicas[1])
        #y acumulamos su valor donde corresponda
        obtenida = f(caracteristicas[0], caracteristicas[1], a, b)
        if(etiqueta==1 and obtenida==etiqueta):#Unos clasificados como unos
            positivasClasificadaBien+=1
        elif(etiqueta==-1 and obtenida==etiqueta):#Menos unos clasificados como menos unos
            negativasClasificadaBien+=1
        elif(etiqueta==1 and obtenida!=etiqueta):#Unos clasificados como menos unos
            positivasClasificadaMal+=1
        elif(etiqueta==-1 and obtenida!=etiqueta):#Menos unos clasificados como unos
            negativasClasificadaMal+=1
    
    #Devolvemos la matriz construida
    return np.array([[positivasClasificadaBien,negativasClasificadaMal],
                     [positivasClasificadaMal,negativasClasificadaBien]])
     
#Pintamos la matriz de confusion   
print("\nConfusion matriz y=ax+b")
print(confusionMatrix(a, b, x, yRuido))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

#####################################################################################
# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la 
#frontera de clasificación de los puntos de la muestra en lugar de una recta
#####################################################################################

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()

#Primera funcion
def f_1(x):
    return ((x[:,0]-10)**2+(x[:,1]-20)**2-400)
#Segunda funcion
def f_2(x):
    return (0.5*(x[:,0]+10)**2+(x[:,1]-20)**2-400)
#Tercera funcion
def f_3(x):
    return (0.5*(x[:,0]-10)**2-(x[:,1]+20)**2-400)
#Cuarta funcion
def f_4(x):
    return (x[:,1]-20*x[:,0]**2-5*x[:,0]+3)


#Funcion que genera una matriz de confusion para una muestra dada por x e y
#y una funcion cualquiera de la forma f(x,y)
def confusionMatrix(fun,x,y):
    #Unos clasificados como unos
    positivasClasificadaBien=0
    #Unos clasificados como menos unos
    positivasClasificadaMal=0
    #Menos unos clasificados como menos unos
    negativasClasificadaBien=0
    #Menos unos clasificados como unos
    negativasClasificadaMal=0
     #Calculamos como clasifica la funcion a cada punto
     #y acumulamos su valor donde corresponda
    for caracteristicas, etiqueta in  zip(x,y):
        obtenida = signo(fun(caracteristicas.reshape([-1,2])))
        if(etiqueta==1 and obtenida==etiqueta):#Unos clasificados como unos
            positivasClasificadaBien+=1
        elif(etiqueta==-1 and obtenida==etiqueta):#Menos unos clasificados como menos unos
            negativasClasificadaBien+=1
        elif(etiqueta==1 and obtenida!=etiqueta):#Unos clasificados como menos unos
            positivasClasificadaMal+=1
        elif(etiqueta==-1 and obtenida!=etiqueta):#Menos unos clasificados como unos
            negativasClasificadaMal+=1
    #Devolvemos la matriz construida
    return np.array([[positivasClasificadaBien,negativasClasificadaMal],
                     [positivasClasificadaMal,negativasClasificadaBien]])

#Funcion que dada una funcion de la forma f(x,y) devuelve un vector de etiquetas
#asociadas a la x
def getEtiquetas(f,x):
    y = np.ndarray(shape=(x.shape[0]),dtype=np.float64)
    for i in range(x.shape[0]):
        y[i]=signo(f(x[i].reshape([-1,2])))
    return y

#####################################################################################################
#Pintamos la primera funcion
plot_datos_cuad(x, yRuido, f_1, title='Etiquetas Lineales f1(x,y)=(x-10)^2+(y-20)^2-400', xaxis='x axis', yaxis='y axis')
print("Funcion 1 matriz de confusion para etiquetas anteriores:")
#Imprimimos la matriz de confusion para la muestra con ruido anterior y la primera funcion
print(confusionMatrix(f_1,x,yRuido))
y_f1=getEtiquetas(f_1,x)
print("\nAniadimos un 10% de ruido")
y_f1_ruido=aniadeRuideACadaClase(y_f1,10)

plot_datos_cuad(x, y_f1_ruido, f_1, title='Etiquetas Propias f1(x,y)=(x-10)^2+(y-20)^2-400', xaxis='x axis', yaxis='y axis')
print("\nFuncion 1 matriz de confusion para etiquetas propias:")
#Imprimimos la matriz de confusion para la muestra con ruido anterior y la primera funcion
print(confusionMatrix(f_1,x,y_f1_ruido))

input("\n--- Pulsar tecla para continuar ---\n")

######################################################################################################
#Pintamos la segunda funcion
plot_datos_cuad(x, yRuido, f_2, title='Etiquetas Lineales f2(x,y)=0.5(x+10)^2+(y-20)^2-400', xaxis='x axis', yaxis='y axis')
print("\nFuncion 2 matriz de confusion para etiquetas anteriores:")
#Imprimimos la matriz de confusion para la muestra con ruido anterior y la segunda funcion
print(confusionMatrix(f_2,x,yRuido))

y_f2=getEtiquetas(f_2,x)
print("\nAniadimos un 10% de ruido")
y_f2_ruido=aniadeRuideACadaClase(y_f2,10)

plot_datos_cuad(x, y_f2_ruido, f_2, title='Etiquetas Propias f2(x,y)=0.5(x+10)^2+(y-20)^2-400', xaxis='x axis', yaxis='y axis')
print("\nFuncion 2 matriz de confusion para etiquetas propias:")
#Imprimimos la matriz de confusion para la muestra con ruido anterior y la primera funcion
print(confusionMatrix(f_2,x,y_f2_ruido))

input("\n--- Pulsar tecla para continuar ---\n")


######################################################################################################
#Pintamos la tercera funcion
plot_datos_cuad(x, yRuido, f_3, title='Etiquetas Lineales f3(x,y)=0.5(x-10)^2-(y+20)^2-400', xaxis='x axis', yaxis='y axis')
print("\nFuncion 3 matriz de confusion para etiquetas anteriores:")
#Imprimimos la matriz de confusion para la muestra con ruido anterior y la tercera funcion
print(confusionMatrix(f_3,x,yRuido))

y_f3=getEtiquetas(f_3,x)
print("\nAniadimos un 10% de ruido")
y_f3_ruido=aniadeRuideACadaClase(y_f3,10)

plot_datos_cuad(x, y_f3_ruido, f_3, title='Etiquetas Propias f3(x,y)=0.5(x-10)^2-(y+20)^2-400', xaxis='x axis', yaxis='y axis')
print("\nFuncion 3 matriz de confusion para etiquetas propias:")
#Imprimimos la matriz de confusion para la muestra con ruido anterior y la primera funcion
print(confusionMatrix(f_3,x,y_f3_ruido))

input("\n--- Pulsar tecla para continuar ---\n")

######################################################################################################
#Pintamos la cuarta funcion
plot_datos_cuad(x, yRuido, f_4, title='Etiquetas Lineales f4(x,y)=y-20x^2-5x+3', xaxis='x axis', yaxis='y axis')
print("\nFuncion 4 matriz de confusion para etiquetas anteriores:")
#Imprimimos la matriz de confusion para la muestra con ruido anterior y la cuarta funcion
print(confusionMatrix(f_4,x,yRuido))

y_f4=getEtiquetas(f_4,x)
print("\nAniadimos un 10% de ruido")
y_f4_ruido=aniadeRuideACadaClase(y_f4,10)

plot_datos_cuad(x, y_f4_ruido, f_4, title='Etiquetas Propias f4(x,y)=y-20x^2-5x+3', xaxis='x axis', yaxis='y axis')
print("\nFuncion 4 matriz de confusion para etiquetas propias:")
#Imprimimos la matriz de confusion para la muestra con ruido anterior y la primera funcion
print(confusionMatrix(f_4,x,y_f4_ruido))
input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

#####################################################################################
# EJERCICIO 2.a: Algoritmo del perceptron
#####################################################################################

#Error del algoritmo de una clasificacion
#Lo unico que hace es calcular la media de las etiquetas que estan mal 
#clasificadas
def Err_PLA(x,y,w):
    error = 0
    for caracteristicas, etiqueta in zip(x,y):
        #Si esta mal clasificada
        if(signo(np.matmul(w,caracteristicas))!=etiqueta):
            error+=1#Aniado uno al error total
    return error/y.shape[0] #Divido entre el numero de instancias
            

#ALGORITMO PERCEPTRON BASICO
def ajusta_PLA(datos, label, max_iter, vini):
    #vini pasa a ser w, lo copiamos para no modificar vini
    w=np.copy(vini)
    modificado = True
    numIteraciones=0
    #Mientras algun punto no este bien ajustado y el numero de iteraciones
    #sea menor que el maximo, seguimos iterando
    while modificado and numIteraciones<max_iter:
        modificado = False
        #Iteramos sobre cada una de las instancias
        for caracteristicas, etiqueta in zip(datos,label):
            #Si esta mal clasificada
            if(signo(np.matmul(w,caracteristicas))!=etiqueta):
                w = w + etiqueta*caracteristicas #Ajustamos w
                modificado = True                #Como hemos modificada w
                                                 #Volveremos a iterar sobre todo 
                                                 #el dataset cuando acabemos 
        numIteraciones+=1
        #Devolvemos el ajuste obtenido en la ultima iteracion
        #asi como el numero de iteraciones
    return w, numIteraciones  

#ALGORITMO PERCEPTRON POCKET
def ajusta_PLA_Pocket(datos, label, max_iter, vini):
    #vini pasa a ser w, lo copiamos para no modificar vini
    w=np.copy(vini)
    #Guardamos siempre el mejor valor hasta el momento para que en caso de que
    #no converja no tener un valor distinto al optimo malo
    mejor_w = np.copy(vini)
    #Calculamos el error del ajuste inicial
    menor_error = Err_PLA(datos,label,w)
    
    modificado = True
    numIteraciones=0
    #Mientras algun punto no este bien ajustado y el numero de iteraciones
    #sea menor que el maximo, seguimos iterando
    while modificado and numIteraciones<max_iter:
        modificado = False
        #Iteramos sobre cada una de las instancias
        for caracteristicas, etiqueta in zip(datos,label):
            #Si esta mal clasificada
            if(signo(np.matmul(w,caracteristicas))!=etiqueta):
                w = w + etiqueta*caracteristicas #Ajustamos w
                modificado = True                #Como hemos modificada w
                                                 #Volveremos a iterar sobre todo 
                                                 #el dataset cuando acabemos 
        #Calculamos el error tras iterar sobre toda la muestra
        error_actual = Err_PLA(datos,label,w)
        #Si el error actual es menor que el mejor que teniamos hasta el momento
        if(error_actual<menor_error):
            #Lo guardamos asi como el ajuste que hace este error minimo
            menor_error = error_actual
            mejor_w = np.copy(w)
        numIteraciones+=1
        #Devolvemos el ajuste que ha obtenido el menor error durante todo el 
        #algoritmo
        #asi como el numero de iteraciones
    return mejor_w, numIteraciones  



print("Datos ejercicio 1.2.a)")
#####################################################################################
# EJERCICIO 2.a.1: SIN RUIDO
#####################################################################################
"""
Ajustando con PLA
"""
#Inicializamos a 0
valorInicial = np.array([0,0,0])
#Aniadimos una columna de unos al principio de la matriz de caracteristicas
x_train = np.append(np.ones(x.shape[0]).reshape((-1,1)), x, axis=1)
#Ejecutamos algoritmo PLA con un numero maximo de 1000 iteraciones
w,it = ajusta_PLA(x_train,y,1000,valorInicial)
print("\nPLA, Valor inicial [0,0,0]")
print("w:[",w,"]")
print("Numero iteraciones:",it)
print("Error: ", Err_PLA(x_train,y,w))

"""
Dibujamos la funcion g encontrada por el algoritmo PLA
"""
#Dibujamos la recta obtenida de ajustar medianteel algoritmo PLA
#
# PARA DIBUJAR TODAS LAS RECTAS DE LOS PLOTS HE UTILIZADO LO QUE VOY A 
# HACER A CONTINUACION, LO VOY A COMENTAR ESTA VEZ CON MAS PROFUNDIDAD
# PERO EN LAS SIGUIENTES ESCRIBIRE UNICAMENTE QUE DIBUJO LA RECTA
#
#Generamos un array de 100 valores equidistantes entre el minimo valor de x
#y el maximo
xDraw = np.linspace(np.amin(x_train[:,1]),np.amax(x_train[:,1]), 100)
#Generamos un array de 100 valores equidistantes entre el minimo valor de y
#y el maximo
yDraw = np.linspace(np.amin(x_train[:,2]),np.amax(x_train[:,2]), 100)
#Combina los valores de los vectores xDraw e yDraw en dos matrices
X, Y = np.meshgrid(xDraw,yDraw)
#Obtenemos los valores del contorno generado por el ajuste de los pesos de w 
F = w[0]+w[1]*X+w[2]*Y
#Dibujamos el contorno generado por w
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('PLA') 
#Dibujamos los puntos con etiqueta = 1 de azul 
plt.scatter(x[y==1,0],x[y==1,1], c='blue', marker='.',label='1')
#Dibujamos los puntos con etiqueta = -1 de rojo 
plt.scatter(x[y==-1,0],x[y==-1,1], c='red', marker='.',label='-1')
plt.title(label='ScatterPlor etiquetado y PLA ajustado')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()
"""
Ajustando con PLA-Pocket
"""
#Inicializamos a 0
valorInicial = np.array([0,0,0])
#Aniadimos una columna de unos al principio de la matriz de caracteristicas
x_train = np.append(np.ones(x.shape[0]).reshape((-1,1)), x, axis=1)
#Ejecutamos algoritmo PLA-pocket con un numero maximo de 1000 iteraciones
w,it = ajusta_PLA_Pocket(x_train,y,1000,valorInicial)
print("\nPLA-Pocket Valor inicial [0,0,0]")
print("w:[",w,"]")
print("Numero iteraciones:",it)
print("Error: ", Err_PLA(x_train,y,w))

"""
Dibujamos la funcion g encontrada por el algoritmo PLA-Pocket
"""
#Dibujamos la recta obtenida de ajustar medianteel algoritmo PLA-Pocket
xDraw = np.linspace(np.amin(x_train[:,1]),np.amax(x_train[:,1]), 100)
yDraw = np.linspace(np.amin(x_train[:,2]),np.amax(x_train[:,2]), 100)
X, Y = np.meshgrid(xDraw,yDraw)
F = w[0]+w[1]*X+w[2]*Y
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('PLA-Pocket')  
#Dibujamos los puntos con etiqueta = 1 de azul 
plt.scatter(x[y==1,0],x[y==1,1], c='blue', marker='.',label='1')
#Dibujamos los puntos con etiqueta = -1 de rojo 
plt.scatter(x[y==-1,0],x[y==-1,1], c='red', marker='.',label='-1')
plt.title(label='ScatterPlor etiquetado y PLA-Pocket ajustado')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print("Valores aleatorios sin ruido")

#Aniadimos una columna de unos al principio de la matriz de caracteristicas
x_train = np.append(np.ones(x.shape[0]).reshape((-1,1)), x, axis=1)
# Random initializations
iterations = []
num_ejecuciones = 10
#Iteramos 10 veces ejecutando en cada iteracion el algoritmo PLA 
#con un punto inicial diferente generado aleatoriamente 
for i in range(0,num_ejecuciones):
    #inicializamos w aleatoriamente
    valorInicial=np.random.rand(3)
    #Ejecutamos algoritmo PLA-pocket con un numero maximo de 1000 iteraciones
    w,it = ajusta_PLA_Pocket(x_train,y,1000,valorInicial)
    print("----------------------------------")
    print(str(i)+"-.")
    print("Valor Inicial: ",valorInicial)
    print("Numero de iteraciones: "+str(it))
    #Aniadimos el numero de iteraciones a una lista para hacer posteriormente
    #la media
    iterations.append(it)

    
#Calculamos y mostramos la media por pantalla
print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")
#####################################################################################
# EJERCICIO 2.a.2: ANIADIMOS 10% DE RUIDO
#####################################################################################
"""
Simulacion sobre datos con ruido del apartado 2a
"""
# Ahora con los datos del ejercicio 1.2.b
print("Datos ejercicio 1.2.b)")
"""
Ajustando con PLA
"""
#Inicializamos a 0
valorInicial = np.array([0,0,0])
#Aniadimos una columna de unos al principio de la matriz de caracteristicas
x_train = np.append(np.ones(x.shape[0]).reshape((-1,1)), x, axis=1)
#Ejecutamos algoritmo PLA con un numero maximo de 1000 iteraciones
w,it = ajusta_PLA(x_train,yRuido,1000,valorInicial)
print("\nPLA, Valor inicial [0,0,0]")
print("w:[",w,"]")
print("Numero iteraciones:",it)
print("Error: ", Err_PLA(x_train,yRuido,w))

"""
Dibujamos la funcion g encontrada por el algoritmo PLA
"""
#Dibujamos la recta obtenida de ajustar medianteel algoritmo PLA
xDraw = np.linspace(np.amin(x_train[:,1]),np.amax(x_train[:,1]), 100)
yDraw = np.linspace(np.amin(x_train[:,2]),np.amax(x_train[:,2]), 100)
X, Y = np.meshgrid(xDraw,yDraw)
F = w[0]+w[1]*X+w[2]*Y
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('PLA')  
#Dibujamos los puntos con etiqueta = 1 de azul 
plt.scatter(x[yRuido==1,0],x[yRuido==1,1], c='blue', marker='.',label='1')
#Dibujamos los puntos con etiqueta = -1 de rojo 
plt.scatter(x[yRuido==-1,0],x[yRuido==-1,1], c='red', marker='.',label='-1')
plt.title(label='ScatterPlor etiquetado y PLA ajustado')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()
"""
Ajustando con PLA-Pocket
"""
#Inicializamos a 0
valorInicial = np.array([0,0,0])
#Aniadimos una columna de unos al principio de la matriz de caracteristicas
x_train = np.append(np.ones(x.shape[0]).reshape((-1,1)), x, axis=1)
#Ejecutamos algoritmo PLA-Pocket con un numero maximo de 1000 iteraciones
w,it = ajusta_PLA_Pocket(x_train,yRuido,1000,valorInicial)
print("\nPLA-Pocket Valor inicial [0,0,0]")
print("w:[",w,"]")
print("Numero iteraciones:",it)
print("Error: ", Err_PLA(x_train,yRuido,w))

"""
Dibujamos la funcion g encontrada por el algoritmo PLA-Pocket
"""
#Dibujamos la recta obtenida de ajustar medianteel algoritmo PLA-pocket
xDraw = np.linspace(np.amin(x_train[:,1]),np.amax(x_train[:,1]), 100)
yDraw = np.linspace(np.amin(x_train[:,2]),np.amax(x_train[:,2]), 100)
X, Y = np.meshgrid(xDraw,yDraw)
F = w[0]+w[1]*X+w[2]*Y
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('PLA-Pocket')  
#Dibujamos los puntos con etiqueta = 1 de azul 
plt.scatter(x[yRuido==1,0],x[yRuido==1,1], c='blue', marker='.',label='1')
#Dibujamos los puntos con etiqueta = -1 de rojo 
plt.scatter(x[yRuido==-1,0],x[yRuido==-1,1], c='red', marker='.',label='-1')
plt.title(label='ScatterPlor etiquetado y PLA-Pocket ajustado')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print("Valores aleatorios con ruido")

x_train = np.append(np.ones(x.shape[0]).reshape((-1,1)), x, axis=1)
# Random initializations
iterations = []
num_ejecuciones = 10
#Iteramos 10 veces ejecutando en cada iteracion el algoritmo PLA 
#con un punto inicial diferente generado aleatoriamente 
for i in range(0,num_ejecuciones):
    #inicializamos w aleatoriamente
    valorInicial=np.random.rand(3)
    #Ejecutamos algoritmo PLA-pocket con un numero maximo de 1000 iteraciones
    w,it = ajusta_PLA_Pocket(x_train,yRuido,1000,valorInicial)
    print("----------------------------------")
    print(str(i)+"-.")
    print("Valor Inicial: ",valorInicial)
    print("Numero de iteraciones: "+str(it))
    #Aniadimos el numero de iteraciones a una lista para hacer posteriormente
    #la media
    iterations.append(it)
    
print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
###############################################################################
"""
Funcion para calcular el error sigmoidal medio, para una muestra etiquetada (x,y)
Y un ajuste W
"""
def Err_sig(x,y,w):
    error = 0
    for caracteristicas, etiqueta in zip(x,y):
        #Acumulamos el error sigmoidal que generan cada una de las instancias
        error += np.log(1 + np.e**(-(etiqueta*np.matmul(w.T,caracteristicas))))
    return error/y.shape[0]#Dividimos entre el numero de instancias
    
"""
Funcion para calcular el gradiente del error sigmoidal en un punto w
y para un conjunto de datos (x,y)
"""
def gradErrSig(x,y,w):
    #Creamos un array de ceros con el mismo tamanio que w
    newW = np.zeros(w.shape[0])
    #Iteramos sobre cada elemento de x, y acumulamos en w
    for i in range(y.shape[0]):
        newW += (y[i]*x[i])/(1+np.e**(y[i]*np.matmul(w.T,x[i])))
    return -newW/y.shape[0]#Dividimos entre el numero de instancias

"""
Funcion que nos divide en un array en bachs de un tamanio dado
"""
def getBachs(indices,tamanioBachs=32):
    #Iteramos de 0 al tamanio de la muestra dando pasos de tamanioBachs elementos
    for i in range(0, indices.shape[0], tamanioBachs):
        #En cada iteracion devolvemos un mini-bach, cuando se vuelva a llamar
        #a la funcion esta seguira por donde iba
        yield indices[i:i+tamanioBachs]


"""
Algoritmo del gradiente descendente estocastico con regresion logistica
"""  
def sgdRL(x, y, w, numMaxIter, learning_rate, diferenciaMinima=0.01, tamanioBachs=1):
    iterations=0
    #Asignamos a diferencia actual un valor mayor que diferencia minima para 
    #entrar en el bucle
    diferenciaActual = diferenciaMinima+1
    #Mientras el numero de iteraciones sea menor que el maximo y la norma
    #de la diferencia entre W(t)-w(t+1) sea menor que diferenciaMinima
    while(iterations<numMaxIter and diferenciaMinima<diferenciaActual):
        #Generamos un vector que indexe cada una de las instancias
        indices = np.arange(y.shape[0])
        #Barajamos los indices
        np.random.shuffle(indices)
        #Obtenemos una lista de mini-Baches de la muestra
        mis_baches = list(getBachs(indices,tamanioBachs))
        #Guardamos el ajuste actual para compararlo posteriormente
        w_old = np.copy(w)
        for mi_bach in mis_baches:
            #Calculamos el incremento a W del descenso de gradiente sigmoidal
            incrementoW = - learning_rate * gradErrSig(x[mi_bach[:]],y[mi_bach[:]],w)
            #Modificamos w
            w = w + incrementoW
        iterations+=1
        #Calculamos la norma de la diferencia de ajustes actual |w_old - w|
        diferenciaActual=np.linalg.norm(w_old-w)
        
    return w, iterations


"""
Generamos la muestra y aplicamos el gradiente descendente estocastico
con regresion logistica
"""
#Generamos 100 instancias, en dos dimensiones con una distribucion normal
#en un rango [0,2]
x = simula_unif(100, 2, [0,2])
#Dibujamos la muestra en un scatter plot
plt.scatter(x[:,0],x[:,1], c='blue', marker='.')
plt.title(label='Dataset entrenamiento sin etiquetar')
plt.xlabel('x')
plt.xlim([0,2])
plt.ylabel('y')
plt.ylim([0,2])
plt.show()

print("Dibujamos la muestra sin etiquetar")

input("\n--- Pulsar tecla para continuar ---\n")

#Generamos una recta de forma aleatoria 
a,b = simula_recta([0,2])
#Creamos el vector de las etiquetas
y = np.ndarray(shape=(x.shape[0]),dtype=np.float64)
#Asignamos a cada instancia una etiqueta una etiqueta 1 o -1 en funcion de si
#esta a un lado u a otro de la recta
for i,instancia in enumerate(x):
    y[i] = f(instancia[0],instancia[1],a,b)

#Dibujamos el dataset etiquetado
#Instancias etiquetadas como 1 de azul
plt.scatter(x[y==1,0],x[y==1,1], c='blue', marker='.',label='1')
#Instancias etiquetadas como -1 de rojo
plt.scatter(x[y==-1,0],x[y==-1,1], c='red', marker='.',label='-1')
#Asignamos un titulo
plt.title(label='Dataset entrenamiento etiquetado por y=x'+str(round(a,2))+"+"+str(round(b,2)))
plt.xlabel('x')
plt.xlim([0,2])
plt.ylabel('y')
plt.ylim([0,2])
plt.legend()
plt.show()

print("Dibujamos la muestra etiqueda por:"+
      "y=x"+str(round(a,2))+"+"+str(round(b,2)))

input("\n--- Pulsar tecla para continuar ---\n")
#Posicionamos el punto inicial en 0,0,0
w=np.array([0,0,0])
#Aniadimos una columna de unos al principio del dataset x
x_train = np.append(np.ones(x.shape[0]).reshape((-1,1)), x, axis=1)
#Ejecutamos el algoritmo de gradiente descendente estocastico
#con regresion logistica
#Asignamos un numero de iteraciones muy grande para que llegue antes a 
#al minimo de la norma de la diferencia que al numero de iteraciones
#maximas
w,it=sgdRL(x_train, y, w, 10000000, 0.01, 0.01,1)

#Dibujamos el dataset de entrenamiento etiquetado, y con la recta obtenida de 
#aplicar el algoritmo de descenso del gradiente estocastico basado en la RL
xDraw = np.linspace(np.amin(x_train[:,1]),np.amax(x_train[:,1]), 100)
yDraw = np.linspace(np.amin(x_train[:,2]),np.amax(x_train[:,2]), 100)
X, Y = np.meshgrid(xDraw,yDraw)
F = w[0]+w[1]*X+w[2]*Y
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('SGD-RL')
#Instancias etiquetadas como 1 de azul  
plt.scatter(x[y==1,0],x[y==1,1], c='blue', marker='.',label='1')
#Instancias etiquetadas como -1 de rojo
plt.scatter(x[y==-1,0],x[y==-1,1], c='red', marker='.',label='-1')
plt.title(label='Dataset entrenamiento sgdRL')
plt.xlabel('x')
plt.xlim([0,2])
plt.ylabel('y')
plt.ylim([0,2])
plt.legend()
plt.show()

print("Error sigmoidal en la muestra: "+str(Err_sig(x_train,y,w)))
input("\n--- Pulsar tecla para continuar ---\n")
    
"""
Generamos los datos de test
"""
#Generamos 1000 instancias, en dos dimensiones con una distribucion uniforme
#en un rango [0,2]
x = simula_unif(1000, 2, [0,2])
#Creamos el vector de las etiquetas
y = np.ndarray(shape=(x.shape[0]),dtype=np.float64)
#Asignamos a cada instancia una etiqueta una etiqueta 1 o -1 en funcion de si
#esta a un lado u a otro de la recta
for i,instancia in enumerate(x):
    y[i] = f(instancia[0],instancia[1],a,b)

#Dibujamos el dataset de test etiquetado, y con la recta obtenida de 
#aplicar el algoritmo anterior
#Dibujamos el contorno generado por los pesos w
cont = plt.contour(X,Y,F,[0])
cont.collections[0].set_label('SGD-RL')  
#Dibujamos los puntos con valor 1 de color azul
plt.scatter(x[y==1,0],x[y==1,1], c='blue', marker='.',label='1')
#Dibujamos los puntos con valor -1 de color rojo
plt.scatter(x[y==-1,0],x[y==-1,1], c='red', marker='.',label='-1')
plt.title(label='Dataset test sgdRL')
plt.xlabel('x')
plt.xlim([0,2])
plt.ylabel('y')
plt.ylim([0,2])
plt.legend()
plt.show()

#Aniadimos una columna de unos al principio del test para que se pueda 
#calcular en error
x_test = np.append(np.ones(x.shape[0]).reshape((-1,1)), x, axis=1)
print("Error sigmoidal en test: "+str(Err_sig(x_test,y,w)))
input("\n--- Pulsar tecla para continuar ---\n")


"""
1-.Generamos una muestra, y la etiquetamos con la recta aleatoria obtenida anteriormente
2-.Ajustamos mediante SGD-RL dicha muestra
3-.Vemos como funciona sobre un nuevo test etiquetado con la funcion aleatoria
   anterior
"""

print("Iteramos 100 veces")

#Generamos una recta de forma aleatoria 
a,b = simula_recta([0,2])

N=100
ERR_OUT_SIG=0
NUM_IT=0
for numExperimento in range(0,N):
    #Generamos 100 instancias, en dos dimensiones con una distribucion normal
    #en un rango [0,2]
    x = simula_unif(100, 2, [0,2])
    #Creamos el vector de las etiquetas
    y = np.ndarray(shape=(x.shape[0]),dtype=np.float64)
    #Asignamos a cada instancia una etiqueta una etiqueta 1 o -1 en funcion de si
    #esta a un lado u a otro de la recta
    for i,instancia in enumerate(x):
        y[i] = f(instancia[0],instancia[1],a,b)

    #Posicionamos el punto inicial en 0,0,0
    w=np.array([0,0,0])
    #Aniadimos una columna de unos al principio del dataset x
    x_train = np.append(np.ones(x.shape[0]).reshape((-1,1)), x, axis=1)
    #Ejecutamos el algoritmo de gradiente descendente estocastico
    #con regresion logistica para ajustar la muestra
    w,it=sgdRL(x_train, y, w, 10000000, 0.01, 0.01,1)
    NUM_IT+=it
    #Generamos los datos de test
    #Generamos 1000 instancias, en dos dimensiones con una distribucion normal
    #en un rango [0,2]
    x = simula_unif(1000, 2, [0,2])
    #Creamos el vector de las etiquetas
    y = np.ndarray(shape=(x.shape[0]),dtype=np.float64)
    #Asignamos a cada instancia una etiqueta una etiqueta 1 o -1 en funcion de si
    #esta a un lado u a otro de la recta
    for i,instancia in enumerate(x):
        y[i] = f(instancia[0],instancia[1],a,b)
    #Aniadimos la columna de unos al principio del test
    x_test = np.append(np.ones(x.shape[0]).reshape((-1,1)), x, axis=1)
    #Acumulamos el error sigmoidal
    ERR_OUT_SIG+=Err_sig(x_test,y,w)
    
   
    if numExperimento%10==0:
        print(str(numExperimento)+"/"+str(N))
    
#Calculamos la media del error sigmoidal
ERR_OUT_SIG = ERR_OUT_SIG/N
#Calculamos la media del numero de iteraciones
NUM_IT = NUM_IT/N

print ('Valores medios tras 100 ejecuciones:\n')
print ("Iteraciones medias: ", NUM_IT)
print ("Eout medio sigmoidal: ", ERR_OUT_SIG)
#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
print("Dibujando los datos train y test")
#Dibujamos un scatter plot de los datos de entrenamiento
fig, ax = plt.subplots()
#Dibujamos los puntos que pertenecen a la clase -1 de color rojo
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), '.', color='red', label='4')
#Dibujamos los puntos que pertenecen a la clase 1 de color azul
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), '.', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()
#Dibujamos un scatter plot de los datos de test
fig, ax = plt.subplots()
#Dibujamos los puntos que pertenecen a la clase -1 de color rojo
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), '.', color='red', label='4')
#Dibujamos los puntos que pertenecen a la clase 1 de color azul
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), '.', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
# Pseudoinversa
def pseudoinverse(X):
    #Calculo de la Pseudo Inversa de x
    pseudoinversa = np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T)
    return pseudoinversa

#Ajuste de w mediante la pseudo inversa
def Regress_Lin(x,y):
    #Obtenemos la pseudo inversa
    pseudoinversa = pseudoinverse(x)
    #Valculamos w(Producto matricial)
    w = np.matmul(pseudoinversa,y)
    return w


#LINEAR REGRESSION FOR CLASSIFICATION 

#Calculamos el ajuste mediante el descenso de gradiente estcastico 
#con regresion logistica
w=Regress_Lin(x, y)


"""
Dibujamos el ajuste obtenido en la muetra
"""
fig, ax = plt.subplots()
#Dibujamos la recta ajustada mediante la pseudo inversa
xDraw = np.linspace(np.amin(x[:,1]),np.amax(x[:,1]), 100)
yDraw = np.linspace(np.amin(x[:,2]),np.amax(x[:,2]), 100)
X, Y = np.meshgrid(xDraw,yDraw)
F = w[0]+w[1]*X+w[2]*Y
cont = ax.contour(X,Y,F,[0],zorder=2)
cont.collections[0].set_label('Pseudo-Inversa') 
#Dibujamos los puntos y=-1
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), '.', color='red', label='4',zorder=1)
#Dibujamos los puntos y=1
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), '.', color='blue', label='8',zorder=1)
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Ajuste Pseudo Inversa sobre los datos de entrenamiento')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

print("Error Pseudo-Inversa")
print("Error in: "+str(Err_PLA(x,y,w)))
print("Error test: "+str(Err_PLA(x_test,y_test,w)))


"""
Dibujamos el ajuste obtenido en el test
"""
fig, ax = plt.subplots()
#Dibujamos la recta ajustada
cont = ax.contour(X,Y,F,[0],zorder=2)
cont.collections[0].set_label('Pseudo-Inversa') 
#Dibujamos los puntos y=-1
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), '.', color='red', label='4',zorder=1)
#Dibujamos los puntos y=1
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), '.', color='blue', label='8',zorder=1)
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Ajuste Pseudo Inversa sobre el test')
ax.set_xlim((0, 1))
plt.legend()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
#POCKET ALGORITHM
#Utilizamos el ajuste anterior como punto de inicio del PLA-pocket
#Calculamos el ajuste mediante PLA-Pocket
w,it = ajusta_PLA_Pocket(x,y,5000,w)


"""
Dibujamos el ajuste obtenido en la muetra
"""
fig, ax = plt.subplots()
#Dibujamos la recta ajustada mediante PLA-Pocket
xDraw = np.linspace(np.amin(x[:,1]),np.amax(x[:,1]), 100)
yDraw = np.linspace(np.amin(x[:,2]),np.amax(x[:,2]), 100)
X, Y = np.meshgrid(xDraw,yDraw)
F = w[0]+w[1]*X+w[2]*Y
cont = ax.contour(X,Y,F,[0],zorder=2)
cont.collections[0].set_label('PLA pocket') 
#Dibujamos los puntos y=-1
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), '.', color='red', label='4',zorder=1)
#Dibujamos los puntos y=1
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), '.', color='blue', label='8',zorder=1)
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Ajuste PLA-Pocket sobre los datos de entrenamiento')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

#Guardamos los erros para calcular posteriormente las cotas
Error_PLA_train=Err_PLA(x,y,w)
Error_PLA_test=Err_PLA(x_test,y_test,w)

print("Error PLA-pocket")
print("Error in: "+str(Error_PLA_train))
print("Error test: "+str(Error_PLA_test))

"""
Dibujamos el ajuste obtenido en el test
"""
fig, ax = plt.subplots()
#Dibujamos la recta ajustada
cont = ax.contour(X,Y,F,[0],zorder=2)
cont.collections[0].set_label('PLA pocket') 
#Dibujamos los puntos y=1
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), '.', color='red', label='4',zorder=1)
#Dibujamos los puntos y=-1
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), '.', color='blue', label='8',zorder=1)
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Ajuste PLA-Pocket sobre el test')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
#Calculamos una cota basada en Ein
def cotaVCsobreClasePercepron2D(error_in,dvc,N,delta):
    cota = error_in+np.sqrt((8/N)*np.log(4*((2*N)**dvc+1)/delta))
    return cota

#Calculamos una cota basada en Etest
def cotaHoeffding(error_test,N,delta):
    cota = error_test+np.sqrt((1/(2*N))*np.log(2/delta))
    return cota
    


print("\nCota VC basada en Ein")
#En el guion se explica por que dvc=3
#Es debido a que el punto de ruptura de la clase del perceptron
#en dos dimensiones es 4
print("Eout<="+str(cotaVCsobreClasePercepron2D(Error_PLA_train,3,y.shape[0],0.05)))


print("\nCota Hoeffding basada en Etest")
print("Eout<="+str(cotaHoeffding(Error_PLA_test,y_test.shape[0],0.05)))

input("\n--- Pulsar tecla para continuar ---\n")
print("\nCreamos dos graficas para explicar el punto de ruptura en el guion")
#Generamos puntos no separables por el perceptron
x = np.array([[1,0],[0,1],[0,-1],[-1,0]])
y = np.array([1,-1,-1,1])

fig, ax = plt.subplots()
#Dibujamos los puntos y=-1
ax.plot(x[y == -1,0], x[y == -1,1], 'o', color='red', label='-1',zorder=1)
#Dibujamos los puntos y=1
ax.plot(x[y == 1,0], x[y == 1,1], 'o', color='blue', label='1',zorder=1)
ax.set(xlabel='x', ylabel='y', title='Ajuste de puntos para N=4 no separable')
plt.legend()
plt.show()

#Generamos puntos separables por el perceptron
x = np.array([[1,0],[0,1],[0,-1],[-1,0]])
y = np.array([1,1,-1,-1])

fig, ax = plt.subplots()
#Dibujamos los puntos y=-1
ax.plot(x[y == -1,0], x[y == -1,1], 'o', color='red', label='-1',zorder=1)
#Dibujamos los puntos y=1
ax.plot(x[y == 1,0], x[y == 1,1], 'o', color='blue', label='1',zorder=1)
#Dibujamos la linea que los separa
point1 = [-1, 1]
point2 = [1, -1]
x_values = [point1[0], point2[0]] 
y_values = [point1[1], point2[1]]
ax.plot(x_values, y_values, label='Funcion lineal')
ax.set(xlabel='x', ylabel='y', title='Ajuste de puntos para N=4 separable')
plt.legend()
plt.show()
