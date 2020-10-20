#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================
# Ampliación de Inteligencia Artificial
# Implementación de clasificadores 
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autores:
#
# APELLIDOS: Garrucho Fernández
# NOMBRE: Javier
#
#
# APELLIDOS: Núñez García
# NOMBRE: José Enrique
# ----------------------------------------------------------------------------



import random
import numpy as np
import math
from carga_datos import *


# ----------------------------------------------
# PARTICIÓN ENTRENAMIENTO PRUEBA
# ----------------------------------------------


# Recibiendo un conjunto de datos X, y su correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser aleatoria y
# estratificada respecto del valor de clasificación. Por supuesto, en el orden
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.



def particion_entr_prueba(X, y, test=0.20):
    i_train, i_test = [], []  # Listas con los indices de los ejemplos

    # DIVIDIR POR CLASES
    indices_por_clase = {c: [] for c in np.unique(y)}

    for i in range(len(y)):
        indices_por_clase[y[i]].append(i)  # Cada elemento de 'y' se guarda por su clase

    # COGER CONJUNTOS DE ENTRENAMIENTO Y TEST DE CADA CLASE
    for indices in indices_por_clase.values():
        n = int(round(len(indices) * test, 0))  # Numero de ejemplos en el conjunto de test de cada clase
        random.shuffle(indices)  # Mezclar lista de indices
        i_test += indices[:n]  # Coger indices del conjunto de test
        i_train += indices[n:]  # Coger indices del conjunto de entrenamiento

    # UNA ULTIMA MEZCLA para que no esten todos los de la misma clase juntos
    random.shuffle(i_test)
    random.shuffle(i_train)

    # TRANSFORMAR LISTA DE INDICES A LISTAS DE EJEMPLOS Y CLASIFICACIONES
    X_test, y_test = zip(*map(lambda i: (X[i], y[i]), i_test))
    X_train, y_train = zip(*map(lambda i: (X[i], y[i]), i_train))

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)




# ----------------------------------------------
# IMPLEMENTACIÓN DE CLASIFICADOR NAIVE BAYES CATEGÓRICA
# ----------------------------------------------

# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1)
# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los
#   datos y los valores de clasificación respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan.
# * Método clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribución de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase.
# * Método clasifica: recibe un ejemplo (en forma de array de numpy) y
#   devuelve la clase que el modelo predice para ese ejemplo.



class ClasificadorNoEntrenado(Exception): pass


def normaliza(probabilidades):  # Recive un array con las probabilidades y lo normaliza
    return probabilidades / probabilidades.sum()


def exp_normaliza(probabilidades):
    b = probabilidades.max()
    y = np.exp2(probabilidades - b)  # Para evitar overflow restamos a todo el vector de probabilidades el mayor
    return y / y.sum()               # elemento de este, 'b'. Se puede demostrar que esta resta no afecta al calculo final.



class NaiveBayes():

    def __init__(self, k=1):
        self.k = k
        self.entrenado = False

    def entrena(self, X, y):
        n_ejemplos = len(y)
        n_atributos = len(X[0])  # Numero de atributos posibles
        valores = [np.unique(X[:, a]).tolist() for a in range(n_atributos)]  # Valores posibles para cada atributo

        self.clases = np.unique(y)  # Clases posibles


        # 'count_valores_por_clase' almacenara el numero de veces que aparece cada valor de cada atributo en cada
        # clase en un diccionario con la siguiente forma:
        #
        #{'+':[{"sol":3, "nubes":1}, {"Fuerte":2,"Debil":2}],
        # '-':[{"sol":1, "nubes":2}, {"Fuerte":3,"Debil":0}]}
        self.count_valores_por_clase = {c: [{v: 0 for v in valores[a]}  # Inicializacion a 0; calculado en bucle
                                            for a in range(n_atributos)]
                                        for c in self.clases}

        # 'n_por_clase' almacena el numero de ejemplos de cada clase en un diccionario de la siguiente forma:
        #
        # {'+':4,'-':3}
        self.n_por_clase = {c: 0 for c in self.clases} # Inicializacion a 0; calculado en bucle

        # Numero de posibles valores para cada atributo
        self.n_valores_por_atributo = list(map(lambda l: len(l), valores))

        # CALCULAR "n_por_clase" y "count_valores_por_clase"
        for i in range(n_ejemplos):
            clase = y[i]
            self.n_por_clase[clase] += 1
            for a, v in enumerate(X[i, :]):
                self.count_valores_por_clase[clase][a][v] += 1

        self.prob_clases = np.array([(v / n_ejemplos) for v in self.n_por_clase.values()])   # Probabilidades de cada clase
        self.entrenado = True


    def probabilidad(self, atributo, valor, clase):
        return (self.count_valores_por_clase[clase][atributo][valor] + self.k) / \
               (self.n_por_clase[clase] + self.k * self.n_valores_por_atributo[atributo])

    def clasifica_prob(self, ejemplo):
        if not self.entrenado:
            raise ClasificadorNoEntrenado

        probabilidades = np.log2(self.prob_clases.copy())   # log-probabilidades de cada clase
        for a, valor in enumerate(ejemplo):
            probabilidades += np.log2(np.array([self.probabilidad(a, valor, clase) for clase in self.clases]))


        return dict(zip(self.clases, exp_normaliza(probabilidades)))

    def clasifica(self, ejemplo):
        if not self.entrenado:
            raise ClasificadorNoEntrenado

        probabilidades = np.log2(self.prob_clases) # log-probabilidades de cada clase
        for a, valor in enumerate(ejemplo):
            probabilidades += np.log2(np.array([self.probabilidad(a, valor, clase) for clase in self.clases]))

        return self.clases[np.argmax(probabilidades)]


# ------------------------------------------------------------------------------
# Ejemplo "jugar al tenis":

# >>> nb_tenis=NaiveBayes(k=0.5)
# >>> nb_tenis.entrena(X_tenis,y_tenis)
# >>> ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
# >>> nb_tenis.clasifica_prob(ej_tenis)
# {'no': 0.7564841498905581, 'si': 0.24351585014409202}
# >>> nb_tenis.clasifica(ej_tenis)
# 'no'
# ------------------------------------------------------------------------------



# ----------------------------------------------
# CÁLCULO RENDIMIENTO
# ----------------------------------------------

def rendimiento(clasificador, X, y):
    aciertos = 0
    for ejemplo, clase in zip(X, y):
        aciertos += 1 if (clasificador.clasifica(ejemplo) == clase) else 0

    return aciertos / len(y)




# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Concesión de credito
# - Críticas de películas en IMDB

# En todos los casos, será necesario separar los datos en entrenamiento y
# prueba, para dar la valoración final de los clasificadores obtenidos (usar
# para ello la función particion_entr_prueba anterior). Ajustar también el
# valor del parámetro de suavizado k. Mostrar el proceso realizado en cada
# caso, y los rendimientos obtenidos.


# SEPARAR DATOS (datos de IMDB ya separados)
# >>> X_train_votos, X_test_votos, y_train_votos, y_test_votos = particion_entr_prueba(X_votos, y_votos, test=0.25)
# >>> X_train_credito, X_test_credito, y_train_credito, y_test_credito = particion_entr_prueba(X_credito, y_credito, test=0.25)
#
# INICIALIZAR NAIVE BAYES
# >>> nb_votos = NaiveBayes()
# >>> nb_credito = NaiveBayes()
# >>> nb_imdb = NaiveBayes()
#
# ENTRENAMIENTO
# >>> nb_votos.entrena(X_train_votos, y_train_votos)
# >>> nb_credito.entrena(X_train_credito, y_train_credito)
# >>> nb_imdb.entrena(X_train_imdb, y_train_imdb)
#
# MOSTRAR RENDIMIENTO
# >>> print("Rendimiento sobre test nb_votos: ", rendimiento(nb_votos, X_test_votos, y_test_votos))
# Rendimiento sobre test nb_votos:  0.926605504587156

# >>> print("Rendimiento sobre test nb_credito: ", rendimiento(nb_credito, X_test_credito, y_test_credito))
# Rendimiento sobre test nb_credito:  0.6604938271604939

# >>> print("Rendimiento sobre test nb_imd: ", rendimiento(nb_imdb, X_test_imdb, y_test_imdb))
# Rendimiento sobre test nb_imd:  0.785

# Para buscar el mejor valor para k usamos el metodo de grid_search:

def grid_search_NV(X,y, k_range):
    mejor = 0
    mejor_k = 0
    for k in range(k_range):
        rend = rendimiento_validacion_cruzada(NaiveBayes, {'k':k}, X, y)

        if rend > mejor:
            mejor = rend
            mejor_k = k

    print("Mejor rendimiento:", mejor)
    print("Con k =",mejor_k)


#BUSCANDO K PARA VOTOS
# >>> grid_search_NV(X_train_votos, y_train_votos, 50)
# Mejor rendimiento: 0.8874877810361681
# Con k = 4
#
# Para ver el rendimiento real sobre el conjunto de test:
# >>> nb_votos = NaiveBayes(k=4)
# >>> nb_votos.entrena(X_train_votos, y_train_votos)
# >>> print("Rendimiento sobre test nb_votos: ", rendimiento(nb_votos, X_test_votos, y_test_votos))
# Rendimiento sobre test nb_votos:  0.944954128440367

#BUSCANDO K PARA CREDITOS
# >>> grid_search_NV(X_train_credito, y_train_credito, 50)
# Mejor rendimiento: 0.6717687074829932
# Con k = 9
#
# Para ver el rendimiento real sobre el conjunto de test:
# >>> nb_credito = NaiveBayes(k=9)
# >>> nb_credito.entrena(X_train_credito, y_train_credito)
# >>> print("Rendimiento sobre test nb_credito: ", rendimiento(nb_credito, X_test_credito, y_test_credito))
# Rendimiento sobre test nb_credito:  0.6666666666666666

#BUSCANDO K PARA IMDB
# >>> grid_search_NV(X_train_imdb, y_train_imdb, 25)    (TARDA)
# Mejor rendimiento: 0.7919960200508829
# Con k = 0
#
# Para ver el rendimiento real sobre el conjunto de test:
# >>> nb_imdb = NaiveBayes(k=0)
# >>> nb_imdb.entrena(X_train_imdb, y_train_imdb)
# >>> print("Rendimiento sobre test nb_imdb: ", rendimiento(nb_imdb, X_test_imdb, y_test_imdb))
# Rendimiento sobre test nb_imdb:  0.785


# =================================================
# IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================


def rendimiento_validacion_cruzada(clase_clasificador, params, X, y, n=5):
    Xfolds = [X[i:i + len(X) // n + 1] for i in range(0, len(X), len(X) // n + 1)]
    yfolds = [y[i:i + len(y) // n + 1] for i in range(0, len(y), len(y) // n + 1)]

    rendimientos = []

    for x in range(n):
        clasificador = clase_clasificador(**params)
        Xfolds_c = Xfolds.copy()
        yfolds_c = yfolds.copy()
        Xfolds_c.pop(x)
        yfolds_c.pop(x)

        clasificador.entrena(np.concatenate(Xfolds_c), np.concatenate(yfolds_c))
        rendimientos.append(rendimiento(clasificador, Xfolds[x], yfolds[x]))

    return np.mean(rendimientos)

# ========================================================
# MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# ========================================================


class RegresionLogisticaMiniBatch():

    def __init__(self, clases=[0, 1], normalizacion=False,
                 rate=0.1, rate_decay=False, batch_tam=64, n_epochs=200,
                 pesos_iniciales=None):
        self.clases = clases
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.n_epochs = n_epochs
        self.pesos_iniciales = pesos_iniciales
        self.entrenado = False


    def entrena(self, X, y):
        if self.pesos_iniciales is None:
            self.pesos = np.random.rand(X.shape[1])
        else:
            self.pesos = self.pesos_iniciales

        for i in range(self.n_epochs):
            if self.rate_decay is True:
                self.rate = (self.rate) * (1 / (1 + i))
            x_batchs, y_batchs = self.division_batches(X, y)
            for x in range(len(x_batchs)):
                X_batch = x_batchs[x]
                Y_batch = y_batchs[x]
                self.pesos = self.pesos - self.rate * np.dot(X_batch.T, sigmoide(np.dot(X_batch, self.pesos)) - Y_batch)

        self.entrenado = True

    def clasifica_prob(self, ejemplo): 
        if not self.entrenado:
            raise ClasificadorNoEntrenado

        hipotesis = sigmoide(np.dot(self.pesos, ejemplo))
        return {1:hipotesis, 0: 1-hipotesis}

    def clasifica(self, ejemplo):
        if not self.entrenado:
            raise ClasificadorNoEntrenado

        hipotesis = sigmoide(np.dot(self.pesos, ejemplo))
        if hipotesis > 0.5:
            return self.clases[1]
        else:
            return self.clases[0]

    def division_batches(self, X, y):

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        x_reordenado = X[indices]
        y_reordenado = y[indices]

        x_batches = [x_reordenado[i:i + self.batch_tam] for i in range(0, len(x_reordenado), self.batch_tam)]
        y_batches = [y_reordenado[i:i + self.batch_tam] for i in range(0, len(y_reordenado), self.batch_tam)]
        return x_batches, y_batches


from scipy.special import expit


def sigmoide(x):
    return expit(x)



# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama:

# Xe_cancer,Xp_cancer,ye_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer)
# lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True,n_epochs=1000)
# lr_cancer.entrena(Xe_cancer,ye_cancer)

# print(rendimiento(lr_cancer,Xe_cancer,ye_cancer))
# 0.9912280701754386

# print(rendimiento(lr_cancer,Xp_cancer,yp_cancer))
# 0.9557522123893806

# -----------------------------------------------------------------



# >>> cancer, Xp_cancer, ye_cancer, yp_cancer = particion_entr_prueba(X_cancer, y_cancer)
# >>> Xe_votos, Xp_votos, ye_votos, yp_votos = particion_entr_prueba(X_votos, y_votos)


def ajustar_hiperparam(Xe, ye, Xp, yp):

    Xe, Xv, ye, yv = particion_entr_prueba(Xe, ye)

    mejor_rendimiento = 0
    mejor_rendimiento_params = []
    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for n_epochs in [50, 100, 150, 200]:
            for batch_size in [32, 64, 128, 256]:
                lr = RegresionLogisticaMiniBatch(rate=rate, rate_decay=True, batch_tam=batch_size, n_epochs=100)
                lr.entrena(Xe, ye)
                print("=================================")
                print("Rendimiento con rate: {}, n_epochs: {}, batch_tam: {}".format(rate, n_epochs, batch_size))
                rend = rendimiento(lr, Xv, yv)
                print(rend)
                print("=================================")
                if rend > mejor_rendimiento:
                    mejor_rendimiento = rend
                    mejor_rendimiento_params = [rate, n_epochs, batch_size]
    print("El mejor rendimiento sobre el conjunto de validacion ha sido: {}, usando los parametros rate: {}, n_epochs: {} y batch_tam: {}".format(
        mejor_rendimiento, mejor_rendimiento_params[0], mejor_rendimiento_params[1], mejor_rendimiento_params[2]))
    lr = RegresionLogisticaMiniBatch(rate=mejor_rendimiento_params[0], rate_decay=True, batch_tam=mejor_rendimiento_params[1], n_epochs=mejor_rendimiento_params[2])
    lr.entrena(Xe, ye)
    rendimiento_prueba = rendimiento(lr, Xp, yp)
    print("Lo que nos proporciona un rendimiento sobre el conjunto de test: {}".format(rendimiento_prueba))


# >>> ajustar_hiperparam(Xe_cancer, ye_cancer, Xp_cancer, yp_cancer)
# >>> ajustar_hiperparam(Xe_votos, ye_votos, Xp_votos, yp_votos)


# =====================================
# CLASIFICACIÓN MULTICLASE
# =====================================


# ------------------------------------
# Implementación de One vs Rest
# ------------------------------------





class RL_OvR():
    def __init__(self, clases, rate=0.1, rate_decay=False, batch_tam=64, n_epochs=200):
        self.clases = clases
        self.clasificadores = [RegresionLogisticaMiniBatch(clases=[0,1], normalizacion=False,rate=rate,rate_decay=rate_decay,
                                                      batch_tam=batch_tam,n_epochs=n_epochs) for _ in clases]   # Inicializa un clasificador para cada clase, en el cual las clases posibles son 1(pertenece a la clase) o 0 (no pertenece)


    def entrena(self, X, y):
        for clase, clasificador in zip(self.clases, self.clasificadores):
            y_ovr = np.vectorize(lambda y_i: 1 if  y_i == clase else 0)   # Convierte el vector 'y' a one vs rest, con 1 si es la clase buscada y 0 si es cualquier otra
            clasificador.entrena(X,y_ovr(y))


    def clasifica(self, ejemplo):
        probabilidades = [clasificador.clasifica_prob(ejemplo)[1] for clasificador in self.clasificadores] # Para cada clasificador (uno para cada clase) toma la probabilidad de que el ejemplo pertenezca a esa clase #TODO "clasif.clasifica_prob(ejemplo)**[1]**" Comprobar que se coge la probabilidad de la clase que se quiere
        return self.clases[np.argmax(probabilidades)]   # Devuelve la clase que mayor probabilidad haya obtenido



#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20,n_epochs=1000)

# >>> rl_iris.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------


# ---------------------------------------------------------
# Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


# Convierte la imagen en un array unidimensional de caracteristicas binarias
def flat_bin(ls):
    return [0 if e == ' ' else 1 for l in ls for e in l if e != '\n']


# Devuelve un array con todos los digitos separados y en forma de array de caracteristicas
def lee_digitos(url):
    digitos = []
    f = open(url)
    a = np.array(f.readlines())

    for i in range(0, len(a),28):
        digitos.append(flat_bin(a[i:i+28]))

    f.close()
    return np.array(digitos)


# Devuelve un array de labels
def lee_labels(url):
    y = []
    with open(url) as fp:
        for line in fp:
            y.append(line[0]) # [0] para solo coger el numero y no '\n'

    return np.array(y)


# Devuelve los arrays ya preparados y separados
def carga_digitos():
    urls_X = ["trainingimages","validationimages","testimages"]
    urls_y = ["traininglabels", "validationlabels", "testlabels"]
    res = []

    for url in urls_X:
        res.append(lee_digitos("./digitdata/"+url))

    for url in urls_y:
        res.append(lee_labels("./digitdata/"+url))

    return tuple(res)


# >>> X_train_digits, X_validation_digits, X_test_digits, y_train_digits, y_validation_digits, y_test_digits = carga_digitos()
#
# >>> clases_digits = ['0','1','2','3','4','5','6','7','8','9']
# >>> rl_digits = RL_OvR(clases_digits, rate_decay=True)
# >>> rl_digits.entrena(X_train_digits, y_train_digits)
#
# >>> print("Rendimiento sobre entrenamiento rl_digits: ", rendimiento(rl_digits, X_train_digits, y_train_digits))
# Rendimiento sobre entrenamiento rl_digits: 0.9248
# >>> print("Rendimiento sobre test rl_digits: ", rendimiento(rl_digits, X_test_digits, y_test_digits))
# Rendimiento sobre test rl_digits: 0.848


# Ahora para ajustar los parametros utilizaremos el metodo de grid-search:

def grid_search_OvR(clases, Xe, Xv, Xp, ye, yv, yp):
    mejor_rendimiento = 0
    mejor_rendimiento_params = {"rate": 0, "n_epochs": 0, "batch_tam": 0}
    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for n_epochs in [50, 100, 150, 200]:
            for batch_size in [32, 64, 128, 256]:
                lr = RL_OvR(clases, rate_decay=True, batch_tam=batch_size, n_epochs=n_epochs, rate=rate)
                lr.entrena(Xe, ye)
                print("=================================")
                print("Rendimiento con rate: {}, n_epochs: {}, batch_tam: {}".format(rate, n_epochs, batch_size))
                rend = rendimiento(lr, Xv, yv)
                print(rend)
                print("=================================")
                if rend > mejor_rendimiento:
                    mejor_rendimiento = rend
                    mejor_rendimiento_params = {"rate": rate, "n_epochs": batch_size, "batch_tam": n_epochs}
    print(
        "El mejor rendimiento sobre el conjunto de validacion ha sido: {}, usando los parametros rate: {}, n_epochs: {} y batch_tam: {}".format(
            mejor_rendimiento, mejor_rendimiento_params["rate"], mejor_rendimiento_params["n_epochs"],
            mejor_rendimiento_params["batch_tam"]))

    lr = RL_OvR(clases, rate_decay=True, **mejor_rendimiento_params)
    Xev = np.concatenate((Xe, Xv))
    yev = np.concatenate((ye, yv))
    lr.entrena(Xev, yev)  # Se entrena una ultima vez con todo el conjunto de entrenamiento y validacion
    rendimiento_prueba = rendimiento(lr, Xp, yp)
    print("Lo que nos proporciona un rendimiento sobre el conjunto de test: {}".format(rendimiento_prueba))


# >>> grid_search_OvR(clases_digits, X_train_digits, X_validation_digits, X_test_digits, y_train_digits, y_validation_digits, y_test_digits)    (TARDA VARIOS MINUTOS)
# =================================
# Rendimiento con rate: 0.1, n_epochs: 50, batch_tam: 32
# 0.83
# =================================
# =================================
# Rendimiento con rate: 0.1, n_epochs: 50, batch_tam: 64
# 0.843
# =================================
# =================================
# Rendimiento con rate: 0.1, n_epochs: 50, batch_tam: 128
# 0.861
# =================================
#   ...
# =================================
# Rendimiento con rate: 0.5, n_epochs: 200, batch_tam: 256
# 0.844
# =================================
# El mejor rendimiento sobre el conjunto de validacion ha sido: 0.873, usando los parametros rate: 0.3, n_epochs: 64 y batch_tam: 150
# Lo que nos proporciona un rendimiento sobre el conjunto de test: 0.843