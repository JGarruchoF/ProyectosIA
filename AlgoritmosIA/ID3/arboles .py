# ==========================================================
# Inteligencia Artificial. Tercer curso. Grupo 2.
# Grado en Ingeniería Informática - Tecnologías Informáticas
# Curso 2019-20
# Universidad de Sevilla
# Trabajo práctico
# Profesor: José Luis Ruiz Reina
# ===========================================================

# --------------------------------------------------------------------------
# Autor: 
#
# APELLIDOS:GARRUCHO FERNÁNDEZ
# NOMBRE: JAVIER
# ----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PARTE 0: Conjuntos de datos
# ---------------------------------------------------------------------------
#
# Los archivos jugar_tenis.py, lentes.py, votos.py y credito.py contienen los conjuntos de datos que
# vamos a usar para probar los algoritmos implementados.

# Cada archivo contiene la definición correspondiente de las siguientes
# variables:

# * atributos: es una lista de pares (Atributo,Valores) para cada atributo o
#   característica del conjunto de datos. Atributo es el nombre del atributo y
#   Valores es la lista de sus posibles valores.

# * atributo_clasificación: nombre del atributo de clasificación

# * clases: posibles valores (o clases) del atributo de clasificación

# * entr: conjunto de entrenamiento, una lista de ejemplos en los que cada
#   ejemplo es una lista de valores (cada valor indica el valor del atributo
#   correspondiente, en el mismo orden en el que aparecen en lalista d
#   atributos). El último valor del ejemplo es su clase.

# Además, votos.py y credito.py contienen las siguientes variables adicionales:

# * valid: conjunto de validación, una lista de ejemplos con el mismo formato
#     que la de entrenamiento. Este conjunto de ejemplo se usará para
#     generalizar el modelo aprendido con el de entrenamiento (en nuestro
#     caso, para hacer la poda). 

# * test: conjunto de test, una lista de ejemplos con el mismo formato
#     que la de entrenamiento. Este conjunto de ejemplo se usará para
#     medir el rendimiento final del clasificador aprendido. Solo debe ser
#     usado para dar el rendimiento final del árbol aprendido y ajustado. 

# Cargamos los cuatro conjuntos de datos:

import jugar_tenis
import lentes
import votos
import credito
import math
from collections import defaultdict
import sys
import copy
import operator
import random
import titanic


# ---------------------------------------------------------------------------
# PARTE 1: Árboles de decisión e ID3
# ---------------------------------------------------------------------------
# Representación de árboles de decisión:
# ======================================

# Representaremos los árboles de decisión mediante la siguiente estructura de
# datos, de manera recursiva:


class NodoDT(object):

    def __init__(self, atributo=-1, distr=None, ramas=None, clase=None):
        self.distr = distr
        self.atributo = atributo
        self.ramas = ramas
        self.clase = clase

    
# Un objeto de esta clase representará un nodo de un árbol de decisión,
# mediante cuatro campos (distr,atributo,ramas y clase) que pasamos a
# describir. Habrá dos tipos de nodos, ambos representables mediante esta
# clase: nodos HOJA, que son los que dan un valor de clasificación, y nodos
# INTERIORES, que se corresponden con un atributo y tienen un hijo por cada
# posible valor del atributo. Según el tipo de nodo, algunos campos tendrán
# valor None. En concreto: 

# * En un nodo hoja, los campos atributo y ramas valen None y el nodo clase es
#   el valor de clasificación que da el nodo.

# * En un nodo interior el campo clase es None, el campo atributo indica el
#   atributo del nodo en la lista de atributos, y ramas es un diccionario que
#   representa los distintos subárboles que son hijos del nodo. En concreto,
#   en ese diccionario las claves son los valores del atributo (la etiqueta de
#   la rama) y el valor asignado a cada clave es un objeto de la clase NodoDT
#   que a su vez representa al subárbol correspondiente a la rama de ese
#   valor.

# En el campo atributo no almacenaremos el nombre del atributo, sino un índice
# que indica la posición del atributo en la lista atributos. 

# En ambos tipos de nodo, el campo distr (que es opcional) contiene la
# distribución, según los distintos valores de clasificación, de los ejemplos
# del conjunto de entrenamiento correspondientes a ese nodo. 

# Con un ejemplo se entenderá mejor.  


# EJEMPLO
# -------

# Vamos a describir cómo se representa el árbol de "Jugar Tenis" que aparece
# en las diapositivas.

# Supongamos que la variable jt_tree contiene dicho árbol (en este caso, se
# genera como resultado de aplicar el algoritmo id3, que se pide más
# adelante):

# >>> jt_tree=id3(jugar_tenis.entr,jugar_tenis.atributos)

# El árbol es un objeto de la clase NodoCT, el nodo raíz del árbol: 

# >>> jt_tree
# <__main__.NodoDT object at 0x7f1b8d74b550>

# Es un nodo interior (en este caso, la raíz), cuyo atributo es "Cielo". En la
# variable atributos de jugar_tenis.py, "Cielo" es el primer atributo, y por
# tanto se corresponde con índice 0. En el campo atributo lo que guardamos es
# precisamente ese índice:

# >>> jt_tree.atributo
# 0

# En el campo distr están la distribución por clase de los ejemplos
# correspondientes al nodo (en este caso, como es el nodo raíz, son todos los
# ejemplos):

# >>> jt_tree.distr
# defaultdict(<class 'int'>, {'no': 5, 'si': 9})

# Nótese que la distribución se almacena con un "defauldict", un tipo de datos
# que se comporta exactamente igual que los diccionarios "dict", pero con la
# salvedad que tienen definido un valor por defecto para las claves que no
# están. En este caso concreto, creando el dicionario con "defauldict(int)", 
# tenemos un diccionario en el que se considera que una clave que no está
# explícitamente en el diccionario, tiene  asociado un valor 0. Más detalles
# sobre los "defaultdict" en el manual de referencia de python, módulo
# "collections". 


# En el campo ramas del nodo hay un diccionario, en el que hay una clave por
# cada valor del atributo "Cielo". El valor asociado a cada clave es un NodoDT
# que a su vez representa el correspondiente subárbol:

# >>> jt_tree.ramas
# {'Soleado': <__main__.NodoDT object at 0x7f1b8d783850>, 'Nublado': <__main__.NodoDT object at 0x7f1b8d783390>, 'Lluvia': <__main__.NodoDT object at 0x7f1b8d783ad0>}

# Por ejemplo, el subárbol de la rama "Cielo=Soleado", empieza en un nodo
# interior correspondiente al atributo de índice 2 ("Humedad"):

# >>> jt_tree.ramas["Soleado"].atributo
# 2

# Con "Cielo=Soleado" hay 3 negativos y 2 positivos: 

# >>> jt_tree.ramas["Soleado"].distr
# defaultdict(<class 'int'>, {'no': 3, 'si': 2})

# Y las ramas de ese nodo son:

# >>> jt_tree.ramas["Soleado"].ramas
# {'Normal': <__main__.NodoDT object at 0x7f1b8d783550>, 'Alta':<__main__.NodoDT object at 0x7f1b8d7839d0>}

# El nodo correspondiente a "Cielo=Soleado" y "Humedad=Normal" es ya una hoja
# del árbol, que clasifica con clase positiva:

# >>> jt_tree.ramas["Soleado"].ramas["Normal"].clase
# 'si'

# Y con  "Cielo=Soleado" y "Humedad=Normal" son 2 positivos y 0 negativos: 

# >>> jt_tree.ramas["Soleado"].ramas["Normal"].distr
# defaultdict(<class 'int'>, {'si': 2})

# El nodo correspondiente a "Cielo=Soleado" y "Humedad=Alta" es también una
# hoja, que clasifica como negativo, y cuya distribución son 3 negativos y 0
# positivos: 

# >>> jt_tree.ramas["Soleado"].ramas["Alta"].clase
# 'no'
# >>> jt_tree.ramas["Soleado"].ramas["Alta"].distr
# defaultdict(<class 'int'>, {'no': 3})

# A continuación los restantes nodos, que se describen análogamente:

# >>> jt_tree.ramas["Nublado"].clase
# 'si'
# >>> jt_tree.ramas["Nublado"].distr
# defaultdict(<class 'int'>, {'si': 4})


# >>> jt_tree.ramas["Lluvia"].atributo
# 3
# >>> jt_tree.ramas["Lluvia"].ramas
# {'Fuerte': <__main__.NodoDT object at 0x7f1b8d783990>, 'Débil': <__main__.NodoDT object at 0x7f1b8d783050>}
# >>> jt_tree.ramas["Lluvia"].distr
# defaultdict(<class 'int'>, {'no': 2, 'si': 3})

# >>> jt_tree.ramas["Lluvia"].ramas["Fuerte"].clase
# 'no'
# >>> jt_tree.ramas["Lluvia"].ramas["Fuerte"].distr
# defaultdict(<class 'int'>, {'no': 2})

# >>> jt_tree.ramas["Lluvia"].ramas["Débil"].clase
# 'si'
# >>> jt_tree.ramas["Lluvia"].ramas["Débil"].distr
# defaultdict(<class 'int'>, {'si': 3})


# Funciones que se piden:
# =======================

# Usando la estructura de datos descrita, se pide definir las siguientes
# cuatro funciones: 

# 1.Una función "id3(ejemplos,atributos)", que recibiendo como entrada un
#   conjunto de ejemplos y una lista de atributos (con nombres y valores, como
#   aparece en los archivos de datos) aplique el algoritmo id3 descrito en
#   clase, para obtener un árbol de decisión. 

#   Esta función tiene argumentos de entrada adicionales, que se explican más
#   abajo. 

# 2.Una función "imprime_DT(árbol_dt,atributos,atributo_clasificación)" que
#   recibiendo un árbol de decisión, la lista de los atributos del problema
#   (como aparece en el fichero de datos) y el nombre del atributo de
#   clasificación, imprime el árbol tal y como se muestra en los ejemplos que
#   se muestran más abajo.

# 3.Una función "clasifica_DT(ejemplo,árbol_dt)" que recibiendo un ejemplo
#   (sin el valor de clasificación) y un árbol de decisión, devuelve la
#   clasificación que el árbol le asigna a dicho ejemplo.  

# 4.Una función "rendimiento_DT(árbol, ejemplos)", que recibiendo una lista
#   de ejemplos (con su clasificación) y un árbol de decisión, devuelve la
#   proporción de ejemplos bien clasificados por el árbol.
# ------

# Expliquemos ahora a continuación una serie de argumentos adicionales de la
# función id3. La especificación completa de los argumentos de entrada es:

#        id3(ejemplos,
#            atributos,
#            criterio=entropía,
#            max_freq_split=1.0,
#            min_prop_ejemplos=0)

# donde:

# - ejemplos es el conjunto de entrenamiento: una lista de ejemplos, tal y
#   como se proporcionan en los ficheros de datos.   

# - atributos es una lista de atributos (con sus nombres y valores posibles),
#   tal y como aparece en los ficheros de datos. 

# - criterio: es el nombre de la función que usaremos para medir la
#   "impureza" de la distribución de ejemplos según su valor de
#   clasificación (aquí generalizamos lo visto en las diapositivas). El
#   criterio puede ser entropía (valor por defecto) o gini:

#     * La entropía es básicamente la definida en clase, pero teniendo en
#       cuenta que  podemos tener más de dos clases: si hay n valores de 
#       clasificación, la entropía de una distribución [x1,...,xn] es 
#       sumatorio_{i=1,n} -(xi/N)log(xi/N), donde N es x1+...+xn y log es el
#       logaritmo en base 2. 

#     * Análogamente, la función gini correspondiente a una distribución se 
#       se define de la siguiente manera: 
#                      1- sum_{i=1,n} (xi/N)**2

#   Ver más información sobre entropía y gini en 
#       https://en.wikipedia.org/wiki/Decision_tree_learning

#   Tanto entropía como gini miden la heterogeneidad de una distribución
#   (también llamada "impureza") y pueden usarse como criterio para decidir
#   cuál es el mejor atributo para colocar en un nodo dado del árbol de
#   decisión, de la siguiente manera. Si tenemos un conjunto de ejemplos E en
#   un Nodo, y un atributo A que sea candidato para colocar en ese Nodo, con
#   valores v1,...,vm, entonces definimos la Ganancia de A en E respecto del
#   criterio f (donde f puede ser entropía o gini), de la siguiente manera:


#      Ganancia_f(A,E)=f(E)-sum_{j=1,m}(N_vj/N)*f(E_vj)


#   donde E_vj es el subconjunto de ejemplos de E con A=v_j, N_vj es el número
#   de ejemplos de E_vj, N es el número total de ejemplos de E, y f puede ser
#   entropía o gini. Es decir, es la diferencia entre la impureza antes de la
#   partición y la impureza media después de la partición por ese atributo. 

#   En el algoritmo de aprendizaje del árbol, seleccionaremos en cada momento
#   el atributo con mayor Ganancia (y usaremos entropía o gini, dependiendo
#   del argumento "criterio"). 

# - max_freq_split es un número entre 0 y 1.0 (por defecto 1.0). Si en un
#   Nodo, la proporción de la clase más frecuante es mayor o igual que
#   max_freq_split, entonces que ese nodo sea una hoja del árbol, con valor de
#   clasificación esa clase mayoritaria. 

# - min_prop_ejemplos es un número entre 0 y 1.0 (por defecto 0). Si la
#   proporción de ejemplos en un nodo, respecto del total de ejemplos inicial,
#   es menor o igual que una hoja, hacemos que ese nodo sea una hoja, con
#   valor de clasificación la clase mayoritaria entre los ejempos de ese nodo.  


# Estos dos ultimos argumentos (max_freq_split y min_prop_ejemplos) suponen
# una manera de hacer "parada anticipada" (early stopping) en el aprendizaje
# del árbol, mediante prepoda. En la siguiente sección, también se pedirá la
# implementación de la postpoda, una manera alternativa de evitar el
# sobreajuste. 


# Algunos ejemplos: 
# -----------------

# Jugar al tenis:

# >>> jt_tree=id3(jugar_tenis.entr,jugar_tenis.atributos)

# >>> imprime_DT(jt_tree,jugar_tenis.atributos,jugar_tenis.atributo_clasificación)

# Nodo raiz (no: 5  si: 9)
#  Cielo = Soleado. (no: 3  si: 2)
#       Humedad = Alta. (no: 3)
#            Jugar Tenis: no.
#       Humedad = Normal. (si: 2)
#            Jugar Tenis: si.
#  Cielo = Nublado. (si: 4)
#       Jugar Tenis: si.
#  Cielo = Lluvia. (si: 3  no: 2)
#       Viento = Débil. (si: 3)
#            Jugar Tenis: si.
#       Viento = Fuerte. (no: 2)
#            Jugar Tenis: no.

# >>> clasifica_DT(["Soleado","Suave","Alta","Fuerte"],jt_tree)
# 'no'

# -----

# Lentes de contacto:


# >>> lc_tree=id3(lentes.entr,lentes.atributos)

# >>> imprime_DT(lc_tree,lentes.atributos,lentes.atributo_clasificación)

# Nodo raiz (Ninguna: 15  Blanda: 5  Rígida: 4)
#  Lagrima = Reducida. (Ninguna: 12)
#       Lente: Ninguna.
#  Lagrima = Normal. (Blanda: 5  Rígida: 4  Ninguna: 3)
#       Astigmatismo = +. (Rígida: 4  Ninguna: 2)
#            Diagnóstico = Miope. (Rígida: 3)
#                 Lente: Rígida.
#            Diagnóstico = Hipermétrope. (Rígida: 1  Ninguna: 2)
#                 Edad = Joven. (Rígida: 1)
#                      Lente: Rígida.
#                 Edad = Prepresbicia. (Ninguna: 1)
#                      Lente: Ninguna.
#                 Edad = Presbicia. (Ninguna: 1)
#                      Lente: Ninguna.
#       Astigmatismo = -. (Blanda: 5  Ninguna: 1)
#            Edad = Joven. (Blanda: 2)
#                 Lente: Blanda.
#            Edad = Prepresbicia. (Blanda: 2)
#                 Lente: Blanda.
#            Edad = Presbicia. (Ninguna: 1  Blanda: 1)
#                 Diagnóstico = Miope. (Ninguna: 1)
#                      Lente: Ninguna.
#                 Diagnóstico = Hipermétrope. (Blanda: 1)
#                      Lente: Blanda.

# >>> clasifica_DT(["Prepresbicia","Hipermétrope","-","Normal"],lc_tree)
# 'Blanda'

# >>> rendimiento_DT(lc_tree,lentes.entr)
# 1.0


# >>> lc_tree_2=id3(lentes.entr,lentes.atributos,criterio=gini,max_freq_split=0.75,min_prop_ejemplos=0.15)

# Nodo raiz (Ninguna: 15  Blanda: 5  Rígida: 4)
#  Lagrima = Reducida. (Ninguna: 12)
#       Lente: Ninguna.
#  Lagrima = Normal. (Blanda: 5  Rígida: 4  Ninguna: 3)
#       Astigmatismo = +. (Rígida: 4  Ninguna: 2)
#            Diagnóstico = Miope. (Rígida: 3)
#                 Lente: Rígida.
#            Diagnóstico = Hipermétrope. (Rígida: 1  Ninguna: 2)
#                 Lente: Ninguna.
#       Astigmatismo = -. (Blanda: 5  Ninguna: 1)
#            Lente: Blanda.

# >>> clasifica_DT(["Prepresbicia","Hipermétrope","-","Normal"],lc_tree_2)
# 'Blanda'

# >>> rendimiento_DT(lc_tree_2,lentes.entr)
# 0.9166666666666666

# ------

# Votos

# >>> votos_tree=id3(votos.entr,votos.atributos)

# >>> imprime_DT(votos_tree,votos.atributos,votos.atributo_clasificación)

# Nodo raiz (republicano: 107  demócrata: 172)
#  voto4 = s. (republicano: 103  demócrata: 6)
#       voto3 = s. (republicano: 10  demócrata: 4)
#            voto7 = s. (republicano: 9)
#                 Partido: republicano.
#            voto7 = n. (demócrata: 4  republicano: 1)
#                 voto2 = s. (demócrata: 3)
#                      Partido: demócrata.
#                 voto2 = n. (demócrata: 1)
#                      Partido: demócrata.
#                 voto2 = ?. (republicano: 1)
#                      Partido: republicano.
#            voto7 = ?. (Sin ejemplos)
#                 Partido: republicano.
#       voto3 = n. (republicano: 92  demócrata: 1)       
# ...
# ...
# ... (un árbol grande, no se muestra aquí completo)
# ...
# ...

# >>> rendimiento_DT(votos_tree,votos.entr)
# >>> 1.0

# >>> rendimiento_DT(votos_tree,votos.valid)
# >>> 0.9420289855072463

# >>> rendimiento_DT(votos_tree,votos.test)
# >>> 0.9195402298850575


# >>> votos_tree_2=id3(votos.entr,votos.atributos,max_freq_split=0.95,criterio=gini,min_prop_ejemplos=0.05)

# >>> imprime_DT(votos_tree_2,votos.atributos,votos.atributo_clasificación)

# Nodo raiz (republicano: 107  demócrata: 172)
#  voto4 = s. (republicano: 103  demócrata: 6)
#       voto3 = s. (republicano: 10  demócrata: 4)
#            voto7 = s. (republicano: 9)
#                 Partido: republicano.
#            voto7 = n. (demócrata: 4  republicano: 1)
#                 Partido: demócrata.
#            voto7 = ?. (Sin ejemplos)
#                 Partido: republicano.
#       voto3 = n. (republicano: 92  demócrata: 1)
#            Partido: republicano.
#       voto3 = ?. (republicano: 1  demócrata: 1)
#            Partido: republicano.
#  voto4 = n. (demócrata: 163  republicano: 2)
#       Partido: demócrata.
#  voto4 = ?. (demócrata: 3  republicano: 2)
#       Partido: demócrata.


# >>> rendimiento_DT(votos_tree_2,votos.entr)
# >>> 0.974910394265233

# >>> rendimiento_DT(votos_tree_2,votos.valid)
# >>> 0.9565217391304348

# >>> rendimiento_DT(votos_tree_2,votos.test)
# >>> 0.9195402298850575


# -------

# Crédito bancario: 

# >>> ct_tree=id3(credito.entr,credito.atributos)

# >>> imprime_DT(ct_tree,credito.atributos,credito.atributo_clasificación)

# Nodo raiz (estudiar: 116  no conceder: 107  conceder: 102)
#  Ingresos = bajos. (no conceder: 73  estudiar: 19  conceder: 11)
#       Empleo = parado. (estudiar: 2  conceder: 1  no conceder: 24)
#            Productos = ninguno. (no conceder: 8)
#                 Crédito: no conceder.
#            Productos = uno. (conceder: 1  no conceder: 10)
#                 Propiedades = ninguna. (no conceder: 3)
#                      Crédito: no conceder.
#                 Propiedades = una. (no conceder: 5)
#                      Crédito: no conceder.
#                 Propiedades = dos o más. (conceder: 1  no conceder: 2)
#                      Hijos = ninguno. (no conceder: 1)
#                           Crédito: no conceder.
#                      Hijos = uno. (no conceder: 1)
#                           Crédito: no conceder.
#                      Hijos = dos o más. (conceder: 1)
#                           Crédito: conceder.
# ...
# ...
# ... (un árbol grande, no se muestra aquí completo)
# ...
# ...

# >>> rendimiento_DT(ct_tree,credito.entr)
# 1.0

# >>> rendimiento_DT(ct_tree,credito.valid)
# 0.9197530864197531

# >>> rendimiento_DT(ct_tree,credito.test)
# 0.8650306748466258

# >>> ct_tree_2=id3(credito.entr,credito.atributos, max_freq_split=0.75,min_prop_ejemplos=0.1)

# >>> imprime_DT(ct_tree_2,credito.atributos,credito.atributo_clasificación)

# Nodo raiz (estudiar: 116  no conceder: 107  conceder: 102)
#  Ingresos = bajos. (no conceder: 73  estudiar: 19  conceder: 11)
#       Empleo = parado. (estudiar: 2  conceder: 1  no conceder: 24)
#            Crédito: no conceder.
#       Empleo = funcionario. (no conceder: 9  estudiar: 8  conceder: 9)
#            Crédito: no conceder.
#       Empleo = laboral. (no conceder: 17  estudiar: 7)
#            Crédito: no conceder.
#       Empleo = jubilado. (estudiar: 2  conceder: 1  no conceder: 23)
#            Crédito: no conceder.
#  Ingresos = medios. (conceder: 37  no conceder: 34  estudiar: 36)
#       Propiedades = ninguna. (no conceder: 23  conceder: 1  estudiar: 14)
#            Empleo = parado. (estudiar: 2  no conceder: 13)
#                 Crédito: no conceder.
#            Empleo = funcionario. (estudiar: 6)
#                 Crédito: estudiar.
#            Empleo = laboral. (conceder: 1  no conceder: 1  estudiar: 6)
#                 Crédito: estudiar.
#            Empleo = jubilado. (no conceder: 9)
#                 Crédito: no conceder.
#       Propiedades = una. (no conceder: 11  estudiar: 22  conceder: 1)
#            Productos = ninguno. (estudiar: 1  conceder: 1  no conceder: 7)
#                 Crédito: no conceder.
#            Productos = uno. (estudiar: 14)
#                 Crédito: estudiar.
#            Productos = dos o más. (no conceder: 4  estudiar: 7)
#                 Crédito: estudiar.
#       Propiedades = dos o más. (conceder: 35)
#            Crédito: conceder.
#  Ingresos = altos. (estudiar: 61  conceder: 54)
#       Empleo = parado. (conceder: 3  estudiar: 29)
#            Crédito: estudiar.
#       Empleo = funcionario. (conceder: 26)
#            Crédito: conceder.
#       Empleo = laboral. (estudiar: 3  conceder: 25)
#            Crédito: conceder.
#       Empleo = jubilado. (estudiar: 29)
#            Crédito: estudiar.

# >>> rendimiento_DT(ct_tree_2,credito.entr)
# 0.8584615384615385

# >>> rendimiento_DT(ct_tree_2,credito.valid)
# 0.9012345679012346

# >>> rendimiento_DT(ct_tree_2,credito.test)
# 0.8834355828220859

def entropía(distr):
    T = sum(distr.values())
    res = 0
    for k, v in distr.items():
        if v > 0:
            res += -(v / T) * math.log(v / T, 2)
    return res  # sum{i=1,n} -(xi/N)log(xi/N)


def gini(distr):
    T = sum(distr.values())
    res = 0
    for k, v in distr.items():
        if v > 0:
            res += (v / T) ** 2
    return 1 - res  # 1-sum_{i=1,n} (xi/N)**2


def id3(ejemplos, atributos, criterio=entropía, max_freq_split=1.0, min_prop_ejemplos=0):
    return ide3Aux(ejemplos, atributos, criterio, max_freq_split, min_prop_ejemplos, len(ejemplos), [], None)


def ide3Aux(ejemplos, atributos, criterio, max_freq_split, min_prop_ejemplos, tamEjemplosO, atributosYaTomados, clasePredominante0):
    # tamEjemplos0 es el tamano original del conjunto de entrenamiento
    # clasePredominante0 es la clase predominante en el nodo anterior
    # atributosYaTomados es una lista con los atributos ya decididos en esta rama
    distr = distribucion(ejemplos)
    proporcionEj = len(ejemplos) / tamEjemplosO
    clasePredominante, maxProp = maxProporcion(distr)
    if not maxProp:     # maxProp == None -> No hay ejemplos; luego se clasificara con la clase predominante en el nodo anterior
        res = NodoDT(-1, distr, None, clasePredominante0)
    elif proporcionEj <= min_prop_ejemplos or maxProp >= max_freq_split or (len(atributos) == len(atributosYaTomados)):
        res = NodoDT(-1, distr, None, clasePredominante)  # si se ha alcanzado condicion de parada se devuelve nodo con la clase que predomine
    else:
        A = atributoClasificador(ejemplos, atributos, criterio, atributosYaTomados)
        cpAtributosYaTomados = atributosYaTomados.copy()
        cpAtributosYaTomados.append(A)  # Anadimos el atributo usado para clasificar a la lista de ya tomados
        ejemplosPorValor = divideEjemplosPorValorAtributo(ejemplos, A)
        ramas = {}
        for valor in atributos[A][1]:   # Para cada posible valor del atributo creamos una rama cuyo nodo es la llamada recursica con, los ejemplos que tienen el valor de la rama en el atributo
            ramas[valor] = ide3Aux(ejemplosPorValor[valor], atributos, criterio, max_freq_split, min_prop_ejemplos, tamEjemplosO, cpAtributosYaTomados, clasePredominante)
        res = NodoDT(A, distr, ramas, None)

    return res


def distribucion(ejemplos):
    # Devuelve un diccionario cuyas claves son las posibles clasificaciones y los valores el numero de ejemplos que hay con dicha clasificacion
    res = defaultdict(int)
    for e in ejemplos:
        clase = e[-1]
        if clase in res:
            res[clase] = res[clase] + 1
        else:
            res[clase] = 1
    return res


def atributoClasificador(ejemplos, atributos, criterio, atributosYaTomados):
    # Devuelve el indice del atributo que mayor ganancia de informacion consigue segun el criterio dado
    maxG = 0
    valor0 = criterio(distribucion(ejemplos))
    for i in range(len(atributos)):
        if not (atributosYaTomados and (i in atributosYaTomados)):  # Si atributosYaTomados esta vacio(para evitar errores) o el atributo i no ha sido tomado ya
            posiblesValores = atributos[i][1]    # Posibles valores del atributo, por ejemplo: para el atributo 0 ('cielo') -> vs = ['Soleado','Nublado','Lluvia']       Aclaracion: atributos[0] = ('Cielo',['Soleado','Nublado','Lluvia'])
            valorExperado = 0
            for valor in posiblesValores:
                ejemplosFiltrados = list(filter(lambda e: e[i] == valor, ejemplos))
                valorExperado += len(ejemplosFiltrados)/len(ejemplos) * criterio(distribucion(ejemplosFiltrados))       # Se suma, ponderadamente, al valor esperado el valor que tendrian los ejemplos filtrados
            G = valor0 - valorExperado  # G = ganancia de informacion en caso de tomar dicho atributo
            if G >= maxG:
                maxG, maxA = G, i
    return maxA


def maxProporcion(distr):
    # Devuelve la clase predominante de una distribucion y su proporcion
    T = 0
    maxProp = 0
    clase = None
    for k, v in distr.items():
        T = T + v
        if v > maxProp:
            clase = k
            maxProp = v

    if T <= 0:
        return None, 0
    else:
        return clase, maxProp/T


def divideEjemplosPorValorAtributo(ejemplos, indice):
    # Devuelve una diccionario cuyas clasves son los valores posibles del atributo y el valor una lista de los ejemplos con dicho valor en el atributo
    res = defaultdict(lambda : [])
    for ejemplo in ejemplos:
        res[ejemplo[indice]].append(ejemplo)
    return res


def imprime_DT(árbol_dt, atributos, atributo_clasificación):
    sys.stdout.write("Nodo raiz ")
    imprime_DT_Aux(árbol_dt, atributos,atributo_clasificación, 1)


def imprime_DT_Aux(árbol_dt, atributos, atributo_clasificación, ind):
    sys.stdout.write(toString(árbol_dt.distr) + "\n")

    if árbol_dt.clase:
        indenta(ind)
        sys.stdout.write(atributo_clasificación + ": " + árbol_dt.clase + ".\n")
    elif árbol_dt.atributo > -1:
        a, vs = atributos[árbol_dt.atributo]
        for v in vs:
            indenta(ind)
            sys.stdout.write(a + " = " + v + ". ")
            imprime_DT_Aux(árbol_dt.ramas[v], atributos, atributo_clasificación, ind+1)


def indenta(n):
    if n > 0:
        sys.stdout.write("      ")
        indenta(n - 1)


def toString(diccionario):
    res = "( "
    if diccionario and diccionario.items():
        for k, v in diccionario.items():
            res += (str(k)+": "+str(v) + " ")
    else:
        res += "Sin ejemplos "

    res += ")"
    return res


def clasifica_DT(ejemplo, árbol_dt):
    if árbol_dt.atributo < 0:   # Si el atributo es <0 (-1) dicho arbol es una hoja, por lo que se devuelve su clase
        return árbol_dt.clase
    else:                       # En caso de que no sea hoja, se hace llamada recursiva con el arbol hijo tomando la rama cuyo valor toma nuestro ejemplo en este atributo
        valor = ejemplo[árbol_dt.atributo]
        return clasifica_DT(ejemplo, árbol_dt.ramas[valor])


def rendimiento_DT(árbol, ejemplos):
    numeroEjemplos = 0
    numeroAciertos = 0
    for ejemplo in ejemplos:
        numeroEjemplos += 1
        cpEjemplo = ejemplo.copy()
        valor_clasificacion = ejemplo[len(cpEjemplo)-1]
        cpEjemplo.pop(-1)
        if valor_clasificacion == clasifica_DT(cpEjemplo, árbol):
            numeroAciertos += 1
    return numeroAciertos/numeroEjemplos

# ---------------------------------------------------------------------------
# PARTE 2: Poda para reducir el error
# ---------------------------------------------------------------------------

# El sobreajuste es un fenómeno que ocurre típicamente en el aprendizaje
# supervisado, cuando el modelo aprendido se ajusta demasiado al conjunto de
# entrenamiento, sin ser lo suficientemente general para tener un buen
# rendimiento sobre datos que no son del conjunto que se ha usado para
# entrenar. 

# Una manera de evitar el sobreajuste en el aprendizaje de árboles de decisión
# es podar el árbol obtenido, tomando como base para la poda el rendimiento
# que se obtenga sobre un conjunto de ejemplos distinto de los usados para el
# entrenamiento. Esto se puede hacer durante la fase de entrenamiento,
# aplicando algún criterio de prepoda como se ha visto en el aprtado
# anterior. Otra manera de hacerlo es mediante poda a posteriori, sobre el
# árbol aprendido. 


# Para ello, en los casos en que tengamos suficientes datos,
# vamos a dividirlos en tres partes: entrenamiento, validación y test. Con los
# datos de entrenamiento aprenderemos el aŕbol mediante id3, los datos de
# validación los usaremos para realizar la poda del árbol anterior y con los
# de test daremos una medida final del rendimiento del árbol podado. Nótese
# que los datos de votos.py y credito.py vienen ya divididos en estas tres
# partes.   

# Una de las técnicas básicas de poda se realiza con el algoritmo de poda para
# reducir el error ("reduced error pruning"), descrito en las diapositivas.

# En esta parte se pide implementar dicho algoritmo en python. Para ello, las
# siguientes funciones pueden ser de utilidad:


# * La función "nodos_interiores_DT" recibe un árbol de decisión y devuelve
# una lista de caminos a los nodos interiores del árbol:


def nodos_interiores_DT_rec(árbol_dt, camino_actual, acum_nodos):
    if árbol_dt.clase is not None:
        return acum_nodos
    else:
        atr = árbol_dt.atributo
        acum_nodos.append(camino_actual)
        for valor in árbol_dt.ramas:
            acum_nodos = nodos_interiores_DT_rec(árbol_dt.ramas[valor], camino_actual + [valor], acum_nodos)
        return acum_nodos


def nodos_interiores_DT(árbol_DT):
    return nodos_interiores_DT_rec(árbol_DT, [], [])


# Ejemplos:

# >>> nodos_interiores_DT(jt_tree)
# [[], ['Lluvia'], ['Soleado']]
# >>> nodos_interiores_DT(lentes_tree)
# [[], ['Normal'], ['Normal', '+'], ['Normal', '+', 'Hipermétrope'], 
#                  ['Normal', '-'], ['Normal', '-', 'Presbicia']]

# Nótese que cada camino a un nodo interior del árbol viene definido por las
# etiquetas (valores de los atributos) de las ramas que que llevan a ese
# nodo. El camino al nodo raíz es la lista vacía. 


# * La función "poda_nodo_DT(árbol,nodo)" recibe un árbol de decisión y un
#   camino a un nodo (como los que devuelve la función anterior), y devuelve
#   una copia del árbol, en el que se ha podado el nodo, sustituyéndolo por un
#   nodo hoja con la clase mayoritaria en dicho nodo (nótese que es necesario
#   definir la función clase_más_frecuente).  

def clase_más_frecuente(distr):
    r, _ = maxProporcion(distr)
    if r:
        return r
    else:   # En caso de que maxProporcion devuelva None diremos que la clse mas frecuente es indeterminada "IND"
        return "IND"


def poda_nodo_DT(árbol, nodo):
    if not nodo:
        return NodoDT(clase=clase_más_frecuente(árbol.distr), distr=copy.copy(árbol.distr))     # anadido distr=copy.copy(árbol.distr)
    else:
        val_nodo = nodo[0]
        árbol_podado = NodoDT(atributo=árbol.atributo, distr=copy.copy(árbol.distr))
        dict_subárboles = {}
        for valor in árbol.ramas:
            if val_nodo == valor:
                dict_subárboles[valor] = poda_nodo_DT(árbol.ramas[valor], nodo[1:])
            else:
                dict_subárboles[valor] = copy.deepcopy(árbol.ramas[valor])
        árbol_podado.ramas = dict_subárboles
        return árbol_podado

# Ejemplo:
# >>> imprime_DT(poda_nodo_DT(lc_tree,['Normal','+']), lentes.atributos, lentes.atributo_clasificación)

# Nodo raiz (Ninguna: 15  Blanda: 5  Rígida: 4)
#  Lagrima = Reducida. (Ninguna: 12)
#       Lente: Ninguna.
#  Lagrima = Normal. (Blanda: 5  Rígida: 4  Ninguna: 3)
#       Astigmatismo = +. (Rígida: 4  Ninguna: 2)
#            Lente: Rígida.
#       Astigmatismo = -. (Blanda: 5  Ninguna: 1)
#            Edad = Joven. (Blanda: 2)
#                 Lente: Blanda.
#            Edad = Prepresbicia. (Blanda: 2)
#                 Lente: Blanda.
#            Edad = Presbicia. (Ninguna: 1  Blanda: 1)
#                 Diagnóstico = Miope. (Ninguna: 1)
#                      Lente: Ninguna.
#                 Diagnóstico = Hipermétrope. (Blanda: 1)
#                      Lente: Blanda.


# Función que se pide:
# ====================

# * Una función "poda_DT(árbol,ejemplos)", que recibiendo como entrada un
#   árbol de decisión y un conjunto de ejemplos, aplique la poda para reducir
#   el error que se describe en las diapositivas. 

# Ejemplos:
# >>> votos_tree=id3(votos.entr,votos.atributos)
# >>> imprime_DT(votos_tree,votos.atributos,votos.atributo_clasificación)
# >>> votos_podado=poda_DT(votos_tree,votos.valid)
# >>> imprime_DT(votos_podado,votos.atributos,votos.atributo_clasificación)
# Nodo raiz (republicano: 107  demócrata: 172)
#  voto4 = s. (republicano: 103  demócrata: 6)
#       Partido: republicano.
#  voto4 = n. (demócrata: 163  republicano: 2)
#       Partido: demócrata.
#  voto4 = ?. (demócrata: 3  republicano: 2)
#       Partido: demócrata.


# >>> ct_podado=poda_DT(ct_tree,credito.valid)

# >>> imprime_DT(ct_podado,credito.atributos,credito.atributo_clasificación)

# Nodo raiz (estudiar: 116  no conceder: 107  conceder: 102)
#  Ingresos = bajos. (no conceder: 73  estudiar: 19  conceder: 11)
#       Empleo = parado. (estudiar: 2  conceder: 1  no conceder: 24)
#            Crédito: no conceder.
#       Empleo = funcionario. (no conceder: 9  estudiar: 8  conceder: 9)
#            Propiedades = ninguna. (no conceder: 9)
#                 Crédito: no conceder.
#            Propiedades = una. (estudiar: 6)
#                 Crédito: estudiar.
#            Propiedades = dos o más. (estudiar: 2  conceder: 9)
#                 Crédito: conceder.
#       Empleo = laboral. (no conceder: 17  estudiar: 7)
#            Productos = ninguno. (no conceder: 8)
#                 Crédito: no conceder.
#            Productos = uno. (no conceder: 9)
#                 Crédito: no conceder.
#            Productos = dos o más. (estudiar: 7)
#                 Crédito: estudiar.
#       Empleo = jubilado. (estudiar: 2  conceder: 1  no conceder: 23)
#            Crédito: no conceder.
#  Ingresos = medios. (conceder: 37  no conceder: 34  estudiar: 36)
#       Propiedades = ninguna. (no conceder: 23  conceder: 1  estudiar: 14)
#            Empleo = parado. (estudiar: 2  no conceder: 13)
#                 Crédito: no conceder.
#            Empleo = funcionario. (estudiar: 6)
#                 Crédito: estudiar.
#            Empleo = laboral. (conceder: 1  no conceder: 1  estudiar: 6)
#                 Crédito: estudiar.
#            Empleo = jubilado. (no conceder: 9)
#                 Crédito: no conceder.
#       Propiedades = una. (no conceder: 11  estudiar: 22  conceder: 1)
#            Productos = ninguno. (estudiar: 1  conceder: 1  no conceder: 7)
#                 Crédito: no conceder.
#            Productos = uno. (estudiar: 14)
#                 Crédito: estudiar.
#            Productos = dos o más. (no conceder: 4  estudiar: 7)
#                 Crédito: estudiar.
#       Propiedades = dos o más. (conceder: 35)
#            Crédito: conceder.
#  Ingresos = altos. (estudiar: 61  conceder: 54)
#       Empleo = parado. (conceder: 3  estudiar: 29)
#            Crédito: estudiar.
#       Empleo = funcionario. (conceder: 26)
#            Crédito: conceder.
#       Empleo = laboral. (estudiar: 3  conceder: 25)
#            Crédito: conceder.
#       Empleo = jubilado. (estudiar: 29)
#            Crédito: estudiar.

def poda_DT(árbol, ejemplos):
    res = copy.deepcopy(árbol)
    continuar = True
    while continuar:
        mejorRendimiento = rendimiento_DT(res, ejemplos)
        mejorRendPoda = 0
        for nodo in nodos_interiores_DT(res):   # Para cada nodo del arbol se hace una poda y se comprueba su rendimiento
            arbolPodado = poda_nodo_DT(copy.deepcopy(res), nodo)
            rendPoda = rendimiento_DT(arbolPodado, ejemplos)
            if rendPoda > mejorRendPoda:    # Nos quedamos con la mejor poda
                mejorRendPoda = rendPoda
                mejorPoda = arbolPodado
        if mejorRendPoda >= mejorRendimiento:   # Si el rendimiento obtenido con la mejor poda es mejor que el que teniamos hasta ahora, se efectua la poda
            res = mejorPoda
        else:   # Cuando no se consiga mejora en ningun nodo paramos el bucle
            continuar = False
    return res


# ---------------------------------------------------------------------------
# PARTE 3: Clasificadores
# ---------------------------------------------------------------------------

# En este trabajo, por clasificador, entendemos una clase que incluye métodos
# para el entrenamiento y la clasificación, junto con otros métodos, como la
# evaluación del rendimiento. En concreto, un clasificador será una subclase
# de la siguiente clase general:

class MetodoClasificacion:
    """
    Clase base para métodos de clasificación
    """

    def __init__(self, atributo_clasificacion, clases, atributos):
        """
        Argumentos de entrada al constructor (ver jugar_tenis.py, por ejemplo)
         
        * atributo_clasificacion: nombre del atributo de clasificación 
        * clases: lista de posibles valores del atributo de clasificación.  
        * atributos: lista con pares en los que están los atributos (o
                     características)  y su lista de valores posibles.
        """

        self.atributo_clasificacion = atributo_clasificacion
        self.clases = clases
        self.atributos = atributos

    def entrena(self, entr, valid=None):
        """
        Método genérico para entrenamiento y ajuste del
        clasificador. Deberá ser definido para cada clasificador en
        particular. 
        
        Argumentos de entrada:

        * entr: ejemplos del conjunto de entrenamiento 
        * valid: ejemplos del conjunto de validación. 
                 Algunos clasificadores simples no usan conjunto de
                 validación, por lo que en esos casos se 
                 omitiría este argumento. 
        """
        pass

    def clasifica(self, ejemplo):
        """
        Método genérico para clasificación de un ejemplo, una vez entrenado el
        clasificador. Deberá ser definido para cada clasificador en
        particular.

        Si se llama a este método sin haber entrenado previamente el
        clasificador, debe devolver un excepción ClasificadorNoEntrenado
        (introducida más abajo) 
        """
        pass

    def imprime_clasificador(self):
        """
        Método genérico para imprimir por pantalla el clasificador
        obtenido. Deberá ser definido para cada clasificador en 
        particular. 

        Si se llama a este método sin haber entrenado previamente el
        clasificador, debe devolver un excepción ClasificadorNoEntrenado
        (introducida más abajo) 
        """
        pass


# Excepción que ha de devolverse se llama al método de clasificación (o al de
# impresión) antes de ser entrenado:  

class ClasificadorNoEntrenado(Exception):
    pass

# Clases que se piden:
# ====================

# * Implementar la clase ClasificadorID3, como subclase de la clase
#   MetodoClasificacion anterior. En esta clase, los métodos son:
#   - Entrenamiento: algoritmo ID3
#   - Clasificación: clasificar con el árbol obtenido con ID3
#   - Imprimir clasificador: imprimir el árbol obtenido con ID3
#   Además de los atributos de la clase genérica, se pueden incluir otros si
#   fuera necesario (por ejemplo, será necesario un atributo de la clase para
#   guardar el árbol aprendido).  

# * Implementar la clase ClasificadorID3Poda, de manera análoga a la
#   anterior, pero en el que el entrenamiento consiste en aplicar ID3, seguido
#   de una poda. ID3 se aplica con el conjunto de entrenamiento y la poda con
#   el conjunto de validación.


class ClasificadorID3(MetodoClasificacion):

    def __init__(self, atributo_clasificacion, clases, atributos):
        self.atributo_clasificacion = atributo_clasificacion
        self.clases = clases
        self.atributos = atributos
        self.arbol_dt = None

    def entrena(self, entr, valid=None, criterio=entropía, max_freq_split=1.0, min_prop_ejemplos=0):
        self.arbol_dt = id3(entr, self.atributos, criterio, max_freq_split, min_prop_ejemplos)

    def clasifica(self, ejemplo):
        if not self.arbol_dt:
            raise ClasificadorNoEntrenado
        return clasifica_DT(ejemplo, self.arbol_dt)

    def imprime_clasificador(self):
        if not self.arbol_dt:
            raise ClasificadorNoEntrenado
        imprime_DT(self.arbol_dt, self.atributos, self.atributo_clasificacion)


class ClasificadorID3Poda(MetodoClasificacion):

    def __init__(self, atributo_clasificacion, clases, atributos):
        self.atributo_clasificacion = atributo_clasificacion
        self.clases = clases
        self.atributos = atributos
        self.arbol_dt = None

    def entrena(self, entr, valid, criterio=entropía, max_freq_split=1.0, min_prop_ejemplos=0):
        self.arbol_dt = id3(entr, self.atributos, criterio, max_freq_split, min_prop_ejemplos)
        self.arbol_dt = poda_DT(self.arbol_dt, valid)

    def clasifica(self, ejemplo):
        if not self.arbol_dt:
            raise ClasificadorNoEntrenado

        return clasifica_DT(ejemplo, self.arbol_dt)

    def imprime_clasificador(self):
        if not self.arbol_dt:
            raise ClasificadorNoEntrenado

        imprime_DT(self.arbol_dt, self.atributos, self.atributo_clasificacion)


# Función que se pide:
# ====================

# * Implementar la función "rendimiento(clasificador,ejemplos)", que calcula
#   el rendimiento de un clasificador (entrenado) sobre un conjunto de
#   ejemplos cuya clasificación se conoce.  

def rendimiento(clasificador, ejemplos):
    if not clasificador.arbol_dt:
        raise ClasificadorNoEntrenado

    return rendimiento_DT(clasificador.arbol_dt, ejemplos)

# Experimentación que se pide
# ===========================

# En el caso de los datos de votos.py y de creditos.py, comparar el
# rendimiento de las distintas variantes de clasificadores (sin poda, con
# prepoda con distintos parámetros, con postpoda, con ambas,...) sobre los
# conjuntos de entrenamiento, validación y test, comentando y analizando los
# resultados obtenidos.


# Ejemplos:

# Jugar al tenis:

# >>> clasificador_id3_jt=ClasificadorID3(jugar_tenis.atributo_clasificación, jugar_tenis.clases, jugar_tenis.atributos)

# >>> clasificador_id3_jt.clasifica(['Soleado','Suave', 'Alta','Fuerte'])
#                                          #error porque no está entrenado aún
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/tmp/python3-2944D0s.py", line 1072, in clasifica
# __main__.ClasificadorNoEntrenado
# >>> clasificador_id3_jt.entrena(jugar_tenis.entr)
# >>> clasificador_id3_jt.imprime_clasificador()
#  Cielo = Soleado
#       Humedad = Normal
#            Jugar Tenis: si
#       Humedad = Alta
#            Jugar Tenis: no
#  Cielo = Lluvia
#       Viento = Fuerte
#            Jugar Tenis: no
#       Viento = Débil
#            Jugar Tenis: si
#  Cielo = Nublado
#       Jugar Tenis: si
# >>> clasificador_id3_jt.clasifica(['Soleado','Suave', 'Alta','Fuerte'])
# 'no'
# >>> rendimiento(clasificador_id3_jt,jugar_tenis.entr)
# 1.0
# 
# -----------
# Crédito bancario:
#
# >>> clasificador_id3_ct=ClasificadorID3(credito.atributo_clasificación, credito.clases, credito.atributos)
# >>> clasificador_id3_ct.entrena(credito.entr)
# >>> clasificador_id3_ct.imprime_clasificador()
# ... Omitimos la salida, árbol muy grande .....
# >>> rendimiento(clasificador_id3_ct,credito.entr)
# 1.0
# >>> rendimiento(clasificador_id3_ct,credito.valid)
# 0.9197530864197531
# >>> rendimiento(clasificador_id3_ct,credito.test)
# 0.8650306748466258

# -------

# Crédito bancario con pre y post poda:

# >>> clasificador_ctp=ClasificadorID3Poda(credito.atributo_clasificación, credito.clases,credito.atributos)

# >>> clasificador_ctp.entrena(credito.entr,credito.valid, max_freq_split=0.85, min_prop_ejemplos=0.05)

# >>> clasificador_ctp.imprime_clasificador()

# Nodo raiz (estudiar: 116  no conceder: 107  conceder: 102)
#  Ingresos = bajos. (no conceder: 73  estudiar: 19  conceder: 11)
#       Empleo = parado. (estudiar: 2  conceder: 1  no conceder: 24)
#            Crédito: no conceder.
#       Empleo = funcionario. (no conceder: 9  estudiar: 8  conceder: 9)
#            Propiedades = ninguna. (no conceder: 9)
#                 Crédito: no conceder.
#            Propiedades = una. (estudiar: 6)
#                 Crédito: estudiar.
#            Propiedades = dos o más. (estudiar: 2  conceder: 9)
#                 Crédito: conceder.
#       Empleo = laboral. (no conceder: 17  estudiar: 7)
#            Productos = ninguno. (no conceder: 8)
#                 Crédito: no conceder.
#            Productos = uno. (no conceder: 9)
#                 Crédito: no conceder.
#            Productos = dos o más. (estudiar: 7)
#                 Crédito: estudiar.
#       Empleo = jubilado. (estudiar: 2  conceder: 1  no conceder: 23)
#            Crédito: no conceder.
#  Ingresos = medios. (conceder: 37  no conceder: 34  estudiar: 36)
#       Propiedades = ninguna. (no conceder: 23  conceder: 1  estudiar: 14)
#            Empleo = parado. (estudiar: 2  no conceder: 13)
#                 Crédito: no conceder.
#            Empleo = funcionario. (estudiar: 6)
#                 Crédito: estudiar.
#            Empleo = laboral. (conceder: 1  no conceder: 1  estudiar: 6)
#                 Crédito: estudiar.
#            Empleo = jubilado. (no conceder: 9)
#                 Crédito: no conceder.
#       Propiedades = una. (no conceder: 11  estudiar: 22  conceder: 1)
#            Productos = ninguno. (estudiar: 1  conceder: 1  no conceder: 7)
#                 Crédito: no conceder.
#            Productos = uno. (estudiar: 14)
#                 Crédito: estudiar.
#            Productos = dos o más. (no conceder: 4  estudiar: 7)
#                 Crédito: estudiar.
#       Propiedades = dos o más. (conceder: 35)
#            Crédito: conceder.
#  Ingresos = altos. (estudiar: 61  conceder: 54)
#       Empleo = parado. (conceder: 3  estudiar: 29)
#            Crédito: estudiar.
#       Empleo = funcionario. (conceder: 26)
#            Crédito: conceder.
#       Empleo = laboral. (estudiar: 3  conceder: 25)
#            Crédito: conceder.
#       Empleo = jubilado. (estudiar: 29)
#            Crédito: estudiar.


# >>> rendimiento(clasificador_ctp,credito.entr)
# 0.9261538461538461

# >>> rendimiento(clasificador_ctp,credito.valid)
# 0.9753086419753086

# >>> rendimiento(clasificador_ctp,credito.test)
# 0.9815950920245399


# ---------------------------------------------------------------------------
# PARTE 4: Entendiendo la supervivencia en el hundimiento del Titanic
# ---------------------------------------------------------------------------

# En este apartado, se pide usar alguno de los clasificadores anteriores para,
# a partir de los datos sobre pasajeros del Titanic (a descargar desde la
# página del trabajo), tratar de obtener un árbol de decisión para explicar la
# supervivencia o no de un pasajero del Titanic.

# Para ello, realizar los siguientes pasos:

# - Preprocesado de los datos: los datos están "en bruto", así que hay que
#   preparar los datos para que los puedan usar los clasificadores.
# - Aprendizaje y ajuste del modelo: aplicando a los datos los entrenamientos
#   de alguno de los clasificadores del apartado anterior.
# - Evaluacion del rendimiento del clasificador. 


# Damos a continuación algunos comentarios sobre la etapa del preprocesado de
# los datos:

# - En el conjunto de datos que se proporcionan hay una serie de atributos que
#   obviamente no influyen en la supervivencia (por ejemplo, el nombre del
#   pasajero). Esto hace que haya que seleccionar como atributos las
#   características que se crean realmente relevantes. Esto se suele
#   realizar con algunas técnicas estadísticas, pero en este trabajo sólo
#   vamos a pedir elegir (eligiendo razonablemente, o probando varias
#   alternativas) TRES ATRIBUTOS que se consideren son los que mejor
#   determinan la supervivencia o no.
# - El atributo "Edad" es numérico, y nuestra implementación no trata bien los
#   atributos con valores numéricos. Existen técnicas para tratar los
#   atributos numéricos, que básicamente dividen los posibles valores a tomar
#   en intervalos, de la mejor manera posible. En nuestro caso, por
#   simplificar, lo vamos a hacer directamente con el siguiente criterio:
#   transformar el valor EDAD en un valor binario, en el que sólo anotamos si
#   el pasajero tiene 13 AÑOS O MENOS, o si tiene MÁS DE 13 AÑOS.
# - En los datos, hay algunos valores de algunos ejemplos, que aparecen como
#   NA (desconocidos). Dos técnicas muy simples para tratar valores
#   desconocidos pueden ser: sustituir NA por el valor más frecuente de entre 
#   los ejemplos de la clase, o por la media aritmética de ese valor en los
#   ejemplos de la misma clase (esta última opción sólo tiene sentido con los
#   atributos numéricos).
# - Para realizar el entrenamiento, la poda y la medida del rendimiento, se
#   necesita dividir el conjunto de datos en tres partes: entrenamiento,
#   validación y test. Hay que decidir la proporción adecuada de datos que van
#   a cada parte. También hay que procurar además que la partición sea
#   estratificada: la proporción de los ejemplos según los distintos valores
#   de los atributos debe de ser en cada parte, similar a la proporción en el
#   total de ejemplos.   


# El resultado final de esta parte debe ser:

# * Un archivo titanic.py, con un formato análogo a los archivos de datos que
#   se han proporcionado (votos.py o credito.py, por ejemplo), en el que se
#   incluye el resultado del preprocesado de los datos en bruto.  

# * Un árbol de decisión (el que mejor rendimiento obtenga finalmente), en el
#   que se explique la supervivencia o no de un pasajero en el Titanic. Se pide
#   explicar (mediante comentarios) todo el proceso realizado hasta llegar a ese
#   árbol. Incluir este árbol de decisión (como comentario al código) en el
#   fichero titanic.py
# --------------------------------------------------------------------------------


# Dado que el nombre, el ticket, el lugar donde embarcó, el origen y destino, son atributos que considero son
# irrelevantes y la habitacion y el bote son atributos muy dificiles de clasificar, ya que que existen muchas
# posibles opciones, los atributos que tomaré para determinar la supervivencia seran clase en la que viaja, edad y sexo.
def genera_datos_titanic():
    # IMPORTANTE: Funcion que genera los datos para el problema del titanic, si se ejecuta reescribe el archivo titanitc.py
    datos = open("titanic.txt", "r").read().split("\n")
    datos.pop(0)    # eliminamos primera linea que contiene los atributos
    datos = divide_ejemplos(datos)     # Dividimos cada ejemplo en una lista con sus atributos
    datos = datos_relevantes(datos)     # Nos quedamos con los atributos "clase" "edad" (clasificando en <=13, >13 o NA) y "sexo" y si ha sobrevivido
    datos = datos_sin_NA(datos)     # Eliminamos los valores desconocidos poniendo el valor mas frecuente de su clase, en este caso solo es necesario hacerlo para la edad
    dictAtributos = genera_posibles_valores_atributo(datos)     # Guardamos en un diccionario cada posible valor para cada atributo

    try:
        f = open("titanic.py", "w")
    except:
        f = open("titanic.py", "x")

    f.write("atributo_clasificacion = \"survived\" \n")
    f.write("clases = [\"0\", \"1\"] \n")
    escribe_atributos(dictAtributos, f)     # Escribir lista de atributos con sus posibles valores en el documento titanic.py
    escribe_listas(datos, f)     # Escribir listas con conjunto de entrenamiento, validacion y entrenamiento en el documento titanic.py
    f.close()


def divide_ejemplos(datos):
    datos = list(map(lambda s: s.split("\","), datos))   # Dividimos cada linea del documento para sacar los atributos
    for i in range(len(datos)):                          # Como algunos atributos como la edad no empiezan por comillas no han sido dividido, lo hacemos con un bucle
        datos[i] = list(map(lambda x: x.replace("\"", "").split(","), datos[i]))
        for j in range(len(datos[i])):
            datos[i][j] = datos[i][j][0]    # Se pierden los atributos "name","embarked" y "home.dest" pero estos atributos no nos interesan

    for d in datos:     # Eliminamos las lineas vacias
        if len(d) <= 1:
            datos.remove(d)
    return datos


def datos_relevantes(datos):
    res = []
    for ejemplo in datos:
        aux = [ejemplo[1]]  # clase
        edad = ejemplo[3]   # edad
        aux.append("NA" if edad == "NA" else "<=13" if float(edad) <= 13 else ">13")
        aux.append(ejemplo[8])     # sexo
        aux.append(ejemplo[2])      # superviviente
        res.append(aux)
    return res


def datos_sin_NA(datos):
    # Calcular valores mas frecuentes para la edad segun clasificacion
    vMasFrecSi = {"<=13": 0, ">13": 0}
    vMasFrecNo = {"<=13": 0, ">13": 0}
    for ejemplo in datos:
        if ejemplo[1] == "<=13":
            if ejemplo[3] == "1":
                vMasFrecSi["<=13"] += 1
            else:
                vMasFrecNo["<=13"] += 1

        if ejemplo[1] == ">13":
            if ejemplo[3] == "1":
                vMasFrecSi[">13"] += 1
            else:
                vMasFrecNo[">13"] += 1

    masFrecSi = max(vMasFrecSi.items(), key=operator.itemgetter(1))[0]      # valor para edad mas frecuente entre los supervivientes
    masFrecNo = max(vMasFrecNo.items(), key=operator.itemgetter(1))[0]      # valor para edad mas frecuente entre los no supervivientes

    # Dar valor al atributo edad en ejemplos con edad desconocida
    for ejemplo in datos:
        edad = ejemplo[1]
        if ejemplo[3] == "1":
            ejemplo[1] = masFrecSi if edad == "NA" else edad
        else:
            ejemplo[1] = masFrecNo if edad == "NA" else edad
    return datos


def genera_posibles_valores_atributo(datos):
    res = {"pclass": set(), "age": set(), "sex": set()}
    for ejemplo in datos:
        res["pclass"].add(ejemplo[0])
        res["age"].add(ejemplo[1])
        res["sex"].add(ejemplo[2])
    return res


def escribe_atributos(dictAtributos, f):
    f.write("atributos = [")
    for atr in dictAtributos.keys():
        f.write("(\""+atr+"\", [")
        for v in dictAtributos[atr]:
            f.write("\""+v+"\", ")
        f.write("]),\n             ")
    f.write("]\n")


def escribe_listas(datos, f):
    entrenamiento, validacion, test = divide_datos(datos)
    f.write("entr = ")
    escribe_lista(entrenamiento, f)
    f.write("valid = ")
    escribe_lista(validacion, f)
    f.write("test = ")
    escribe_lista(test, f)


def escribe_lista(datos, f):
    f.write("[")
    for ejemplo in datos[::-1]:
        escribe_ejemplo(ejemplo, f)
        f.write(",\n        ")
    escribe_ejemplo(datos[-1], f)
    f.write("]\n")


def escribe_ejemplo(ejemplo, f):
    f.write("[")
    for i in range(len(ejemplo)-1):
        f.write("\""+ejemplo[i]+"\", ")
    f.write("\""+ejemplo[-1]+"\"")
    f.write("]")


def divide_datos(datos):
    entr = []
    valid = []
    test = []
    t = len(datos)
    for _ in range(round(t*0.65)):    # 65% de los datos usados para entrenamiento
        entr.append(datos.pop(random.randrange(len(datos))))    # coger aleatoriamente ejemplos para el conjunto de entrenamiento
    for _ in range(round(t*0.15)):    # 15% de los datos usados para validacion
        valid.append(datos.pop(random.randrange(len(datos))))    # coger aleatoriamente ejemplos para el conjunto de validacion
    for _ in range(round(t*0.20)):    # 20% de los datos usados para test
        test.append(datos.pop(random.randrange(len(datos))))    # coger aleatoriamente ejemplos para el conjunto de test
    return entr, valid, test


# Para probar el clasificador de supervivients del titanic:
# >>> clasificador_titanic = ClasificadorID3(titanic.atributo_clasificacion, titanic.clases, titanic.atributos)
# >>> clasificador_titanic.entrena(titanic.entr)
# >>> clasificador_titanic.clasifica(["2nd", "<=13", "male"])
# '1'
# >>> clasificador_titanic.clasifica(["3rd", "<=13", "male"])
# '0'
# >>> clasificador_titanic.clasifica(["2nd", ">13", "female"])
# '1'
# >>> rendimiento(clasificador_titanic, titanic.test)
# 0.8295454545454546
# >>> clasificador_titanic.imprime_clasificador()
# Nodo raiz ( 0: 558 1: 296 )
#       sex = female. ( 1: 206 0: 96 )
#             pclass = 2nd. ( 0: 7 1: 62 )
#                   age = >13. ( 0: 7 1: 56 )
#                         survived: 1.
#                   age = <=13. ( 1: 6 )
#                         survived: 1.
#             pclass = 3rd. ( 1: 54 0: 84 )
#                   age = >13. ( 1: 51 0: 80 )
#                         survived: 0.
#                   age = <=13. ( 0: 4 1: 3 )
#                         survived: 0.
#             pclass = 1st. ( 1: 90 0: 5 )
#                   age = >13. ( 1: 90 0: 5 )
#                         survived: 1.
#                   age = <=13. ( Sin ejemplos )
#                         survived: 1.
#       sex = male. ( 0: 462 1: 90 )
#             age = >13. ( 0: 455 1: 75 )
#                   pclass = 2nd. ( 0: 101 1: 12 )
#                         survived: 0.
#                   pclass = 3rd. ( 0: 278 1: 31 )
#                         survived: 0.
#                   pclass = 1st. ( 0: 76 1: 32 )
#                         survived: 0.
#             age = <=13. ( 1: 15 0: 7 )
#                   pclass = 2nd. ( 1: 7 )
#                         survived: 1.
#                   pclass = 3rd. ( 1: 4 0: 7 )
#                         survived: 0.
#                   pclass = 1st. ( 1: 4 )
#                         survived: 1.


