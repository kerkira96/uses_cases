# Carga las dependencias de Spark MLlib
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
import colorsys

# Carga el conjunto de datos de clustering en 2D
sc = SparkContext(appName="KMeans")
text = sc.textFile("home\pruebacaso1.txt")
data = text.map(lambda l : l.strip().split('\t'))

# Entrenamos el modelo de clustering
clusters = KMeans.train(data, 5)

# Visualización de los clusters
def ccolor (cluster):
    rnd.seed(cluster)
    hue = rnd.random()
    val = rnd.uniform(0.3,0.7)
    fill = colorsys.hsv_to_rgb(hue, 0.5, val)
    border = colorsys.hsv_to_rgb(hue, 1, val)
    return (fill, border)

# Dibujo de la gráfica
fig, ax = plt.subplots()
fig.canvas.draw()
ax.set_xlabel('Ingresos')
ax.set_ylabel('Edad')
	
for (x, y) in data.collect():
    cluster = clusters.predict([x,y])
    color = ccolor(cluster)
    plt.scatter(x, y, s=14**2, c=color[0], edgecolors=color[1], alpha=0.1)
			
for i in range(0, len(clusters.centers)):
    (x,y) = clusters.centers[i]
    color = ccolor(i)
    plt.scatter(x, y, s=20**2, c=color[0], edgecolors=color[1], alpha=1)

plt.show()
plt.legend()	    
plt.xlabel('Edad')
plt.ylabel('Ingresos')	
