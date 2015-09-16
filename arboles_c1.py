# coding: utf-8

# Carga las dependencias de Spark MLlib
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark import SparkContext

# Carga el conjunto de datos "Breast Cancer Wisconsin (Diagnostic)"
sc = SparkContext(appName="trees")
text = sc.textFile("home\credito.data")
data = (text.map(lambda l : l.split(' '))
            .map(lambda v : [ int(x.replace("A", "")) for x in v ])
            .map(lambda (a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, c) : (a1-11, a2, a3-30, a4-40 if a4 != 410 else 10, a5, a6-61, a7-71, a8, a9-91, a10-101, a11, a12-121, a13, a14-141, a15-151, a16, a17-171, a18, a19-191, a20-201, c-1))
            .map(lambda v : LabeledPoint(v[-1], v[:-1])))


# Divide los datos en un conjunto de entrenamiento y test (70% - 30% respectivamente)
(trainData, testData) = data.randomSplit([0.7, 0.3])

# Entrena el modelo con el árbol de decisión.
model = DecisionTree.trainClassifier(
            trainData, numClasses=2, categoricalFeaturesInfo={0:4, 2:5, 3:11, 5:5, 6:5, 8:5, 9:3, 11:4, 13:3, 14:3, 16:4, 18:2, 19:2},
            impurity='entropy', maxDepth=3)			
			

# Evaluamos el modelo para saber el porcentaje de aciertos.
predictions = model.predict(testData.map(lambda lp : lp.features))
results = testData.map(lambda lp : lp.label).zip(predictions)
acc = (results.filter(lambda (v, p): v == p)
              .count()) / float(testData.count())
print('% Aciertos: ' + str(acc * 100))

# También podemos calcular otras métricas
tp = results.filter(lambda (v, p): v == 1 and p == 1).count()
tn = results.filter(lambda (v, p): v == 0 and p == 0).count()
fp = results.filter(lambda (v, p): v == 0 and p == 1).count()
fn = results.filter(lambda (v, p): v == 1 and p == 0).count()
print('Matriz de confusión:\n{:d}\t{:d}\n{:d}\t{:d}'.format(tp, fp, fn, tn))
precision = float(tp) / (tp + fp)
recall = float(tp) / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print('Precision: {:f}\nRecall: {:f}\nF1: {:f}'.format(precision, recall, f1))


# Imprimimos el árbol de decisión.
print(model.toDebugString())
