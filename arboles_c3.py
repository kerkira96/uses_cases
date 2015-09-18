# coding: utf-8

# Carga las dependencias de Spark MLlib
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark import SparkContext

# Carga el conjunto de datos 
sc = SparkContext(appName="trees3")
text = sc.textFile("home\cbank.data")
data = (text.map(lambda l : l.split('\t'))
            .map(lambda v : [ int(x.replace("A", "")) for x in v ])
            .map(lambda (a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, c) : (a1, a2-1, a3-1, a4-1, a5-1, a6, a7-1, a8-1, a9-1, a10, a11-1, a12, a13, a14, a15, a16-1, c-1))
            .map(lambda v : LabeledPoint(v[-1], v[:-1])))


# Divide los datos en un conjunto de entrenamiento y test (70% - 30% respectivamente)
(trainData, testData) = data.randomSplit([0.7, 0.3])

# Entrena el modelo con el árbol de decisión.
model = DecisionTree.trainClassifier(
            trainData, numClasses=2, categoricalFeaturesInfo={1:12, 2:3, 3:4, 4:2, 6:2, 7:2, 8:3, 10:12, 15:4},
            impurity='entropy', maxDepth=3)			
			

# Evalua el modelo para saber el porcentaje de aciertos.
predictions = model.predict(testData.map(lambda lp : lp.features))

results = testData.map(lambda lp : lp.label).zip(predictions)

acc = (results.filter(lambda (v, p): v == p)
              .count()) / float(testData.count())
print('% Aciertos: ' + str(acc * 100))

# Calcula otras métricas
tp = results.filter(lambda (v, p): v == 1 and p == 1).count()
tn = results.filter(lambda (v, p): v == 0 and p == 0).count()
fp = results.filter(lambda (v, p): v == 0 and p == 1).count()
fn = results.filter(lambda (v, p): v == 1 and p == 0).count()
print('Matriz de confusión:\n{:d}\t{:d}\n{:d}\t{:d}'.format(tp, fp, fn, tn))
precision = float(tp) / (tp + fp)
recall = float(tp) / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print('Precision: {:f}\nRecall: {:f}\nF1: {:f}'.format(precision, recall, f1))


# Imprime el árbol de decisión.
print(model.toDebugString())
