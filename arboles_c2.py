# coding: utf-8

# Carga las dependencias de Spark MLlib
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark import SparkContext

# Carga el conjunto de datos 
sc = SparkContext(appName="trees2")
text = sc.textFile("home\credito2.data")
data = (text.map(lambda l : l.split('\t'))
            .map(lambda v : [ int(x.replace("A", "")) for x in v ])
            .map(lambda (a1, a2, a3, a4, a5, a6, a7, a8, c) : (a1-11, a2, a3-30, a4-40 if a4 != 410 else 10, a5-61, a6-71, a7, a8, c-1))
            .map(lambda v : LabeledPoint(v[-1], v[:-1])))


# Divide los datos en un conjunto de entrenamiento y test (70% - 30% respectivamente)
(trainData, testData) = data.randomSplit([0.7, 0.3])

# Entrena el modelo con el árbol de decisión.
model = DecisionTree.trainClassifier(
            trainData, numClasses=2, categoricalFeaturesInfo={0:4, 2:5, 3:11, 4:5, 5:5},
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
