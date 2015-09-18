# Import SQLContext and data types
from __future__ import print_function
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark import SparkContext

# Cargar  el conjunto de datos
sc = SparkContext(appName="sql")
sqlContext = SQLContext(sc)
lines = sc.textFile("home\ciudad.txt")
parts = lines.map(lambda l: l.split("\t"))
people = parts.map(lambda p: (p[0], p[1], p[2]))

# Definir la estructura de la tabla y el tipo de cada campo
schemaString = "edad ciudad canal"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
schema = StructType(fields)

# Crear el esquema de la tabla y cargar los datos
schemaPeople = sqlContext.createDataFrame(people, schema)
schemaPeople.registerTempTable("casoSQL")

# Ejecutar la consulta sobre la tabla creada "casoSQL"
results = sqlContext.sql("SELECT ciudad, avg(edad) as media FROM casoSQL where canal = '1' group by ciudad order by media desc")

# Imprimir los resultados
ciudades = results.map(lambda p: (p[0], p[1])) 				  
for ciudad in ciudades.collect():
	print('Ciudad y edad media: ' + str(ciudad))
