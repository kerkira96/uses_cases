# Import SQLContext and data types
from __future__ import print_function
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark import SparkContext

# sc is an existing SparkContext.
sc = SparkContext(appName="sql")
sqlContext = SQLContext(sc)

# Load a text file and convert each line to a tuple.
lines = sc.textFile("home\ciudad.txt")
parts = lines.map(lambda l: l.split("\t"))
people = parts.map(lambda p: (p[0], p[1], p[2]))

# The schema is encoded in a string.
schemaString = "edad ciudad canal"

fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
schema = StructType(fields)

# Apply the schema to the RDD.
schemaPeople = sqlContext.createDataFrame(people, schema)

# Register the DataFrame as a table.
schemaPeople.registerTempTable("casoSQL")

# SQL can be run over DataFrames that have been registered as a table.
results = sqlContext.sql("SELECT ciudad, avg(edad) as media FROM casoSQL where canal = '1' group by ciudad order by media desc")

# The results of SQL queries are RDDs and support all the normal RDD operations.
ciudades = results.map(lambda p: (p[0], p[1]))
				  
for ciudad in ciudades.collect():
	print('Ciudad y edad media: ' + str(ciudad))