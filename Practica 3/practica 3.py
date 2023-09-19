# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName("LogisticRegressionExample").getOrCreate()


data = spark.read.csv(" dbfs:/FileStore/practica2.csv", header=True, inferSchema=True)


categorical_cols = ['nombre_servicio', 'paymentmethod', 'billingcycle', 'state']


numeric_cols = ['meses_activo', 'total']


indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_cols]


indexers_pipeline = Pipeline(stages=indexers)
indexed_data = indexers_pipeline.fit(data).transform(data)


target_indexer = StringIndexer(inputCol='domainstatus', outputCol='domainstatus_index')
target_indexer_model = target_indexer.fit(indexed_data)
indexed_data = target_indexer_model.transform(indexed_data)


encoder = OneHotEncoder(inputCols=[col+"_index" for col in categorical_cols],
                        outputCols=[col+"_onehot" for col in categorical_cols])

encoded_data = encoder.fit(indexed_data).transform(indexed_data)

assembler = VectorAssembler(inputCols=numeric_cols + [col+"_onehot" for col in categorical_cols],
                            outputCol="features")

final_data = assembler.transform(encoded_data)


train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=123)

lr = LogisticRegression(featuresCol="features", labelCol="domainstatus_index", maxIter=10, regParam=0.3, elasticNetParam=0.8)
model = lr.fit(train_data)

predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="domainstatus_index", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")


# COMMAND ----------



# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("DecisionTreeExample").getOrCreate()

data = spark.read.csv("dbfs:/FileStore/practica2.csv", header=True, inferSchema=True)

categorical_cols = ['nombre_servicio', 'paymentmethod', 'billingcycle', 'state']

numeric_cols = ['meses_activo', 'total']

indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_cols]

indexers_pipeline = Pipeline(stages=indexers)
indexed_data = indexers_pipeline.fit(data).transform(data)

# Configurar un StringIndexer para la columna objetivo 'domainstatus'
target_indexer = StringIndexer(inputCol='domainstatus', outputCol='domainstatus_index')
target_indexer_model = target_indexer.fit(indexed_data)
indexed_data = target_indexer_model.transform(indexed_data)


encoder = OneHotEncoder(inputCols=[col+"_index" for col in categorical_cols],
                        outputCols=[col+"_onehot" for col in categorical_cols])

encoded_data = encoder.fit(indexed_data).transform(indexed_data)


assembler = VectorAssembler(inputCols=numeric_cols + [col+"_onehot" for col in categorical_cols],
                            outputCol="features")

final_data = assembler.transform(encoded_data)


train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=123)

dt = DecisionTreeClassifier(featuresCol="features", labelCol="domainstatus_index")
dt_model = dt.fit(train_data)

dt_predictions = dt_model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="domainstatus_index", metricName="accuracy")
accuracy = evaluator.evaluate(dt_predictions)
print(f"Accuracy: {accuracy}")

# Recall
evaluator = MulticlassClassificationEvaluator(labelCol="domainstatus_index", metricName="weightedRecall")
recall = evaluator.evaluate(dt_predictions)
print(f"Recall: {recall}")

# Precisión
evaluator = MulticlassClassificationEvaluator(labelCol="domainstatus_index", metricName="weightedPrecision")
precision = evaluator.evaluate(dt_predictions)
print(f"Precisión: {precision}")

# F1-Score
evaluator = MulticlassClassificationEvaluator(labelCol="domainstatus_index", metricName="weightedFMeasure")
f1_score = evaluator.evaluate(dt_predictions)
print(f"F1-Score: {f1_score}")


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Crear una sesión de Spark
spark = SparkSession.builder.appName("RandomForestExample").getOrCreate()

data = spark.read.csv("dbfs:/FileStore/practica2.csv", header=True, inferSchema=True)

categorical_cols = ['nombre_servicio', 'paymentmethod', 'billingcycle', 'state']

numeric_cols = ['meses_activo', 'total']

indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_cols]

indexers_pipeline = Pipeline(stages=indexers)
indexed_data = indexers_pipeline.fit(data).transform(data)

target_indexer = StringIndexer(inputCol='domainstatus', outputCol='domainstatus_index')
target_indexer_model = target_indexer.fit(indexed_data)
indexed_data = target_indexer_model.transform(indexed_data)

encoder = OneHotEncoder(inputCols=[col+"_index" for col in categorical_cols],
                        outputCols=[col+"_onehot" for col in categorical_cols])

encoded_data = encoder.fit(indexed_data).transform(indexed_data)

assembler = VectorAssembler(inputCols=numeric_cols + [col+"_onehot" for col in categorical_cols],
                            outputCol="features")

final_data = assembler.transform(encoded_data)

train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=123)

rf = RandomForestClassifier(featuresCol="features", labelCol="domainstatus_index", numTrees=10)
rf_model = rf.fit(train_data)

rf_predictions = rf_model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="domainstatus_index")

# Accuracy
accuracy = evaluator.evaluate(rf_predictions, {evaluator.metricName: "accuracy"})
print(f"Accuracy: {accuracy}")

# Recall
recall = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedRecall"})
print(f"Recall: {recall}")

# Precisión
precision = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedPrecision"})
print(f"Precisión: {precision}")

# F1-Score
f1_score = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedFMeasure"})
print(f"F1-Score: {f1_score}")


# COMMAND ----------


