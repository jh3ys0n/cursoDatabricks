# Databricks notebook source
from pyspark.sql import SparkSession

# Crear una sesión de Spark
spark = SparkSession.builder.appName("GroupByExample").getOrCreate()

# Cargar el conjunto de datos desde tu fuente (reemplaza 'nombre_del_archivo.csv' con tu archivo)
data = spark.read.csv("dbfs:/FileStore/practica2.csv", header=True, inferSchema=True)

# Lista de columnas categóricas
categorical_cols = ['nombre_servicio', 'paymentmethod', 'billingcycle', 'state']

# Realizar operación groupBy y contar para las variables categóricas
for col in categorical_cols:
    grouped_data = data.groupBy(col).count()
    grouped_data.show()


# COMMAND ----------

numeric_cols = ['meses_activo', 'total']

# Seleccionar las columnas numéricas y calcular estadísticas resumidas
summary = data.select(numeric_cols).describe()

# COMMAND ----------

summary.show()

# COMMAND ----------

from pyspark.sql.functions import var_pop, kurtosis

numeric_cols = ['meses_activo', 'total']

# Calcular la varianza poblacional y la curtosis
variance = data.select([var_pop(col).alias(f'varianza_{col}') for col in numeric_cols])
kurt = data.select([kurtosis(col).alias(f'cuartil_{col}') for col in numeric_cols])

# Mostrar las medidas de dispersión
variance.show()
kurt.show()

# COMMAND ----------


