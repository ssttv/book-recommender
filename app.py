import os
import config
import flask
import stomp
import gc

import numpy as np
import pandas as pd
import traceback
import pickle

from flask import (Flask, session, g, json, Blueprint,flash, jsonify, redirect, render_template, request,
                   url_for, send_from_directory, send_file)
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from dotenv import load_dotenv

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
import pyspark.sql.functions as psf
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer, Normalizer, BucketedRandomProjectionLSH
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


# Load environment variables from the .env file with python-dotenv module
load_dotenv()

SUBMIT_ARGS = " --driver-memory 4G --executor-memory 1G --executor-cores 4 --packages mysql:mysql-connector-java:5.1.39 pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS

conf = SparkConf().set("spark.jars", "spark-redis/target/spark-redis_2.11-2.4.3-SNAPSHOT-jar-with-dependencies.jar").set("spark.executor.memory", "1g").set("spark.sql.pivotMaxValues", u'1000000')
sc = SparkContext('local[*]','example', conf=conf) 
sql_sc = SQLContext(sc)

schema = StructType(fields = [
    StructField('activity_id', IntegerType(), False),
    StructField('element_id', IntegerType(), False),
    StructField('user_id', IntegerType(), False),
    StructField('rating', IntegerType(), False),
    StructField('status', IntegerType(), False)
])

s_df_raw = sql_sc.read.format("jdbc").options(
        url="jdbc:mysql://localhost/recommender_10042020?use_unicode=1&charset=utf8mb4&binary_prefix=true&useSSL=false",
        driver="com.mysql.jdbc.Driver",
        dbtable="activity",
        user="admin",
        password="password",
        properties={"driver": 'com.mysql.jdbc.Driver'}
    ).load()

# s_df_raw.write.format("org.apache.spark.sql.redis").option("table", "activity").option("key.column", "activity_id").save()
# s_df_raw = sql_sc.read.format("org.apache.spark.sql.redis").option("table", "activity").option("key.column", "activity_id").load()
print(s_df_raw.count())
print(s_df_raw.head())
s_df_marks = s_df_raw.where(s_df_raw['rating'] != 0)
s_df_marks = s_df_marks.drop(s_df_marks['status']).drop(s_df_marks['activity_id']).drop_duplicates().fillna(0)
ratings = (s_df_marks
    .select(
        'user_id',
        'element_id',
        'rating',
    )
).cache()
(training, test) = ratings.randomSplit([0.8, 0.2])

als = ALS(rank=10, maxIter=20, regParam=0.01, 
          userCol="user_id", itemCol="element_id", ratingCol="rating",
          coldStartStrategy="drop",
          implicitPrefs=False)
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
userRecs.count()
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)
movieRecs.count()

userRecs_df = userRecs.toPandas()
movieRecs_df = movieRecs.toPandas()

schema_element = StructType(fields = [
    StructField('element_id', IntegerType(), False),
    StructField('title', StringType()),
    StructField('tags', StringType())
])

s_df_element = sql_sc.read.format("jdbc").options(
        url="jdbc:mysql://localhost/recommender_10042020?use_unicode=1&charset=utf8mb4&binary_prefix=true&useSSL=false",
        driver="com.mysql.jdbc.Driver",
        dbtable="element",
        user="admin",
        password="password",
        properties={"driver": 'com.mysql.jdbc.Driver'}
    ).load()

s_df_element = s_df_element.filter(s_df_element.tags != '')

# s_df_element.write.format("org.apache.spark.sql.redis").option("table", "element").option("key.column", "element_id").save()
# s_df_element = sql_sc.read.format("org.apache.spark.sql.redis").option("table", "element").option("key.column", "element_id").load()
tokenizer = Tokenizer(inputCol="tags", outputCol="units")
s_df_units = tokenizer.transform(s_df_element)

print(s_df_units.head())
vectorizer = CountVectorizer(inputCol="units", outputCol="raw_features").fit(s_df_units)
s_df_raw_features = vectorizer.transform(s_df_units)
print(s_df_raw_features.head())

idf_model = IDF(inputCol="raw_features", outputCol="features").fit(s_df_raw_features)
s_df_raw_rescaled = idf_model.transform(s_df_raw_features)

normalizer = Normalizer(inputCol="features", outputCol="norm")
data = normalizer.transform(s_df_raw_rescaled)
# data.write.format("org.apache.spark.sql.redis").option("table", "data").option("key.column", "element_id").save()
# data = sql_sc.read.format("org.apache.spark.sql.redis").option("table", "data").option("key.column", "element_id").load()
data.select("element_id", "title", "tags", "features").show()
print(data.count())
print('Started calculation of similarity values...')
dot_udf = psf.udf(lambda x,y: float(x.dot(y)), DoubleType())
similarities = data.alias("i").join(data.alias("j"), psf.col("i.element_id") < psf.col("j.element_id"))\
    .select(
        psf.col("i.element_id").alias("element_source_id"), 
        psf.col("j.element_id").alias("element_target_id"), 
        dot_udf("i.norm", "j.norm").alias("dot"))\
    .sort("element_source_id", "element_target_id")
print('Finished calculation of similarity values...')
# print(type(similarities))
# similarities.show(20)
print(similarities.schema)
# similarities.write.format("org.apache.spark.sql.redis").option("table", "similarities").save()
# similarities = sql_sc.read.format("org.apache.spark.sql.redis").option("table", "similarities").option("key.column", "i").load()
# print(similarities.collect().count())
# out_similarities = similarities.where(similarities.i == 85826).orderBy('dot', ascending=False)
# out_similarities.show(5)
# similarities_df = similarities.collect()
out_similarities = similarities.where(similarities.element_source_id == 86009).orderBy('dot', ascending=False)
out_ids = [row['element_target_id'] for row in out_similarities.collect()]
for out_id in out_ids[:5]:
    entry = (s_df_element.where(s_df_element.element_id == out_id))
    dot = out_similarities.where(out_similarities.element_target_id == out_id)
    if entry.head() is not None and dot is not None:
        print(entry.head().element_id, entry.head().title, dot.head().dot)

def create_app():

    # Initialize main Flask application and allow CORS
    app = Flask(__name__)
    cors = CORS(app)
    cors.init_app(app)

    # Load app config from config.py file, use config variable to point at STOMP/ActiveMQ host and ports
    app.config.from_object(os.environ['APP_SETTINGS'])

    from extensions import Session
    from api import api

    with app.app_context():
        # db.init_app(app)
        app.register_blueprint(api)
        app.user_recs = userRecs_df
        app.similarities = similarities
    print(type(app.user_recs))
    print(type(app.similarities))

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        Session.remove()

    return app

app = create_app()