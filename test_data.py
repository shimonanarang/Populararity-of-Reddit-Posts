from pyspark.sql.functions import col, udf, monotonically_increasing_id
from pyspark.sql.types import IntegerType, TimestampType, StringType, BooleanType, StructType, StructField
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import plotly.graph_objects as go
import plotly
import datetime

'''importing test data from file storage in Databricks'''
file_location = "/FileStore/tables/test_project.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
testdf = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(testdf.limit(5))

def data_cleaning(df):
  
    df = df.filter(~df.score.contains('""url"":""https:'))
    df = df.filter(df.score.isNotNull())
    print("Number of Sample after dropping Null Score values",df.count())    
    bol_col = ['brand_safe', 'can_gild', 'is_crosspostable', 'no_follow', 'over_18']
    #casting each of these columne into string to change that can be
    for col_name in bol_col:
      df = df.withColumn(col_name, col(col_name).cast("string"))
    
    return df

testData = data_cleaning(testdf)

bol_col = ['brand_safe', 'can_gild', 'is_crosspostable', 'no_follow', 'over_18']
str_col = ['subreddit_type']
cat_col = bol_col + str_col
int_col = ['num_comments']

# Commented out IPython magic to ensure Python compatibility.
# %run /Users/shimona.narang@mail.utoronto.ca/feature_engineering

testData = features(testData, cat_col, int_col)

testData.columns