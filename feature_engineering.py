from pyspark.sql.functions import regexp_replace, col, udf, monotonically_increasing_id
from pyspark.sql.types import IntegerType, TimestampType
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
import plotly.graph_objects as go
import plotly
import datetime

def timestamp_transform(x):
  return datetime.datetime.fromtimestamp(x).hour
#spark.udf.register("timestamp_transform", timestamp_transform)
format_timestamp_udf = udf(lambda x: timestamp_transform(x))

def cat(cat_list, df):
  # The index of string vlaues multiple columns
  print("categorized varibales function begin")
  indexers = [
    StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
    for c in cat_list
  ]
  
  # The encode of indexed vlaues multiple columns
  encoders = [OneHotEncoder(dropLast=True,inputCol=indexer.getOutputCol(),
              outputCol="{0}_encoded".format(indexer.getOutputCol())) 
      for indexer in indexers
  ]

  # Vectorizing encoded values
  assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders],outputCol="cat_features")
  #cols =[encoder.getOutputCol() for encoder in encoders]
  pipeline = Pipeline(stages=indexers+ encoders)
  model=pipeline.fit(df)
  
  transformed_cat = model.transform(df)
  transformed_cat = assembler.transform(transformed_cat).select("cat_features")
  
  
  
  return transformed_cat

def text_processing(df):
  ''' Regex Tokenizer removes the punctuation and tokenizes the text
  StopWordsRemover to remove stopwords --default list of stopwords from english library
  HashingTF counts the word frequency but with consums lesser memory as it hashes the frequency
  Word2Vec produces word embedding'''
  tok = RegexTokenizer(inputCol="title", outputCol="words", pattern="\\W") 
  remover = StopWordsRemover(inputCol="words", outputCol="filtered")
  #tok = Tokenizer(inputCol="title", outputCol="words") 
  htf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=200) 
  w2v = Word2Vec(inputCol="filtered", outputCol="w2v")
  
  
  pipeline = Pipeline(stages=[tok, remover,htf, w2v]) 
  # Fit the pipeline 
  model = pipeline.fit(df)  
  
  #choosing one feature out of HTF and word2vec
  transform_text = model.transform(df).select("w2v")
  print("text transform done")
  return transform_text

spark.conf.set( "spark.sql.crossJoin.enabled" , "true" )

def features(df, cat_cols, int_cols):
  #changing created_utc to hours of the day
  df = df.withColumn("created_utc", col("created_utc").cast("integer"))
  
  #calling timestamp_transform function
  #print(df.columns)
  df = df.withColumn('hour_of_day', format_timestamp_udf(df['created_utc']))
  df = df.drop('created_utc')
  
  #now hours of the day can be one-hot encoded as a feature, appending it in the list of categorical variables
  
  cat_cols.append("hour_of_day")  
  
  #encoding cateorical columns
  transformed_cat = cat(cat_cols, df)
  transformed_cat = transformed_cat.withColumn("id",monotonically_increasing_id())
  
  #transforming text to word embeddings
  transformed_text = text_processing(df)
  transformed_text = transformed_text.withColumn("id", monotonically_increasing_id())
  
  #integer features
  integer_features = df.select(int_cols).withColumn("id", monotonically_increasing_id())
    
  label = df.select('score').withColumn("id", monotonically_increasing_id())
  label = label.withColumn("score", col("score").cast("integer"))
  
  #combining all the variables in a dataframe
  output_df = transformed_cat.join(transformed_text,on = "id", how = "inner")
  
  output_df = output_df.join(integer_features,on = "id", how =  "inner")
  
  
  input_cols = output_df.columns
  input_cols.remove("id")
  print("input columns for VA are: ", input_cols)
  
  #scaling using StandardScaler and combining using vector assembler as an output column - feature
   
  va = VectorAssembler(inputCols = input_cols , outputCol = "features")
  output_df = va.transform(output_df)
  
  scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
  model = scaler.fit(output_df)
  output_df = model.transform(output_df).select("scaledFeatures")
  
  '''
  pipeline = Pipeline(stages =va+scaler)
  model = pipeline.fit(output_df)
  output_df = model.transform(output_df).select("scaledFeatures")'''
  output_df = output_df.withColumn("id", monotonically_increasing_id())
  output_df = output_df.join(label, on = "id", how = "inner").drop("id")
  
  return output_df

