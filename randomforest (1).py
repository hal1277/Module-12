#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'export version=`python --version |awk \'{print $2}\' |awk -F"." \'{print $1$2}\'`\n\necho $version\n\nif [ $version == \'36\' ] || [ $version == \'37\' ]; then\n    echo \'Starting installation...\'\n    pip3 install pyspark==2.4.8 wget==3.2 pyspark2pmml==0.5.1 > install.log 2> install.log\n    if [ $? == 0 ]; then\n        echo \'Please <<RESTART YOUR KERNEL>> (Kernel->Restart Kernel and Clear All Outputs)\'\n    else\n        echo \'Installation failed, please check log:\'\n        cat install.log\n    fi\nelif [ $version == \'38\' ] || [ $version == \'39\' ]; then\n    pip3 install pyspark==3.1.2 wget==3.2 pyspark2pmml==0.5.1 > install.log 2> install.log\n    if [ $? == 0 ]; then\n        echo \'Please <<RESTART YOUR KERNEL>> (Kernel->Restart Kernel and Clear All Outputs)\'\n    else\n        echo \'Installation failed, please check log:\'\n        cat install.log\n    fi\nelse\n    echo \'Currently only python 3.6, 3.7 , 3.8 and 3.9 are supported, in case you need a different version please open an issue at https://github.com/IBM/claimed/issues\'\n    exit -1\nfi\n')


# In[1]:


from pyspark.sql import SparkSession
import os
import shutil
import glob
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark2pmml import PMMLBuilder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline, Model
import logging
import site
import sys
import wget
import re


# In[2]:


if sys.version[0:3] == '3.9':
    url = ('https://github.com/jpmml/jpmml-sparkml/releases/download/1.7.2/'
           'jpmml-sparkml-executable-1.7.2.jar')
    wget.download(url)
    shutil.copy('jpmml-sparkml-executable-1.7.2.jar',
                site.getsitepackages()[0] + '/pyspark/jars/')
elif sys.version[0:3] == '3.8':
    url = ('https://github.com/jpmml/jpmml-sparkml/releases/download/1.7.2/'
           'jpmml-sparkml-executable-1.7.2.jar')
    wget.download(url)
    shutil.copy('jpmml-sparkml-executable-1.7.2.jar',
                site.getsitepackages()[0] + '/pyspark/jars/')
elif sys.version[0:3] == '3.7':
    url = ('https://github.com/jpmml/jpmml-sparkml/releases/download/1.5.12/'
           'jpmml-sparkml-executable-1.5.12.jar')
    wget.download(url)
elif sys.version[0:3] == '3.6':
    url = ('https://github.com/jpmml/jpmml-sparkml/releases/download/1.5.12/'
           'jpmml-sparkml-executable-1.5.12.jar')
    wget.download(url)
else:
    raise Exception('Currently only python 3.6 , 3.7, 3,8 and 3.9 is supported, in case '
                    'you need a different version please open an issue at '
                    'https://github.com/IBM/claimed/issues')


# In[21]:


data_csv = os.environ.get('data_csv', 'data1.csv')
data_parquet = os.environ.get('data_parquet', 'data.parquet')
master = os.environ.get('master', "local[*]")
data_dir = os.environ.get('data_dir', '../../component-library/data/')


# In[22]:


data_parquet = 'data.parquet'
data_csv = 'data1.csv'


# In[23]:


skip = False
if os.path.exists(data_dir + data_csv):
    skip = True


# In[24]:


if not skip:
    sc = SparkContext.getOrCreate(SparkConf().setMaster(master))
    spark = SparkSession.builder.getOrCreate()


# In[25]:


if not skip:
    df = spark.read.parquet(data_dir + data_parquet)


# In[26]:


if not skip:
    if os.path.exists(data_dir + data_csv):
        shutil.rmtree(data_dir + data_csv)
    df.coalesce(1).write.option("header", "true").csv(data_dir + data_csv)
    file = glob.glob(data_dir + data_csv + '/part-*')
    shutil.move(file[0], data_dir + data_csv + '.tmp')
    shutil.rmtree(data_dir + data_csv)
    shutil.move(data_dir + data_csv + '.tmp', data_dir + data_csv)


# In[27]:


#import pandas as pd
#df = pd.read_csv('data1.csv')

sc = SparkContext.getOrCreate(SparkConf().setMaster(master))
spark = SparkSession.builder.getOrCreate()

dfcsv = spark.read.option("header", "true").csv(data_dir + data_csv)


# In[28]:


dfcsv.show(5)


# In[29]:


from pyspark.sql.types import DoubleType
dfcsv = dfcsv.withColumn("x", dfcsv.x.cast(DoubleType()))
dfcsv = dfcsv.withColumn("y", dfcsv.y.cast(DoubleType()))
dfcsv = dfcsv.withColumn("z", dfcsv.z.cast(DoubleType()))


# In[30]:


input_columns = dfcsv["x", "y", "z"]


# In[31]:


split_data = dfcsv.randomSplit([.6,.4],24)
train_data = split_data[0]
test_data = split_data[1]


# In[44]:


indexer = StringIndexer(inputCol="class", outputCol="label")

#vectorAssembler = VectorAssembler(inputCols=input_columns,
                                  #outputCol="features")

vectorAssembler_features = VectorAssembler(inputCols=["x", "y", "z"], outputCol="features")

normalizer = MinMaxScaler(inputCol="features", outputCol="features_norm")


# In[37]:


#vectorAssembler_features = VectorAssembler(inputCols=["x", "y", "z"], outputCol="features")


# In[45]:


rf = RandomForestClassifier(labelCol="class", featuresCol="features")


# In[46]:


pipeline_rf = Pipeline(stages=[vectorAssembler_features, rf])


# In[47]:


traindf = train_data.withColumnRenamed("prediction", "label")


# In[48]:


model_rf = pipeline_rf.fit(traindf)


# In[42]:


testdf = test_data.withColumnRenamed("prediction", "label")
predictions = model_rf.transform(testdf)
evaluatorRF = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions)

print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))
print ("Number of records: " + str(testdf.count()))


# In[ ]:




