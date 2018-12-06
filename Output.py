import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.data import Dataset
def preprocess_features(game_df):
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = game_df[
    ["batter1",
     "batter2",
     "batter3",
     "batter4",
     "batter5",
     "batter6",
     "batter7",
     "batter8",
     "batter9",
     "batter10",
     "batter11",
     "batter12"
     ]]
  selected_features['pitcher1'] = (game_df['pitcher1'] * 10)
  selected_features['pitcher2'] = (game_df['pitcher2'] * 10)
  processed_features = selected_features.copy()
  # Create a synthetic feature.
#   processed_features["rooms_per_person"] = (
#     california_housing_dataframe["total_rooms"] /
#     california_housing_dataframe["population"])
  return processed_features
def preprocess_targets(game_df):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["score"] = (game_df["score"] / 100)
  return output_targets

# Choose the first 12000 (out of 17000) examples for training.
def construct_feature_columns(input_features):
  return set( [tf.feature_column.numeric_column(my_feature)for my_feature in input_features] )
def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):
    """Trains a linear regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

export_dir = '//OAFILE06/gcyho$/Desktop/AIData/model/1543916968'
dnn_regressor = tf.contrib.estimator.SavedModelEstimator(export_dir)
print('Get dnn_regressors')
pd.options.mode.chained_assignment = None   
game_df = pd.read_excel("//OAFILE06/gcyho$/Desktop/AIData/level3/2016game2.xlsx")

training_examples = preprocess_features(game_df.head(9000))
training_targets = preprocess_targets(game_df.head(9000))
validation_examples = preprocess_features(game_df.tail(1000))
validation_targets = preprocess_targets(game_df.tail(1000))

eval_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["score"], 
                                          batch_size=3)
 
 
eval_results = dnn_regressor.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_results)

print (tf.train.Example())                            
random = game_df.sample()
examples = preprocess_features(random)
targets = preprocess_targets(random)
predict_input_fn = tf.estimator.inputs.pandas_input_fn(x=examples,
                                                    y=targets,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_epochs = 1)

score_predict = dnn_regressor.predict(input_fn=predict_input_fn)

print(list(score_predict))
