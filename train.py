from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.contrib.learn.python.learn.models import linear_regression
from path_definition import ROOT_DIR
from tensorflow.contrib import predictor

def preprocess_features(game_df):
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
     "batter12",
     "num_of_batter",
     "num_of_pitcher"
     ]]
  selected_features['pitcher1'] = (game_df['pitcher1'] * 10)
  selected_features['pitcher2'] = (game_df['pitcher2'] * 10)
  processed_features = selected_features.copy()
  return processed_features
def preprocess_targets(game_df):
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["score"] = (game_df["score"] / 100)
  return output_targets

def construct_feature_columns(input_features):
  return set( [tf.feature_column.numeric_column(my_feature)for my_feature in input_features] )

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
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

def train_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )
  
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["score"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["score"], 
                                                  num_epochs=1, 
                                                  shuffle=True)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["score"], 
                                                    num_epochs=1, 
                                                    shuffle=True)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.6f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  plt.show()
  return dnn_regressor

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.4f}'.format
pd.options.mode.chained_assignment = None
game_df = pd.read_excel(ROOT_DIR + "level3/2016game.xlsx")

game_df = game_df.reindex(
    np.random.permutation(game_df.index))

training_examples = preprocess_features(game_df.head(15000))
training_targets = preprocess_targets(game_df.head(15000))
validation_examples = preprocess_features(game_df.tail(2000))
validation_targets = preprocess_targets(game_df.tail(2000))

dnn_regressor = train_model(
      learning_rate=0.001,
      steps=12000,
      batch_size=5,
      hidden_units=[20,20,20],
      feature_columns=construct_feature_columns(training_examples),
      training_examples=training_examples,
      training_targets=training_targets,
      validation_examples=validation_examples,
      validation_targets=validation_targets)

feature_columns = construct_feature_columns(training_examples)
export_dir_base = ROOT_DIR + "model"

feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
print (feature_spec)
export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec);
servable_model_path = dnn_regressor.export_savedmodel(export_dir_base, export_input_fn, as_text=True);

random = game_df.sample()
print(random)
examples = preprocess_features(random)
targets = preprocess_targets(random)

predict_input_fn = tf.estimator.inputs.pandas_input_fn(x=examples,
                                                    y=targets,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_epochs = 1)

score_predict = dnn_regressor.predict(input_fn=predict_input_fn)
print(list(score_predict))

