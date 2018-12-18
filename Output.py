import tensorflow as tf
import pandas as pd
import numpy as np
from path_definition import ROOT_DIR
from tensorflow.python.data import Dataset
from tensorflow.contrib import predictor
from numpy.core.umath import true_divide
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
     "num_of_pitcher",
     "score"
     ]]
  selected_features['pitcher1'] = (game_df['pitcher1'] * 10)
  selected_features['pitcher2'] = (game_df['pitcher2'] * 10)
  selected_features['score'] = (game_df['score'] * 10)
  processed_features = selected_features.copy()
  return processed_features
def preprocess_targets(game_df):
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["score"] = (game_df["score"] / 100)
    return output_targets
# Choose the first 12000 (out of 17000) examples for training.
def construct_feature_columns(input_features):
    return set( [tf.feature_column.numeric_column(my_feature)for my_feature in input_features] )
def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):   
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

print('Start')
pd.options.mode.chained_assignment = None   
game_df = pd.read_excel(ROOT_DIR + "level3/2016game.xlsx")

training_examples = preprocess_features(game_df)

predict_fn = predictor.from_saved_model(ROOT_DIR + "model/1545123377")
result_df = pd.DataFrame()
# target_list = []
predict_list = []
for index, row in training_examples.iterrows():
    input_dict = {}
    for i in range (1,13):
        key = 'batter'+ str(i)
        content_tf_list = tf.train.FloatList(value=[row[key]])
        input_feature = tf.train.Feature(float_list=content_tf_list)
        input_dict[key] = input_feature
    content_tf_list = tf.train.FloatList(value=[row['pitcher1']])
    input_feature = tf.train.Feature(float_list=content_tf_list)
    input_dict['pitcher1'] = input_feature
    content_tf_list = tf.train.FloatList(value=[row['pitcher2']])
    input_feature = tf.train.Feature(float_list=content_tf_list)
    input_dict['pitcher2'] = input_feature
    content_tf_list = tf.train.FloatList(value=[row['num_of_batter']])
    input_feature = tf.train.Feature(float_list=content_tf_list)
    input_dict['num_of_batter'] = input_feature
    content_tf_list = tf.train.FloatList(value=[row['num_of_pitcher']])
    input_feature = tf.train.Feature(float_list=content_tf_list)
    input_dict['num_of_pitcher'] = input_feature
    
    features = tf.train.Features(feature=input_dict)
    tf_example = tf.train.Example(features=features)
    serialized_tf_example = tf_example.SerializeToString()
    
    input_tf_example = {'inputs' : [serialized_tf_example]}
    predictions = predict_fn(input_tf_example)
    predict_list.append(predictions['outputs'][0][0]*100)
    
#     target_list.append(row['score']*100)
#     predict_list.append(predictions['outputs'][0][0]*100)

game_df['predict'] = pd.Series(predict_list)

game_df = game_df[['score','predict']]

first = True
first_score = 0
first_predict = 0
win_loss_df = pd.DataFrame(columns=['correct'])
index = 0
for index, row in game_df.iterrows():
    if first:
        first_score = row['score']
        first_predict = row['predict']
        first = False
    else:
        if (first_score > row['score'] and first_predict < row['predict']) or (first_score < row['score'] and first_predict > row['predict']):
            win_loss_df = win_loss_df.append({'correct': False}, ignore_index=True)
        else:
            win_loss_df = win_loss_df.append({'correct': True}, ignore_index=True)
        first = True
    index = index+1
# print(win_loss_df)
win_loss_df = win_loss_df.apply(pd.value_counts)
print(win_loss_df)

