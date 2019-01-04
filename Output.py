import tensorflow as tf
import pandas as pd
import numpy as np
from path_definition import ROOT_DIR
from tensorflow.python.data import Dataset
from tensorflow.contrib import predictor
from numpy.core.umath import true_divide
import seaborn as sns
import matplotlib.pyplot as plt; plt.rcdefaults()

feature_list = [
#     "batter1_HitRate",
#     "batter2_HitRate",
    "batter3_HitRate",
    "batter4_HitRate",
#     "batter5_HitRate",
#     "batter6_HitRate",
#     "batter7_HitRate",
    "batter8_HitRate",
    "batter9_HitRate",
#     "batter10_HitRate",
#     "batter11_HitRate",
#     "batter12_HitRate",
#     "batter1_Game",
#     "batter2_Game",
    "batter3_Game",
    "batter4_Game",
#     "batter5_Game",
#     "batter6_Game",
#     "batter7_Game",
    "batter8_Game",
    "batter9_Game",
#     "batter10_Game",
#     "batter11_Game",
#     "batter12_Game",
    "num_of_batter",
    "num_of_pitcher",
    'pitcher1_ERA',
    'pitcher2_ERA',
    'pitcher1_Game',
    'pitcher2_Game'           ]
def preprocess_features(game_df):
  selected_features = game_df[feature_list]
  processed_features = selected_features.copy()
  return processed_features
def preprocess_targets(game_df):
    output_targets = pd.DataFrame()
    output_targets["score"] = (game_df["score"])
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

predict_fn = predictor.from_saved_model(ROOT_DIR + "model/adag_dnn2")
result_df = pd.DataFrame()

predict_list = []
for index, row in training_examples.iterrows():
    input_dict = {}
    for x in feature_list:
        content_tf_list = tf.train.FloatList(value=[row[x]])
        input_feature = tf.train.Feature(float_list=content_tf_list)
        input_dict[x] = input_feature
    
    features = tf.train.Features(feature=input_dict)
    tf_example = tf.train.Example(features=features)
    serialized_tf_example = tf_example.SerializeToString()
    input_tf_example = {'inputs' : [serialized_tf_example]}
    predictions = predict_fn(input_tf_example)
    predict_list.append(predictions['outputs'][0][0])

print(game_df.describe())    
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
        if (first_score < row['score'] and first_predict < row['predict']) or (first_score > row['score'] and first_predict > row['predict']):
            win_loss_df = win_loss_df.append({'correct': str(True), 'score_1': first_score, 'predict_1': first_predict, 'score_2': row['score'], 'predict_2': row['predict']}, ignore_index=True)
        else:
            win_loss_df = win_loss_df.append({'correct': str(False), 'score_1': first_score, 'predict_1': first_predict, 'score_2': row['score'], 'predict_2': row['predict']}, ignore_index=True)
        first = True
    index = index+1
print(win_loss_df)    
print(win_loss_df.groupby('correct')['correct'].count() )
palette = {str(True):"#32CD32", str(False):"#DD0000"}
sns.scatterplot(x="score_1", y="predict_1", hue="correct", data=win_loss_df.sample(300),palette=palette)
plt.show()
win_loss_df['error_1'] = win_loss_df["score_1"]-win_loss_df["predict_1"]
win_loss_df['error_2'] = win_loss_df["score_2"]-win_loss_df["predict_2"]
sns.scatterplot(x="error_1", y="error_2", hue="correct", data=win_loss_df.sample(300),palette=palette)
plt.show()



