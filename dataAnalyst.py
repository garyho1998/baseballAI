import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.tests.frame.test_validate import dataframe
from IDtoValues import game_df

def standardize(data):
    return (data - data.mean()) / data.std(ddof=0)
def pearsons_r(x, y):
    return (standardize(x) * standardize(y)).mean()
def correlations_for_hr(df):
    columns = list(game_df)
    for x in columns:
        if x not in leave_out:
            r = pearsons_r(df['score'], df[x])
            
            # Calculating the strenth of the correlation
            correlation = ''
            if r > 0.7:
                correlation = '++'
                strong_positive_correlation.append(x)
            elif r > 0.5:
                correlation = '+ '
            elif r > 0.3:
                correlation = '+-'
            elif r >= -0.3:
                correlation = 'O '
                no_correlation.append(x)
            elif r > -0.5:
                correlation = '-+'
            elif r > -0.7:
                correlation = '- '
            elif r > -1:
                correlation = '--'
                strong_negative_correlation.append(x)
                
            print('{} Correlation between score and {}:{}'.format(correlation, x, "%.3f"%r))
            print('-----------------------------------------')
            
# Reading the batting data
folderpath = "//OAFILE06/gcyho$/Desktop/AIData/level3/2016game.xlsx"
#,'BOS','CHW','CLE','DET','HOU','KCR','LAA','MIN','NYY','OAK','SEA','TBR','TEX','TOR'
game_df = pd.read_excel(folderpath)

    
strong_positive_correlation = []
strong_negative_correlation = []
no_correlation = []
leave_out = ['team','date', 'score']
            
print('Correlations:')
print('-----------------------------------------------------')
print(correlations_for_hr(game_df))
print('\n')
print('Positive correlations: {}'.format(strong_positive_correlation))
print('\n')
print('no correlations: {}'.format(no_correlation))
print('\n')
print('Negative correlations: {}'.format(strong_negative_correlation))