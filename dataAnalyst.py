import numpy as np
import pandas as pd
import seaborn as sns
from path_definition import ROOT_DIR
from pandas.tests.frame.test_validate import dataframe
import matplotlib.pyplot as plt; plt.rcdefaults()

def standardize(data):
    return (data - data.mean()) / data.std(ddof=0)
def pearsons_r(x, y):
    return (standardize(x) * standardize(y)).mean()
def correlations_for_hr(df):
    columns = list(df)
    for x in columns:
        if x not in leave_out:
            r = pearsons_r(df['score'], df[x])
            feature.append(x)
            performance.append(abs(r))
            
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
            
folderpath = ROOT_DIR + "level3/2016game.xlsx"
game_df = pd.read_excel(folderpath)
pd.set_option('display.expand_frame_repr', False)
print(game_df.describe())

feature = []
performance = []
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

result = pd.DataFrame(feature, columns=["feature"]) 
result["performance"] = pd.Series(performance, index=result.index)# 
# 
result = result.groupby(["feature"])['performance'].aggregate(np.median).reset_index().sort_values('performance')
sns.barplot(x='performance', y="feature", data=result, palette=sns.color_palette("Blues_d"))
plt.show()
# 
# 
# 
# # def Calcalate_correct_prediction(df1, df2):
# #     return 0