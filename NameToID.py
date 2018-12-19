import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from numpy import NaN
from pandas.tests.frame.test_validate import dataframe
from unittest.mock import inplace
from matplotlib.testing.compare import converter
from path_definition import ROOT_DIR

start = time.time()
def f5(seq, idfun=None): 
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

def convert_nameToID_pitcher(cell):                  
    if cell ==" " or cell =="" or cell=="NaN" or cell==NaN:
        return NaN
    else:
        if cell in dict_df.index:
            if type(dict_df.loc[cell]['playerID']) is str:
                return dict_df.loc[cell]['playerID']
            else:       
                return dict_df.loc[cell]['playerID'].iloc[-1]
        else: 
            NotFoundList_pitcher.append(cell)
            return NaN
def convert_nameToID_batter(cell):                  
    if cell ==" " or cell =="" or cell=="NaN" or cell==NaN:
        return NaN
    else:
        if cell in dict_df.index:
            if type(dict_df.loc[cell]['playerID']) is str:
                return dict_df.loc[cell]['playerID']
            else:       
                return dict_df.loc[cell]['playerID'].iloc[-1]
        else: 
            NotFoundList_batter.append(cell)
            return NaN
        
folderpath = ROOT_DIR + "data/Master.xlsx"
master_df = pd.read_excel(folderpath,na_values=[""])

dict_df = master_df[['playerID','nameFirst','nameLast','nameGiven']]
pd.options.mode.chained_assignment = None
dict_df['name'] = dict_df[['nameFirst', 'nameLast']].apply(lambda x: ' '.join(x.map(str)), axis=1)
dict_df['name'] = dict_df['name'].apply(lambda x: str(x))
dict_df.set_index('name', inplace=True)

folderpath = ROOT_DIR + "games/2016game"

temp_dfs = []
NotFoundList_pitcher = []
NotFoundList_batter = []
for i in range(3,29):
    temp_dfs.append( pd.read_excel(folderpath + str(i) + ".xlsx") )
game_df = pd.concat(temp_dfs)
game_df = game_df.reset_index(drop=True)
print (game_df)

converter_dict = {'pitcher1':convert_nameToID_pitcher, 'pitcher2':convert_nameToID_pitcher}
for i in range(1,13):
    converter_dict['batter'+ str(i)]=convert_nameToID_batter
game_df.iloc[:,3:17] = game_df.apply(converter_dict)

game_df['num_of_batter'] = game_df.iloc[:,3:15].apply(lambda x: x.count(), axis=1)
game_df['num_of_pitcher'] = game_df.iloc[:,15:17].apply(lambda x: x.count(), axis=1)

NotFoundList_pitcher = f5(NotFoundList_pitcher)
NotFoundList_pitcher.sort(key=None, reverse=False)
f = open( ROOT_DIR + "NotFoundList_pitcher.txt", "w")
f.write(str(NotFoundList_pitcher))

NotFoundList_batter = f5(NotFoundList_batter)
NotFoundList_pitcher.sort(key=None, reverse=False)
f2 = open(ROOT_DIR + "NotFoundList_batter.txt", "w")
f2.write(str(NotFoundList_batter))

writer = pd.ExcelWriter(ROOT_DIR + "level2/2016game.xlsx")
game_df.to_excel(writer,'Sheet1')
writer.save()

end = time.time()
print((end-start)/60)
