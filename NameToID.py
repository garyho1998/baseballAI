import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from numpy import NaN
from pandas.tests.frame.test_validate import dataframe
from unittest.mock import inplace
from matplotlib.testing.compare import converter

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
def convert_nameToID(cell):                  
    if cell ==" " or cell =="" or cell=="NaN" or cell==NaN:
        return NaN
    else:
        if cell in dict_df.index:
            if type(dict_df.loc[cell]['playerID']) is str:
                return dict_df.loc[cell]['playerID']
            else:       
                return dict_df.loc[cell]['playerID'].iloc[-1]
        else: 
            NotFoundList.append(cell)
            return NaN
        
folderpath = "//OAFILE06/gcyho$/Desktop/AIData/Master.xlsx"
master_df = pd.read_excel(folderpath,na_values=[""])

dict_df = master_df[['playerID','nameFirst','nameLast','nameGiven']]
pd.options.mode.chained_assignment = None
dict_df['name'] = dict_df[['nameFirst', 'nameLast']].apply(lambda x: ' '.join(x.map(str)), axis=1)
dict_df['name'] = dict_df['name'].apply(lambda x: str(x))
dict_df.set_index('name', inplace=True)

folderpath = "//OAFILE06/gcyho$/Desktop/AIData/2016game"

temp_dfs = []
NotFoundList = []
for i in range(3,29):
    temp_dfs.append( pd.read_excel(folderpath + str(i) + ".xlsx") )
game_df = pd.concat(temp_dfs)
game_df = game_df.reset_index(drop=True)
print (game_df)

converter_dict = {'pitcher1':convert_nameToID, 'pitcher2':convert_nameToID}
for i in range(1,13):
    converter_dict['batter'+ str(i)]=convert_nameToID
game_df.iloc[:,3:17] = game_df.apply(converter_dict)
print (game_df)
print((len(game_df) - game_df.count())/len(game_df))

NotFoundList = f5(NotFoundList)
NotFoundList.sort(key=None, reverse=False)
f = open("//OAFILE06/gcyho$/Desktop/AIData/NameToIDLog.txt", "w")
print(len(NotFoundList))
f.write(str(NotFoundList))

writer = pd.ExcelWriter("//OAFILE06/gcyho$/Desktop/AIData/level2/2016game.xlsx")
game_df.to_excel(writer,'Sheet1')
writer.save()

end = time.time()
print((end-start)/60)














