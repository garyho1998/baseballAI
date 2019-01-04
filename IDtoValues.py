import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import numpy
from numpy import NaN
from pandas.tests.frame.test_validate import dataframe
from unittest.mock import inplace
from matplotlib.testing.compare import converter
from path_definition import ROOT_DIR
from sklearn import preprocessing

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

def convert_IDToBatting(cell):                     
    if cell ==" " or cell =="" or cell=="NaN" or cell==NaN or pd.isnull(cell):
        return batting_HitRate_Mean
    else:
        if cell in batter_dict:
            return batter_dict[cell]
        else:
            if cell in batting_df.playerID.values:
                if isinstance(batting_df.loc[batting_df['playerID']==cell]['HitRate'], numpy.float64):
                    batter_dict[cell] = batting_df.loc[batting_df['playerID']==cell]['HitRate']
                    return batter_dict[cell] 
                else:                    
                    batter_dict[cell] = batting_df.loc[(batting_df['playerID']==cell) & (batting_df['yearID']==year-1)]['HitRate'].mean()
                    if(pd.isnull(batter_dict[cell])):
                        batter_dict[cell] = batting_df.loc[batting_df['playerID']==cell]['HitRate'].mean()
                        if(pd.isnull(batter_dict[cell])):
                            batter_dict[cell] = batting_HitRate_Mean 
                            return batter_dict[cell]
                        else:
                            return batter_dict[cell] 
                    else:
                        return batter_dict[cell] 
            else: 
                NotFoundList.append(cell)
                batter_dict[cell] = batting_HitRate_Mean
                return batter_dict[cell]      
        
def convert_IDToPitching(cell):       
    if cell ==" " or cell =="" or cell=="NaN" or cell==NaN or pd.isnull(cell):
        return pitching_ERA_Mean
    else:
        if cell in pitcher_dict:
            return pitcher_dict[cell]
        else:
            if cell in pitching_df.playerID.values:
                if isinstance(pitching_df.loc[pitching_df['playerID']==cell]['ERA'], numpy.float64):
                    pitcher_dict[cell] = pitching_df.loc[pitching_df['playerID']==cell]['ERA']
                    return pitcher_dict[cell] 
                else:                    
                    pitcher_dict[cell] = pitching_df.loc[(pitching_df['playerID']==cell) & (pitching_df['yearID']==year-1)]['ERA'].mean()
                    if(pd.isnull(pitcher_dict[cell])):
                        pitcher_dict[cell] = pitching_df.loc[pitching_df['playerID']==cell]['ERA'].mean()
                        if(pd.isnull(pitcher_dict[cell])):
                            pitcher_dict[cell] = pitching_ERA_Mean 
                            return pitcher_dict[cell]
                        else:
                            return pitcher_dict[cell] 
                    else:
                        return pitcher_dict[cell] 
            else: 
                NotFoundList.append(cell)
                pitcher_dict[cell] = pitching_ERA_Mean
                return pitcher_dict[cell] 
def convert_IDToBatting_Game_Number(cell):
    if cell ==" " or cell =="" or cell=="NaN" or cell==NaN or pd.isnull(cell):
        return batting_HitRate_Mean
    else:
        if cell in batter_Game_dict:
            return batter_Game_dict[cell]
        else:
            if cell in batting_df.playerID.values:
                if isinstance(batting_df.loc[batting_df['playerID']==cell]['G'], numpy.float64):
                    batter_Game_dict[cell] = batting_df.loc[batting_df['playerID']==cell]['G']
                    return batter_Game_dict[cell] 
                else:                    
                    batter_Game_dict[cell] = batting_df.loc[(batting_df['playerID']==cell) & (batting_df['yearID']==year-1)]['G'].mean()
                    if(pd.isnull(batter_Game_dict[cell])):
                        batter_Game_dict[cell] = batting_df.loc[batting_df['playerID']==cell]['G'].mean()
                        if(pd.isnull(batter_Game_dict[cell])):
                            batter_Game_dict[cell] = batting_HitRate_Mean 
                            return batter_Game_dict[cell]
                        else:
                            return batter_Game_dict[cell] 
                    else:
                        return batter_Game_dict[cell] 
            else: 
                NotFoundList.append(cell)
                batter_Game_dict[cell] = batting_HitRate_Mean
                return batter_Game_dict[cell]      
        
def convert_IDToPitching_Game_Number(cell):                         
    if cell ==" " or cell =="" or cell=="NaN" or cell==NaN or pd.isnull(cell):
        return pitching_ERA_Mean
    else:
        if cell in pitcher_Game_dict:
            return pitcher_Game_dict[cell]
        else:
            if cell in pitching_df.playerID.values:
                if isinstance(pitching_df.loc[pitching_df['playerID']==cell]['G'], numpy.float64):
                    pitcher_Game_dict[cell] = pitching_df.loc[pitching_df['playerID']==cell]['G']
                    return pitcher_Game_dict[cell] 
                else:                    
                    pitcher_Game_dict[cell] = pitching_df.loc[(pitching_df['playerID']==cell) & (pitching_df['yearID']==year-1)]['G'].mean()
                    if(pd.isnull(pitcher_Game_dict[cell])):
                        pitcher_Game_dict[cell] = pitching_df.loc[pitching_df['playerID']==cell]['G'].mean()
                        if(pd.isnull(pitcher_Game_dict[cell])):
                            pitcher_Game_dict[cell] = pitching_ERA_Mean 
                            return pitcher_Game_dict[cell]
                        else:
                            return pitcher_Game_dict[cell] 
                    else:
                        return pitcher_Game_dict[cell] 
            else: 
                NotFoundList.append(cell)
                pitcher_Game_dict[cell] = pitching_ERA_Mean
                return pitcher_Game_dict[cell]     
            
def apply_none(cell):
    return cell;
def standardize(data):
    return (data - data.mean()) / data.std(ddof=0)
pd.options.mode.chained_assignment = None   
year = 2016     
folderpath = ROOT_DIR + "data/Batting.csv"
batting_df = pd.read_csv(folderpath,na_values=[""])
batting_df = batting_df[['playerID','yearID','AB','H',"G"]]
batting_df['HitRate'] = batting_df['H'] / batting_df['AB']
batting_df['HitRate'].fillna(0, inplace=True)
batting_HitRate_Mean = batting_df['HitRate'].mean()
batting_df['HitRate'] = batting_df['HitRate'].apply(lambda x: x/batting_df['HitRate'].max())

folderpath = ROOT_DIR + "data/Pitching.csv"
pitching_df = pd.read_csv(folderpath,na_values=[""])
pitching_df = pitching_df[['playerID','yearID','ERA',"G"]]
pitching_df['ERA'] = pitching_df['ERA'].apply(lambda x: x/pitching_df['ERA'].max())
pitching_ERA_Mean = pitching_df['ERA'].mean()

folderpath = ROOT_DIR + "level2/" + str(year) +"game.xlsx"
game_df = pd.read_excel(folderpath,na_values=[""])
columns = list(game_df)
leave_out = ['team','date', 'score',"num_of_batter","num_of_pitcher"]
for x in columns:
    if x not in leave_out:
        game_df[x+"_Game"] = game_df[x]

pitcher_dict = {}
batter_dict = {}
batter_Game_dict = {}
pitcher_Game_dict = {}
NotFoundList = []
 
converter_dict = {'pitcher1':convert_IDToPitching, 'pitcher2':convert_IDToPitching, 
                  'pitcher1_Game':convert_IDToPitching_Game_Number, 'pitcher2_Game':convert_IDToPitching_Game_Number,
                  "num_of_pitcher": apply_none, "num_of_batter": apply_none}
for i in range(1,13):
    converter_dict['batter'+ str(i)]=convert_IDToBatting
for i in range(1,13):
    converter_dict['batter'+ str(i)+"_Game"]=convert_IDToBatting_Game_Number
print(list(game_df))
game_df.iloc[:,3:] = game_df.apply(converter_dict)
game_df.rename(columns={"batter1":"batter1_HitRate",
    "batter2":"batter2_HitRate",
    "batter3":"batter3_HitRate",
    "batter4":"batter4_HitRate",
    "batter5":"batter5_HitRate",
    "batter6":"batter6_HitRate",
    "batter7":"batter7_HitRate",
    "batter8":"batter8_HitRate",
    "batter9":"batter9_HitRate",
    "batter10":"batter10_HitRate",
    "batter11":"batter11_HitRate",
    "batter12":"batter12_HitRate",
    'pitcher1':"pitcher1_ERA",
    'pitcher2':"pitcher2_ERA",}, inplace=True)

game_df.iloc[:,2:] = (game_df.iloc[:,2:] - game_df.iloc[:,2:].min())/(game_df.iloc[:,2:].std())

# NotFoundList = f5(NotFoundList)
# NotFoundList.sort(key=None, reverse=False)
# f = open(ROOT_DIR + "IDToValueLog.txt", "w")
# print("len of not found list", end=' ')
# print(len(NotFoundList))
# f.write(str(NotFoundList))
# print(game_df.describe())
 
writer = pd.ExcelWriter(ROOT_DIR + "level3/" + str(year) +"game.xlsx")  
game_df.to_excel(writer,'Sheet1')
writer.save()
 
end = time.time()
print((end-start)/60)





