from lxml import html
from lxml import etree
from selenium import webdriver
import openpyxl
from urllib.request import urlopen
import requests
from openpyxl import Workbook
from openpyxl import load_workbook
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
from IDtoValues import batting_df, pitching_df

MissingPlayerList = open('//OAFILE06/gcyho$/Desktop/AIData/NotFoundList_pitcher.txt','r')
text = MissingPlayerList.read()
nameList = text[1:-1].split(',')
def edit(element):
   result = element.replace('.', '')
   result = result.replace('-','') 
   result = result.replace('\'','')  
   result = result.strip()   
   result = result.split(' ')
   first_name = result[0]
   last_name = result[1]
   result = last_name[:min(5,len(result[1])+1)] + first_name[:min(2,len(result[1])+1)]
   return result.lower()

def AddBattingRecord(trs, batting_dict_list, nameID):
    for tr in trs:
        year = tr.xpath(".//th")[0].text
        h = tr.xpath(".//td[@data-stat='H']")[0].text
        ab = tr.xpath(".//td[@data-stat='AB']")[0].text
        g = tr.xpath(".//td[@data-stat='G']")[0].text
        temp_dict = {'playerID':nameID, 'yearID': year, 'H': h, 'AB': ab, 'G': g}
        batting_dict_list.append(temp_dict)
    return batting_dict_list
def AddPitchingRecord(trs, pitching_dict_list, nameID):
    for tr in trs:
        year = tr.xpath(".//th")[0].text
        era = tr.xpath(".//td[@data-stat='earned_run_avg']")[0].text
        g = tr.xpath(".//td[@data-stat='G']")[0].text
        temp_dict = {'playerID':nameID, 'yearID': year, 'ERA': era, 'G': g}
        pitching_dict_list.append(temp_dict)
    return pitching_dict_list
m = map(edit, nameList)
nameIDList = list(m)

chrome_options = Options()
chrome_options.add_argument("--disable-extensions")
#chrome_options.add_argument("--headless")
try:
    driver = webdriver.Chrome(executable_path="C:/chromedriver.exe", chrome_options = chrome_options)
except:
    print("chrome error")
    
folderpath = "//OAFILE06/gcyho$/Desktop/AIData/Master.csv"
master_df = pd.read_csv(folderpath,na_values=[""])
folderpath = "//OAFILE06/gcyho$/Desktop/AIData/Batting.csv"
batting_df = pd.read_csv(folderpath,na_values=[""])
folderpath = "//OAFILE06/gcyho$/Desktop/AIData/pitching.csv"
pitching_df = pd.read_csv(folderpath,na_values=[""])
count = 0
batting_dict_list = []
pitching_dict_list = []
master_dict_list = []
for nameID in nameIDList:
    print (nameID)
    link1  = "https://www.baseball-reference.com/players/" + nameID[0] + "/" + nameID + "01.shtml"
    link2  = "https://www.baseball-reference.com/players/" + nameID[0] + "/" + nameID + "02.shtml"
    try:
        doc1 = html.parse(urlopen(link1)) 
        nameID = nameID+'01'
    except:
        print("doc1 error")
        continue
    try:
        doc2 = html.parse(urlopen(link2))
        h1_2 = doc2.xpath("//div[@id='content']/h1")
        print("doc2 exits")
        continue
    except:
        x = 1
        
    driver.get(link1)
    html_source = driver.page_source
    doc = html.fromstring(html_source)

    master_df.set_index('playerID')
    name = nameList[count]
    name = name.replace('\'','')
    name = name.strip()
    name = name.split(' ')   
    master_dict = {'playerID':nameID, 'nameFirst': name[0], 'nameLast': name[1]}
    master_dict_list.append(master_dict)
    
    trs = doc.xpath(".//table[@id='batting_standard']/tbody/tr[@class='full']")
    batting_dict_list = AddBattingRecord(trs, batting_dict_list, nameID)    
    trs = doc.xpath(".//table[@id='pitching_standard']/tbody/tr[@class='full']")
    pitching_dict_list = AddPitchingRecord(trs, pitching_dict_list, nameID)
    
    count = count + 1
    
driver.close()

print(master_dict_list)
print(pitching_dict_list)
master_df = master_df.append(master_dict_list, ignore_index=True, sort=False)
print("!")
batting_df = batting_df.append(batting_dict_list, ignore_index=True, sort=False)
pitching_df = pitching_df.append(pitching_dict_list, ignore_index=True, sort=False)
print("!!")

writer = pd.ExcelWriter("//OAFILE06/gcyho$/Desktop/AIData/Master2.xlsx")
master_df.to_excel(writer,'Sheet1')
writer.save()
writer2 = pd.ExcelWriter("//OAFILE06/gcyho$/Desktop/AIData/Batting2.xlsx")
batting_df.to_excel(writer2,'Sheet1')
writer2.save()
writer3 = pd.ExcelWriter("//OAFILE06/gcyho$/Desktop/AIData/Pitching2.xlsx")
pitching_df.to_excel(writer3,'Sheet1')
writer3.save()
MissingPlayerList.close()

print ("End")