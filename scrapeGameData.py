from lxml import html
from selenium import webdriver
import openpyxl
from urllib.request import urlopen
import requests
from openpyxl import Workbook
from openpyxl import load_workbook
from selenium.webdriver.chrome.options import Options
import time

def getTeambatting(doc, TeamName, data):
    trs = doc.xpath(".//table[@id='"+TeamNameDict[TeamName]+"batting']/tbody/tr")
    count = 0
    for tr in trs:
        if( len(tr.values())>0):
            if tr.values()[0] == 'spacer' or tr.values()[0] == 'thead':
                continue
        if count == 12:
            break
        player = tr.xpath(".//th/a")[0].text
        data.append(player)
        count = count + 1
    if count < 12:
        for i in range (count, 12):
            data.append(" ")

def getTeampatching(doc, TeamName, data):
    trs = doc.xpath(".//table[@id='"+TeamNameDict[TeamName]+"pitching']/tbody/tr")
    count = 0
    for tr in trs:
        if( len(tr.values())>0):
            if tr.values()[0] == 'spacer' or tr.values()[0] == 'thead':
                continue
        if count == 2:
            break
        player = tr.xpath(".//th/a")[0].text
        data.append(player)
        count = count + 1
    if count < 2:
        for i in range (count, 2):
            data.append(" ")

def getHTML(wb, link):
    sheet=wb.active;
    chrome_options = Options()
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--headless")
    try:
        driver = webdriver.Chrome(executable_path="C:/Users/garyHo/Desktop/AI/baseball/chromedriver.exe", chrome_options = chrome_options)
    except:
        print("chrome error")
        return 0
    driver.get(link)
    html_source = driver.page_source
    doc = html.fromstring(html_source)
    date = link[48:57]
    scorebox_div = doc.xpath("//div[@class='scorebox']/div")

    Team1 = scorebox_div[0].xpath(".//div/strong/a")
    Team1score = scorebox_div[0].xpath(".//div[@class='score']")[0].text
    TeamName1_short = Team1[0].values()[0][7:10]
    TeamName1_long = Team1[0].text.replace(" ", "")
    TeamName1_long = TeamName1_long.replace(".", "")
    TeamNameDict[TeamName1_short] = TeamName1_long

    Team2 = scorebox_div[1].xpath(".//div/strong/a")
    Team2score = scorebox_div[1].xpath(".//div[@class='score']")[0].text
    TeamName2_short = Team2[0].values()[0][7:10]
    TeamName2_long = Team2[0].text.replace(" ", "")
    TeamName2_long = TeamName2_long.replace(".", "")
    TeamNameDict[TeamName2_short] = TeamName2_long

    T1data = []
    T1data.append(TeamName1_short)
    T1data.append(date)
    T1data.append(Team1score)
    getTeambatting(doc, TeamName1_short, T1data)
    getTeampatching(doc, TeamName2_short, T1data)
    sheet.append(T1data)

    T2data = []
    T2data.append(TeamName2_short)
    T2data.append(date)
    T2data.append(Team2score)
    getTeambatting(doc, TeamName2_short, T2data)
    getTeampatching(doc, TeamName1_short, T2data)
    sheet.append(T2data)

    driver.close()

year = "2017"
link  = "https://www.baseball-reference.com/leagues/MLB/" + year + "-schedule.shtml"
doc = html.parse(urlopen(link))
p_list = doc.xpath("//p[@class='game']")
TeamNameDict = {"":""}
wb=Workbook()
sheet=wb.active
cols = ['team','date','score']
for i in range(1,13):
    cols.append('batter' + str(i))
cols.append('pitcher1')
cols.append('pitcher2')
sheet.append(cols)
totalTime = 0
count = 0
for p in p_list:
    if count < 1200:
        count = count+1
        continue
    elif (count == 1300):
        break
    else:
        start = time.time()
        count = count+1
        print(count)

    path = p.xpath(".//em/a")[0].values()[0]
    Tlink  = "https://www.baseball-reference.com" + path
    getHTML(wb, Tlink)
    end = time.time()
    print (end - start)
    totalTime = totalTime + end - start
wb.save("C:/Users/garyHo/Desktop/AI/baseball/bgame/" + year + "game022.xlsx")
print ("total time: " + str(totalTime/60) + "mins")





 









