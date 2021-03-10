import sys
import os
import re
import requests,csv
from pyquery import PyQuery as pq
import time,random
import pyautogui

headers = {
    'Cookie':'_lxsdk_cuid=1753bb608fbc8-0be3842885820b-6b111b7e-1fa400-1753bb608fcc8; _lxsdk=1753bb608fbc8-0be3842885820b-6b111b7e-1fa400-1753bb608fcc8; _hc.v=8429fd3f-f5ca-d6a9-fe0f-58cf71ca55b9.1603024588; s_ViewType=10; ua=%E5%8B%BF%E5%BF%98%E5%BF%83%E5%AE%89_9104; ctu=7fc965fa839279cab50ca6c42a998102e40f1aa3353081dc3b314009932e84da; cy=3; cye=hangzhou; fspop=test; Hm_lvt_602b80cf8079ae6591966cc70a3940e7=1606009855,1606186966,1606231686,1606354961; lgtoken=0bda71f55-3bf3-4db8-8b2d-c533b2071c86; dper=b1156a1e2c1b1fd04fe071d421724d8fdce3c369f344975ea270cd8ccaf5b241c098235725f67f3dd0cd94b8c29176a287e33e28e155c682bfa2c88c1ff136d94efb35243cf51154e8ebddaa01c3d13d5114d09f31f6350c7f7ea51726b4bd39; ll=7fd06e815b796be3df069dec7836c3df; uamo=13247877023; dplet=1e617dc369091b5e32e273b4deac3739; _lx_utm=utm_source%3Dwww.sogou%26utm_medium%3Dorganic; Hm_lpvt_602b80cf8079ae6591966cc70a3940e7=1606355071; _lxsdk_s=1760237735e-225-6cf-a1b%7C%7C71',
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36',
    'Referer': '{url}/reviews'
}
#url = 'http://www.dianping.com/member/8540198'
def get_html(url):
    response = requests.get(url =url,headers = headers)
    doc = pq(response.text)
    #print(doc)
    zhuce_shijian = doc("div.user-time>p").text()[-10:]
    try:
        print(doc("div.user-info>span").attr('class').split(' ')[1].replace('urr-rank',''))
    except:
        print('\033[33;46m请按enter键继续     \033[0m')
        # if pyautogui.pixelMatchesColor(209,947,(6,184,184)):
        pyautogui.click(215, 1059, button='left')
        time.sleep(2)
        pyautogui.click(131, 45, button='left')
        time.sleep(2)
        pyautogui.moveTo(884, 324, duration=1)
        pyautogui.dragTo(1088, 329, duration=random.randint(1, 2))
        time.sleep(2)
        pyautogui.click(277, 1063, button='left')
        time.sleep(2)
        pyautogui.click(240, 978, button='left')
        time.sleep(1)
        # pyautogui.keyDown('enter')
        # time.sleep(0.5)
        # pyautogui.keyUp('enter')
        response = requests.get(url =url,headers = headers)
        doc = pq(response.text)
    user_star = doc("div.user-info>span").attr('class').split(' ')[1].replace('urr-rank','')
    print(user_star)
    print(zhuce_shijian)
    url1 = url+'/reviews'
    headers['Referer'] = headers['Referer'].format(url = url )
    response1 = requests.get(url = url1,headers = headers)
    doc1 = pq(response1.text)
    pinglun_didian =doc1('h6').items()
    pinglun_shijian = doc1('span.col-exp').items()
    pinglunds={}
    for item1,item2 in zip(pinglun_didian,pinglun_shijian):
        pinglunds[item1.text()]=item2.text()[3:]
    print(pinglunds)
    return user_star,zhuce_shijian,pinglunds
def get_html1(url):
    headers['Referer'] = headers['Referer'].format(url = url)
    response1 = requests.get(url = url,headers = headers)
    doc1 = pq(response1.text)
    pinglun_didian =doc1('h6').items()
    pinglun_shijian = doc1('span.col-exp').items()
    pinglunds={}
    for item1,item2 in zip(pinglun_didian,pinglun_shijian):
        pinglunds[item1.text()]=item2.text()[3:]
    print(pinglunds)
    return pinglunds
def ReadCSV(filename):
    datas = []
    dicts = {}
    with open(filename, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # header = next(csv_reader)        # 读取第一行每一列的标题
        for rows in csv_reader:
            if rows[1][0:4] !='http':
                continue
            datas.append(rows[0])
            dicts[rows[0]] = rows[1]
        return datas,dicts

if __name__ =='__main__':
    datas,dicts = ReadCSV('获取店铺评论测试1.csv')
    count = 1
    for data in datas:
        print("用户名称",data, "用户id",dicts[data].split('/')[4])
        with open('./获取用户个人评论1.csv', 'a', newline='', encoding='utf-8') as out:
            csv_write = csv.writer(out, dialect='excel')
            if count ==1:
                #csv_write.writerow(["用户名称", "用户id","用户星级",'注册时间',"评论"])
                count-=1
            url = dicts[data]
            user_star,zhuceshijian,pinglunds = get_html(url)
            for i in range(2,20):
                pinglunds1 = get_html1(url+'/reviews?pg={pg}&reviewCityId=0&reviewShopType=0&c=0&shopTypeIndex=0'.format(pg = i))
                if not pinglunds1:
                    break
                if len(pinglunds1) == 1:
                    break
                pinglunds.update(pinglunds1)
                time.sleep(random.randint(1, 3))
            csv_write.writerow([data, dicts[data].split('/')[4], user_star, zhuceshijian, pinglunds])
            time.sleep(random.randint(2,4))
