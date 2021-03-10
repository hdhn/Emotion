import sys
import os
import re
import requests,csv
from pyquery import PyQuery as pq
import time,random

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
sys.path.append("..")

header_pinlun = {
    'Cookie': 'cy=3; cye=hangzhou; _lxsdk_cuid=1753f165f9ec8-0cb558ba6b0e3a-6b111b7e-144000-1753f165f9fbe; _lxsdk=1753f165f9ec8-0cb558ba6b0e3a-6b111b7e-144000-1753f165f9fbe; _hc.v=9030c7fe-0b57-e105-ffa5-45d14dae3ce0.1603081232; ll=7fd06e815b796be3df069dec7836c3df; ua=%E5%8B%BF%E5%BF%98%E5%BF%83%E5%AE%89_9104; ctu=7fc965fa839279cab50ca6c42a9981023b8d10bde71e72095032b366399a6fe8; uamo=13247877023; s_ViewType=10; dper=a692dd00b59f6bb61dbee6cbd356f84ca9759ce61fd38cec1c34dd4af2833e9de75f06443554a7cc84eff48a876de8a9e6df65cf464542041203347a9d37996e57321a8b184aaf0e81b2930341fc1aa1d339073a49843ee04abd69fdff5db521; fspop=test; Hm_lvt_602b80cf8079ae6591966cc70a3940e7=1603081232,1603081243,1605159680; _lx_utm=utm_source%3Dwww.sogou%26utm_medium%3Dorganic; dplet=7a43f825689c976c0d67bccb6df28234; Hm_lpvt_602b80cf8079ae6591966cc70a3940e7=1605159883; _lxsdk_s=175baf83d92-b4a-5d8-2d%7C%7C212',
    'Host': 'www.dianping.com',
    'Accept-Encoding': 'gzip',
    'Referer': 'http://www.dianping.com/shop/l8uNm5kgLP4n6eLq/review_all',
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36'
}

header_css = {
    'Host': 's3plus.meituan.net',
    'Accept-Encoding': 'gzip',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36'

}


# 0-详情页
def get_msg(shopid,page):
    """
    url: http://www.dianping.com/shop/+ 商铺ID +/review_all
    :return:
    """
    # url = "http://www.dianping.com/shop/110620927/review_all"
    url = "http://www.dianping.com/shop/{shopid}/review_all/p{page}".format(shopid = shopid,page = page)
    # url = "https://www.dianping.com/shop/77307732/review_all"
    html = requests.get(url, headers=header_pinlun)
    print("1 ===> STATUS", html.status_code)
    doc = pq(html.text)
    try:
        print(doc("head >link")[3].get('href'))
    except:
        print(url)
        return 0
    # 解析每条评论
    pinglunLi = doc("div.reviews-items > ul > li").items()
    """
    调用评论里的css样式处理和加密字体svg处理
    :return:
    dict_svg_text: svg整个加密字库，以字典形式返回
    list_svg_y：svg背景中的<path>标签里的[x,y]坐标轴，以[x,y]形式返回
    dict_css_x_y：css样式中，每个加密字体的<span> 标签内容，用于匹配dict_svg_text 中的key，以字典形式返回
    """
    dict_svg_text, list_svg_y, dict_css_x_y = css_get(doc)

    for data in pinglunLi:
        # 用户名
        userName = data("div.main-review > div.dper-info > a").text()
        # 用户ID链接
        try:
            userID = "http://www.dianping.com" + data("div.main-review > div.dper-info > a").attr("href")
        except:
            userID = "匿名用户"
        # 用户评分星级[10-50]
        try:
            startShop = str(data("div.review-rank > span").attr("class")).split(" ")[1].replace("sml-str", "")
        except:
            startShop = '0'   #用户评分不存在
        # 用户描述：机器：非常好 环境：非常好 服务：非常好 人均：0元
        describeShop = data("div.review-rank > span.score").text()
        # 关键部分，评论HTML,待处理，评论包含隐藏部分和直接展示部分，默认从隐藏部分获取数据，没有则取默认部分。（查看更多）
        pinglun = data("div.review-words.Hide").html()
        try:
            len(pinglun)
        except:
            pinglun = data("div.review-words").html()
        # 该用户喜欢的美食
        loveFood = data("div.main-review > div.review-recommend").text()
        # 发表评论的时间
        pinglunTime = data("div.main-review > div.misc-info.clearfix > span.time").text()
        print("userName:", userName)
        print("userID:", userID)
        print("startShop:", startShop)
        print("describeShop:", describeShop)
        print("loveFood:", loveFood)
        print("pinglunTime:", pinglunTime)
        print("pinglun:", css_decode(dict_css_x_y, dict_svg_text, list_svg_y, pinglun))
        print("*" * 100)
        pinluncontent = css_decode(dict_css_x_y, dict_svg_text, list_svg_y, pinglun)
        out = open('./获取店铺评论2.csv','a',newline='',encoding='utf-8')
        # 设定写入模式
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([userName,userID,startShop,describeShop,loveFood,pinglunTime,pinluncontent])
        out.close()
        print("successful insert csv!")
    return 1

# 1-评论隐含部分字体css样式, 获取svg链接，获取加密汉字background
def css_get(doc):
    css_link = "http:" + doc("head >link")[3].get('href')#doc("head > link:nth-child(11)").attr("href")
    background_link = requests.get(css_link, headers=header_css)
    r = r'svgmtsi.*?background-image: url(.*?);'
    matchObj = re.compile(r)
    svg_link = matchObj.findall(background_link.text)[0].replace(")", "").replace("(", "http:")
    print(svg_link)
    """
    svg_text() 方法：请求svg字库，并抓取加密字
    dict_svg_text: svg整个加密字库，以字典形式返回
    list_svg_y：svg背景中的<path>标签里的[x,y]坐标轴，以[x,y]形式返回
    """
    dict_avg_text, list_svg_y = svg_text(svg_link)
    """
    css_dict() 方法：生成css样式中background的样式库
    dict_css: 返回css字典样式
    """
    dict_css = css_dict(background_link.text)
    return dict_avg_text, list_svg_y, dict_css


# 2-字体库链接
def svg_text(url):
    html = requests.get(url)
    dict_svg, list_y = svg_dict(html.text)
    print(dict_svg, list_y)
    return dict_svg, list_y


# 3-生成svg字库字典
def svg_dict(csv_html):
    svg_text_r = r'<textPath xlink:href="(.*?)" textLength="(.*?)">(.*?)</textPath>'
    svg_text_re = re.findall(svg_text_r, csv_html)
    print(svg_text_re)
    dict_avg = {}
    # 生成svg加密字体库字典
    for data in svg_text_re:
        dict_avg[data[0].replace("#", "")] = list(data[2])
    """
    重点：http://s3plus.meituan.net/v1/mss_0a06a471f9514fc79c981b5466f56b91/svgtextcss/74d63812e5b327d850ab4a8782833d47.svg
        svg <path> 标签里内容对应css样式中background的y轴参数，小于关系，
        如果css样式中的background的y参数小于 svg_y_re 集合中最小的数，则向上取y轴，('18', 'M0', '748', 'H600')，
        如.gqi4j {background: -98.0px -745.0px;} 中的y-745，取正数745，小于748，则对应加密字库实际y轴为748，对应的18就是<textPath>中的x轴
    """
    svg_y_r = r'<path id="(.*?)" d="(.*?) (.*?) (.*?)"/>'
    svg_y_re = re.findall(svg_y_r, csv_html)
    list_y = []
    # 存储('18', 'M0', '748', 'H600') eg:(x坐标，未知，y坐标，未知)
    for data in svg_y_re:
        list_y.append([data[0], data[2]])
    return dict_avg, list_y


# 4-生成css字库字典
def css_dict(html):
    css_text_r = r'.(.*?){background:(.*?)px (.*?)px;}'
    css_text_re = re.findall(css_text_r, html)
    dict_css = {}
    for data in css_text_re:
        """
        加密字库.gqi4j {background: -98.0px -745.0px;}与svg文件对应关系，x/14，就是svg文件加密字体下标
        y，原样返回，需要在svg函数中做处理
        """
        x = int(float(data[1]) / -14)
        """
        字典参数：{css参数名：(background-x,background-y,background-x/14,background-y)}
        """
        dict_css[data[0]] = (data[1], data[2], x, data[2])
    return dict_css


# 5-最终评论汇总
def css_decode(css_html, svg_dict, svg_list, pinglun_html):
    """
    :param css_html: css 的HTML源码
    :param svg_dict: svg加密字库的字典
    :param svg_list: svg加密字库对应的坐标数组[x, y]
    :param pinglun_html: 评论的HTML源码，对应0-详情页的评论，在此处理
    :return: 最终合成的评论
    """
    css_dict_text = css_html
    csv_dict_text, csv_dict_list = svg_dict, svg_list
    # 处理评论源码中的span标签，生成字典key
    pinglun_text = pinglun_html.replace('<svgmtsi class="', ',').replace('"/>', ",").replace('">', ",")
    pinglun_list = [x for x in pinglun_text.split(",") if x != '']
    pinglun_str = []
    for msg in pinglun_list:
        # 如果有加密标签
        if msg in css_dict_text:
            # 参数说明：[x,y] css样式中background 的[x/14，y]
            x = int(css_dict_text[msg][2])
            y = -float(css_dict_text[msg][3])
            # 寻找background的y轴比svg<path>标签里的y轴小的第一个值对应的坐标就是<textPath>的href值
            for g in csv_dict_list:
                if y < int(g[1]):
                    # print(g)
                    # print(csv_dict_text[g[0]][x])
                    pinglun_str.append(csv_dict_text[g[0]][x])
                    break
        # 没有加密标签
        else:
            pinglun_str.append(msg.replace("\n", ""))
    str_pinglun = ""
    for x in pinglun_str:
        str_pinglun += x
    # 处理特殊标签
    dr2 = re.compile(r'<.*?>', re.S)
    dr = re.compile(r'<img.*?alt="', re.S)
    dr3 = re.compile(r'&(.*?);', re.S)
    dd = dr.sub('', str_pinglun)
    dd2 = dr2.sub('', dd)
    str_pinglun = dr3.sub('', dd2)
    str_pinglun = str_pinglun.replace(' ', '').replace('收起评价', '')
    return str_pinglun
def ReadCSV(filename):
    datas = []
    dicts = {}
    with open(filename, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # header = next(csv_reader)        # 读取第一行每一列的标题
        for rows in csv_reader:
            datas.append(rows[0])
            dicts[rows[0]] = rows[1]
        return datas,dicts

if __name__ == '__main__':
    file_name = './获取店铺id测试.csv'
    datas,dicts  = ReadCSV(file_name)
    for data in datas:
        out = open('./获取店铺评论2.csv', 'a', newline='', encoding='utf-8')
        # 设定写入模式
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(["店铺名称", "店铺id"])
        csv_write.writerow([dicts[data], data])
        csv_write.writerow(["userName", "userID", "startShop", "describeShop", "loveFood", "pinglunTime", "pinglun"])
        out.close()
        print("店铺名称",dicts[data], "店铺id",data)
        for page in range(2,2000):
            print("正在爬取%s页。。。。。" % page)
            a = get_msg(shopid=data,page=page)
            if not a:
                b = input("请按enter键继续，按“1”终止内循环")
                if b == 1:
                    break
            time.sleep(random.randint(3,5))