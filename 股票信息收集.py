# -*- coding: utf-8 -*-

import urllib.request
import re
import os
import datetime
import requests
import pymysql
import pandas as pd
from bs4 import BeautifulSoup

#采集网页信息
def get_html(url):
    try:
        response=requests.get(url)
        response.raise_for_status
        response.encoding=response.apparent_encoding
        return response.text
    except:
        return "wrong"
#获取股票代码列表
def html_to_list(html,stock_num=200):
    codeList=[]
    soup=BeautifulSoup(html,"html.parser")
    a=soup.select(".stockTable>a")
    for i in a[:stock_num]:
        try:
            # if(re.findall(r"银行",i.text)):              #爬取包含关键字的股票代码
            href=i.attrs['href']
            # print(href)
            codeList.append(re.findall("[S][HZ]\d{6}",href)[0][2:])
        except:
            continue
    return codeList
# 爬取股票历史数据
def crawl_data(stock_num=200,db_path = 'info/'):
    stock_CodeUrl = 'https://hq.gucheng.com/gpdmylb.html'
    time_range = "&start=20061130&end=" + datetime.datetime.now().strftime('%Y%m%d')
    data_format = time_range + '&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
    html = get_html(stock_CodeUrl)
    codeList = html_to_list(html,stock_num)
    print(codeList)
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    for code in codeList:
        print('\r正在获取{}股票数据...'.format(code), end='\n')
        if code[0] == '6':
            url = 'http://quotes.money.163.com/service/chddata.html?code=0' + code + data_format
        else:
            url = 'http://quotes.money.163.com/service/chddata.html?code=1' + code + data_format

        file_name = db_path + code + '.csv'
        if not os.path.exists(file_name):
            urllib.request.urlretrieve(url, file_name)
    print("数据爬取完毕，存储于{}".format(db_path))
    return codeList
def save_to_sql(filepath):
    name = 'root'
    password = '111111'       # 数据库名称和密码
    # 建立本地数据库连接(需要先开启数据库服务)
    db = pymysql.connect('localhost', name, password, charset='utf8')
    cursor = db.cursor()
    # 创建数据库stockDataBase
    try:
        sqlSentence1 = "create database stockDataBase"
        cursor.execute(sqlSentence1)  # 选择使用当前数据库
    except:
        print("数据库已存在")
    sqlSentence2 = "use stockDataBase;"
    cursor.execute(sqlSentence2)
    # 获取本地文件列表
    fileList = os.listdir(filepath)
    # 依次对每个数据文件进行存储
    for fileName in fileList:
        try:
            data = pd.read_csv(filepath + fileName, encoding="gbk")
            # 创建数据表，如果数据表已经存在，会跳过继续执行下面的步骤print('创建数据表stock_%s'% fileName[0:6])
            sqlSentence3 = "create table stock_%s" % fileName[0:6] + "(日期 date, 股票代码 VARCHAR(10),     名称 VARCHAR(10),\
                               收盘价 float,    最高价    float, 最低价 float, 开盘价 float, 前收盘 float, 涨跌额    float, \
                               涨跌幅 float, 换手率 float, 成交量 bigint, 成交金额 bigint, 总市值 bigint, 流通市值 bigint)"
            cursor.execute(sqlSentence3)
        except:
            print('数据表已存在！')
        print('正在存储stock_%s' % fileName[0:6])
        length = len(data)
        for i in range(0, length):
            record = tuple(data.loc[i])
        # 插入数据语句
            try:
                sqlSentence4 = "insert into stock_%s" % fileName[0:6] + "(日期, 股票代码, 名称, 收盘价, 最高价, 最低价, 开盘价, 前收盘, 涨跌额, 涨跌幅, 换手率, \
                    成交量, 成交金额, 总市值, 流通市值) values ('%s',%s','%s',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)" % record
                # 获取的表中数据很乱，包含缺失值、Nnone、none等，插入数据库需要处理成空值
                sqlSentence4 = sqlSentence4.replace('nan', 'null').replace('None', 'null').replace('none', 'null')
                cursor.execute(sqlSentence4)
            except:
                continue
    # 关闭游标，提交，关闭数据库连接
    cursor.close()
    db.commit()
    db.close()


def main():
    db_path='info/'
    # codeList=crawl_data(db_path=db_path)
    save_to_sql(db_path)

if __name__=="__main__":
    main()