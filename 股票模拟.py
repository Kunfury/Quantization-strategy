import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from sklearn.externals import joblib
from 机器学习 import make_price_frame,ml_data,buy_sell_clue_2,process_wave,do_ml
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

#让plt会说中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings("ignore")
# 模拟账户
class account():


    def __init__(self, stock_money, avr_cost, available_money, date):
        self.stock_money = stock_money
        self.avr_cost = avr_cost
        self.available_money = available_money
        self.date = date
        self.num = 0
        self.price=0
        self.logs = []
        self.write_log(0)

    # 操作日志
    def write_log(self, quantity):
        log = {}
        log["市场价格"]=  self.price
        log["股票市值"] = self.stock_money
        log["平均成本"] = self.avr_cost
        log["可用金额"] = self.available_money
        log["日期"] = self.date
        log["变化量"] = quantity
        log["股票数目"] = self.num
        self.logs.append(log)

    # 操作实施，quantity 为正时，买入quantity;为负时，卖出quantity比例
    def exchange(self, quantity, price, date):
        if (quantity < 0):
            quantity = quantity * self.stock_money
            self.Exchange(quantity, price, date)
        else:
            self.Exchange(quantity, price, date)

    # 操作实施，quantity仅代表交易金额
    def Exchange(self, quantity, price, date):
        self.price=price
        self.stock_money = self.num * price
        if self.stock_money + quantity <= 0:
            quantity = self.stock_money * -1
            self.avr_cost = 0
            self.num = 0
        else:
            if (quantity > self.available_money):
                quantity = self.available_money
            if (quantity > 0):
                self.avr_cost = (self.stock_money * self.avr_cost + quantity * price) / (self.stock_money + quantity)
            self.num = self.num + quantity / price
        self.available_money = self.available_money - quantity
        self.stock_money = self.stock_money + quantity
        self.date = date
        self.write_log(quantity)

    # 更新股价波动对账户影响，涨时自动卖出
    def wave(self, price, date):
        if (self.avr_cost <= 0):
            self.write_log(0)
            return
        ratio = price / self.avr_cost
        self.stock_money = self.num * price
        self.date = date
        percent = (ratio - 1) * 100
        if (percent >= 10):
            self.exchange(-0.85, price, date)
        elif (percent < 10 and percent > 2):
            self.exchange(-0.00085 * (percent ** 3), price, date)
        else:
            self.write_log(0)

    # 更新股价对账户影响，不自动卖出
    def Wave(self, price, date):
        if (self.avr_cost <= 0):
            self.write_log(0)
            return
        self.stock_money = self.num * price
        self.date = date
        self.price=price
        self.write_log(0)

    # 返回当前账户明细
    def return_dict(self):
        dict = {}
        dict["股票市值"] = self.stock_money
        dict["平均成本"] = self.avr_cost
        dict["可用金额"] = self.available_money
        dict["日期"] = self.date
        return dict

    # 返回操作日志
    def return_logs(self):
        return self.logs


#画图
def graph(fig,X,Y1,Y2,Y3,title,nrows=1,ncols=1,pos=1):
    ax1 = fig.add_subplot(nrows, ncols, pos)
    l1=ax1.plot(X, Y1, label="股价")
    ax1.set_xlabel("日期")
    ax1.set_ylabel("股价")

    ax1.set_title(title,fontsize=15)
    ax2 = ax1.twinx()
    l2=ax2.plot(X, Y2, 'r-', label="总金额")
    l3=ax2.plot(X, Y3, 'b--', label="剩余金额")
    ax2.set_ylabel("金额")
    lns=l1+l2+l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)                    #放个框框
    ax1.grid()                                      #画个格子，好看些



#模拟抢反弹收益  start_date 应该形似"2020-04-20"
def get_rebound(code,bound_days,start_date,end_date):
    prices=[1500 for i in range(bound_days)]
    df=make_price_frame(code,start_date,end_date)
    acc = account(0, 0, 100, pd.Timestamp(start_date))
    start_price=df.iloc[0,0]
    for i in range(df.shape[0]):
        price=df.iloc[i,0]
        prices[i%bound_days]=price
        date=df.index[i]

        if(i>=bound_days-1 and min(prices)==price):
            quantity=10                                #动态规划买入仓位
            acc.exchange(quantity,price,date)
            # plt.plot(date,price,'ro')
        else:
            acc.wave(price, date)  # 涨幅积累到一定程度会自动卖出

    Dict=acc.return_dict()
    logs=acc.return_logs()
    ratio1=100*(price-start_price)/start_price
    list_money=[];list_free=[]
    for log in logs[1:]:
         list_money.append(log["可用金额"]+log["股票市值"])
         list_free.append(log['可用金额'])
    m=np.array(list_money)
    p=np.array(list_free)
    print("短线策略，从{0}---{1}，股票涨跌幅为{2}%，模拟盘涨跌幅为{3}%".format(start_date,date,ratio1,Dict["可用金额"] + Dict["股票市值"]-100))
    return [ratio1,Dict["可用金额"]+Dict["股票市值"]-100],df.index,df.values,m,p


# 统计价格区间
def set_interval(df,refer_days):
    peak = 0;
    buttom = 1500;
    sum = 0
    for i in range(1,refer_days):
        i=-i
        price =df.iloc[i,0]
        sum = sum + price
        if (peak < price): peak = price
        if (buttom > price): buttom = price
    average = sum / (-i + 1)
    middle = (peak + buttom) / 2
    standard = 0.6 * average + 0.4 * middle
    high = 0.35 * peak + 0.64 * (2 * standard - buttom)
    d_high = (1.8 * high + standard) / 2.8
    low = 0.68 * buttom + 0.3 * standard                  #参数都是乱调的，效果还将就
    Dict = {}
    Dict["standard"] = standard
    Dict["high"] = high
    Dict["d_high"] = d_high
    Dict["low"] = low
    return Dict

# 模拟价格区间收益
# 以起点时间为锚，往回统计refer_days的peak，buttom，middle，average
# 计算出standard，high，d_high，low
# 根据区间位置决定流入流出速度，与操作间隔
# 经过continue_days，展示结果
def gain_from_interval(code, refer_days, start_date, end_date):
    buy_quota = 10
    sell_quota = -15
    day_quota = 10
    interval_check=0
    df_interval=make_price_frame(code,end_date=start_date)
    interval= set_interval(df_interval, refer_days)
    df=make_price_frame(code,start_date,end_date)
    acc = account(0, 0, 100, pd.Timestamp(start_date))
    start_price =df.iloc[0,0]
    for i in range(df.shape[0]):
        interval_check+=1
        if(interval_check>refer_days*1.5):
            df_interval = make_price_frame(code, end_date=date)
            interval = set_interval(df_interval, refer_days)
            interval_check-=refer_days*1.5
        price =df.iloc[i,0]
        date=df.index[i]
        if (price > interval["standard"]):
            quantity = (price - interval["standard"]) ** 2 * sell_quota / (interval["high"] - interval["standard"]) ** 2
            if (price >= interval["high"] and day_quota > 0):
                acc.Exchange(sell_quota, price, date)
                day_quota = -1
            elif ((price >= interval["d_high"] and day_quota > 2) or (price < interval["d_high"] and day_quota > 5)):
                acc.Exchange(quantity, price, date)
                day_quota = -1
            else:
                acc.Wave(price, date)
        else:
            if (price <= interval["low"] and day_quota > 2):
                acc.Exchange(buy_quota, price, date)
                day_quota = -1
            elif (day_quota > 6):
                quantity = (price - interval["standard"]) ** 2 * buy_quota / (
                            interval["low"] - interval["standard"]) ** 2
                acc.Exchange(quantity, price, date)
                day_quota = -1
            else:
                acc.Wave(price, date)
        day_quota = day_quota + 1
    Dict = acc.return_dict()
    logs = acc.return_logs()
    ratio1 = 100 * (price - start_price) / start_price
    list_money = [];
    list_free = []
    for log in logs[1:]:
        list_money.append(log["可用金额"] + log["股票市值"])
        list_free.append(log['可用金额'])
    m = np.array(list_money)
    p = np.array(list_free)
    print("区间策略，从{0}---{1}，股票涨跌幅为{2}%，模拟盘涨跌幅为{3}%".format(start_date, date, ratio1, Dict["可用金额"] + Dict["股票市值"] - 100))
    return [ratio1, Dict["可用金额"] + Dict["股票市值"] - 100], df.index, df.values, m, p
#给一个日期，然后回去翻该股票的历史数据，做出一个模型，然后再根据这个模型给出决策list
def get_model_and_order(df,code,model_end,predict_days,model_days=385): #实际模型天数为model_days-20
    delta1 = timedelta(model_days)
    model_start = datetime.fromisoformat(model_end)
    model_start = model_start - delta1
    model_start = model_start.strftime("%Y-%m-%d")             #获取模型开始，和预测结束的时间
    model_pos = do_ml(code,model_start, model_end)
    clf = joblib.load(model_pos)                               # 机器学习模型生成
    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(df.loc[model_end:].iloc[0:predict_days].values)
    order_list = clf.predict(X)                                  #得到决策list
    return list(order_list)
#实时模型建立加实时交易
def Model_gain(code,start_date,end_date):
    quota=4                                   #quota是为了防止频繁交易所设置的
    change_days=125                          #125为半年的交易日数目，每半年都换个新模型，获得新的决策列表
    position = ml_data("2000-01-01", end_date, code)
    df_model = pd.read_csv(position, index_col=0)
    delta = timedelta(days=20)
    df_model=df_model[start_date:end_date]
    transaction_days=df_model.shape[0]                                 #总交易日数
    order_list=get_model_and_order(df_model,code,start_date,change_days)
    df = make_price_frame(code, start_date, end_date)
    acc = account(0, 0, 100, pd.Timestamp(start_date))
    start_price = df.iloc[0, 0]
    for i in range(transaction_days):
        if(i%change_days==0 and not i ==0):
            order_list+=get_model_and_order(df_model,code,date.strftime("%Y-%m-%d"),change_days)
        price = df.iloc[i, 0]
        date = df.index[i]
        if (order_list[i] == 1 and quota >= 4):
            quantity = 25
            quota = 0
            acc.Exchange(quantity, price, date)
            # plt.plot(date,price,'ro')
        elif (order_list[i] == -1 and quota >= 4):
            quantity = -30
            quota = 0
            acc.Exchange(quantity, price, date)
        else:
            acc.Wave(price, date)
        quota += 1
    Dict = acc.return_dict()
    logs = acc.return_logs()
    ratio1 = 100 * (price - start_price) / start_price
    list_money = []
    list_free = []
    for log in logs[1:]:
        list_money.append(log["可用金额"] + log["股票市值"])
        list_free.append(log['可用金额'])
    m = np.array(list_money)
    p = np.array(list_free)
    print("机器模拟，从{0}---{1}，股票涨跌幅为{2}%，模拟盘涨跌幅为{3}%".format(start_date, date, ratio1, Dict["可用金额"] + Dict["股票市值"] - 100))
    return [ratio1, Dict["可用金额"] + Dict["股票市值"] - 100], df.index, df.values, m, p, order_list[:transaction_days]



#标准答案，用机器学习中的Y（即预知未来涨跌的决策list）作为买卖依据
def answer_gain(model_path,code,start_date,end_date):
    quota=8
    df = make_price_frame(code, start_date, end_date)
    df = process_wave(df, code)
    order_list=buy_sell_clue_2(df,code)                 #拿个答案
    acc = account(0, 0, 100, pd.Timestamp(start_date))
    start_price = df.iloc[0, 0]
    for i in range(0,len(order_list)):
        price = df.iloc[i, 0]
        date = df.index[i]
        if (order_list[i]==1 and quota>=8):
            quantity = 50                              #标准答案就要有标准答案的手笔，一次交易一半
            quota=0
            acc.Exchange(quantity, price, date)
            # plt.plot(date,price,'ro')
        elif(order_list[i]==-1 and quota>=8):
            quantity=-50
            acc.Exchange(quantity,price,date)
        else:
            acc.Wave(price, date)
        quota+=1

    Dict = acc.return_dict()
    logs = acc.return_logs()
    ratio1 = 100 * (price - start_price) / start_price
    list_money = [];
    list_free = []
    for log in logs[1:]:
        list_money.append(log["可用金额"] + log["股票市值"])
        list_free.append(log['可用金额'])
    m = np.array(list_money)
    p = np.array(list_free)
    print("标准答案，从{0}---{1}，股票涨跌幅为{2}%，模拟盘涨跌幅为{3}%".format(start_date, date, ratio1, Dict["可用金额"] + Dict["股票市值"] - 100))
    return [ratio1, Dict["可用金额"] + Dict["股票市值"] - 100], df.index, df.iloc[:,0].values, m, p,order_list

def main():
    start_date="2008-12-24"
    end_date="2020-03-23"
    code='000001'
    path="info/{}.csv"
    # code=input("请输入股票代码：")
    # start_date=input("开始日期(yyyy-mm-dd)：")
    # end_date = input("结束日期(yyyy-mm-dd)：")
    fig=plt.figure()
    gain_list,X,Y1,Y2,Y3=get_rebound(code,7,start_date,end_date)
    graph(fig,X,Y1,Y2,Y3,"短线反弹",4,1,1)
    gain_list,X,Y1,Y2,Y3=gain_from_interval(code,200,start_date,end_date)
    graph(fig, X, Y1, Y2, Y3,"区间策略",4, 1, 2)
    gain_list, X, Y1, Y2, Y3,l1 = Model_gain(code, start_date, end_date)
    graph(fig, X, Y1, Y2, Y3, "机器学习", 4, 1, 3)
    gain_list, X, Y1, Y2, Y3 ,l2= answer_gain('model.pkl', code, start_date, end_date)
    graph(fig, X, Y1, Y2, Y3, "标准答案", 4, 1, 4)
    plt.show()
    count=0
    for i in range(len(l1)):
        if l1[i]==l2[i]:
            count+=1
    print("准确率:",count/len(l1))
if __name__ == '__main__':
    main()