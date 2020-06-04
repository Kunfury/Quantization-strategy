import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
#将dataframe存储为csv文件的小函数
def frame_to_csv(df,name,path="info/",bool_print=False):
    position=path+name
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + name)
    # if not os.path.exists(path+name):
    #     df.to_csv(path+name)
    #     if bool_print:
    #         print("saved in {}".format(path+name))
    # else:
    #         print("Already exists {}".format(path+name))
    return position
#检验机器学习成果的小函数，数各个结果的数目
def count(list):
    Dict={}
    for i in list:
        if Dict.get(i):
            Dict[i]+=1
        else:
            Dict[i]=1
    return Dict
#之前用来，排除退市股票用的，现在变换了设计思路，所以不再需要
def out_market_del(codes,start_date,end_date):
    out_market = []  # 退市
    for code in codes:
        # parse_dates指是否转换日期格式，index_col指索引参考列,encoding='gb2312'，内含中文的解决方法
        df = pd.read_csv("info/{}.csv".format(code), parse_dates=True, index_col=0, encoding='gb2312')
        if (df.index[0] < end_date or df.index[-1]>start_date):
            out_market.append(code)
            continue
    for i in out_market:
        codes.remove(i)

#将一系列codes的收盘价信息提取出来并保存，但由于对于数据整齐的要求，会破坏数据完整性
def make_price_frames_save(codes,start_date,end_date):
    df1=pd.DataFrame()                                             #退市
    for count,code in enumerate(codes):
        df=make_price_frame(code,start_date,end_date)
        if df1.empty:
            df1=df
        else:
            df1=df1.join(df)
        df1.replace(0,np.nan,inplace=True)
        df1.fillna(method="pad",inplace=True)
    position=frame_to_csv(df1,"price_frame.csv")
    print("{}支股票{}至{}收盘价信息已被存储于{}".format(count+1,start_date,end_date,position))
    return position

#将一个code收盘价信息提取出来
def make_price_frame(code,start_date="2000-01-01",end_date="2020-06-18",path_format="info/{}.csv"):
    # parse_dates指是否转换日期格式，index_col指索引参考列,encoding='gb2312'，内含中文的解决方法
    df = pd.read_csv(path_format.format(code), parse_dates=True, index_col=0, encoding='gb2312')
    df.rename(index={}, columns={'收盘价': df.iloc[0, 0][1:]}, inplace=True)
    df.drop(df.columns[3:], axis=1, inplace=True)
    df.drop(df.columns[0:2], axis=1, inplace=True)  # 扔掉除收盘价外的其他信息
    df.sort_index(axis=0, ascending=True, inplace=True)
    df = df[start_date:end_date]
    df.replace(0,np.nan,inplace=True)
    df.dropna(inplace=True)
    return df

#收集与目标股票，相关性较高的股票列表，以加大机器学习的样本集（现已退役）
def corr_code_sets(code,position,require=0.6):
    codes=[]
    df = pd.read_csv(position)
    df_corr = df.corr()

    for c in df_corr.columns.values:
        if df_corr[code][c]>=require:
            codes.append(c)
    return codes

#多只股票相关系数图（工具人，已经截完图了，不需要他了）
def visualize_corr(position,code_between=True):
    if code_between==True:
        df=pd.read_csv(position)
    else:
        df = pd.read_csv(position, encoding='gb2312')
    df_corr=df.corr()
    data=df_corr.values
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    heatmap=ax1.pcolor(data)       #方格图绘制，颜色选择
    fig1.colorbar(heatmap)                            #方格图右侧的解释竖条
    ax1.set_xticks(np.arange(data.shape[0] + 1))
    ax1.set_yticks(np.arange(data.shape[1] + 1))      #x,y轴每1单位长度，顿一下
    ax1.invert_yaxis()                                #y轴反过来
    ax1.xaxis.tick_top()                              #x轴跑到顶上
    ax1.set_xticklabels(df_corr.columns,FontProperties="SimHei")
    ax1.set_yticklabels(df_corr.index,FontProperties="SimHei")                     #设置x,y所标内容
    plt.xticks(rotation=90)                           #x轴文字倒转90度
    heatmap.set_clim(0,1)                             #设置显示corr数值范围
    plt.tight_layout()                                #为了看得顺眼，调个位置
    plt.show()

#形成单只股票七日波动数据
def process_wave(df,code):
    i=4
    df.fillna(0,inplace=True)
    df["{}_{}d".format(code,i)]=(df[code].shift(-i)-df[code])/df[code]
    df.fillna(0,inplace=True)
    return df

#形成全部股票的七日波动数据（退役）
def process_wave_s(position):
    df = pd.read_csv(position, index_col=0)
    codes = df.columns.values.tolist()
    for code in codes:
        df=process_wave(df,code)
    position =frame_to_csv(df,"price_wave_frame.csv",bool_print=True)
    return df,position

#波动大于+2%，买入；小于-2%，卖出；反之，持有。返回单只股票包含买入卖出信号的target_list（退役）
def buy_sell_clue(df,*codes):
    requirenment=0.03
    target_list=[]
    for code in codes:
        for i in range(df.shape[0]):
            bl = 0
            ticker_wave=df.iloc[i]
            for col in ticker_wave["{}_1d".format(code):"{}_7d".format(code)]:
                if col>requirenment:
                    target_list.append(1)
                    bl=1
                    break
                if col<-requirenment:
                    target_list.append(-1)
                    bl=1
                    break
            if(bl==0):
                target_list.append(0)
    return target_list
#未来七日内涨幅超过3%买入，跌幅超过-2%卖出
def buy_sell_clue_2(df,code):
    requirenment_1= 0.03
    requirenment_2=-0.02
    target_list = []
    for i in range(df.shape[0]):
        bl = 0
        ticker_wave = df.iloc[i]
        col = ticker_wave["{}_4d".format(code)]
        if col > requirenment_1:
            target_list.append(1)
        elif col < requirenment_2:
            target_list.append(-1)
        else:
            target_list.append(0)

    return target_list
#返回涨跌幅，买入卖出信号，df
def extract_feature_sets(code,start_date,end_date):
    order_list=[]
    df = make_price_frame(code, start_date, end_date)
    df = process_wave(df, code)
    df = df.iloc[20:]
    order_list += buy_sell_clue_2(df, code)
                  #采集机器学习的Y，即标准答案
    # df["{}_target".format(code)] = list  # 操作数据
    #print("{}支股票，共{}个交易日中的信号：".format(len(codes), len(order_list)), count(order_list))

    position=ml_data(start_date,end_date,code)
    df=pd.read_csv(position,index_col=0)                       #机器学习的材料
    return df.values,np.array(order_list)

#机器学习
def do_ml(codes,start_date,end_date):
    X, Y = extract_feature_sets(codes,start_date,end_date)
    standard_scaler=StandardScaler()
    X=standard_scaler.fit_transform(X)                    #标准化一波
    # x_test=X[-7:];y_test=Y[-7:]
    # x_train=X[0:-7];y_train=Y[0:-7]

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.10)
    clf=RandomForestClassifier(n_estimators=11)           #试了巨多模型，还是这个好用些
    clf.fit(x_train,y_train)
    # confidence=clf.score(x_test,y_test)
    # print("准确率：",confidence)
    # prediction=clf.predict(x_test)
    # print("预测结果：",count(prediction))                  #为了不破坏模拟盘输出信息的整齐队形，就先注释掉
    # clf.fit(X,Y)
    # pre=clf.predict(x_test)
    model_path='model.pkl'
    joblib.dump(clf, model_path)
    return model_path

#采用趋势法投资（追涨杀跌）的投资理念，为机器选择学习样本。
#选择成交量，20日加权价的std，20日加权平均价与收盘价的涨跌幅比 作为学习样本
def ml_data(start_date,end_date,code):
    count=0
    df1=pd.DataFrame()

    df = pd.read_csv("info/{}.csv".format(code), index_col=0, parse_dates=True, encoding='gb2312')
    df.drop(df.columns[11:], axis=1, inplace=True)
    df.drop(df.columns[3:8], axis=1, inplace=True)
    df.drop(df.columns[0:2], axis=1, inplace=True)  # 收盘价，成交量，换手率，涨跌幅
    df.sort_index(axis=0, ascending=True, inplace=True)
    df = df[start_date:end_date]
    df.replace('None', np.nan, inplace=True)
    df.dropna(inplace=True)  # 对所有未开市时的无用股票信息，进行删除
    df["单日股价*量"] = df["收盘价"] * df["成交量"]
    df["20日均成交量"] = df["成交量"].rolling(20, min_periods=1).mean()
    df["量幅"] = (df["成交量"] - df["20日均成交量"]) / df["20日均成交量"]
    df["20日均价"] = (df["单日股价*量"].rolling(20, min_periods=1).mean()) / df["20日均成交量"]
    df["sigma1"] = df["单日股价*量"].rolling(20, min_periods=1).std() / df["20日均成交量"]
    df["20日股价均价差比"] = (df["收盘价"] - df["20日均价"]) / df["20日均价"]
    df['u'] = df["20日股价均价差比"].rolling(20, min_periods=1).mean()
    df["sigma"] = df["20日股价均价差比"].rolling(20, min_periods=1).std()
    df["a-u"] = df["20日股价均价差比"] - df['u']
    df["a-u+sigma"]=df["20日股价均价差比"]-df['u']+df["sigma"]
    df["a-u-sigma"] = df["20日股价均价差比"] -df['u']- df["sigma"]
    df.drop(["收盘价", "单日股价*量", "20日均价", "换手率","a-u","a-u+sigma","a-u-sigma"], axis=1, inplace=True)
    df = df.iloc[20:]
    #以下为可挑选的学习参数：
    # "收盘价", "单日股价*量", "涨跌幅", "换手率", "20日均价", "20日股价均价差比", "u", "sigma", "20日均成交量", "成交量","a-u+sigma","a-u-sigma","量幅","a-u"
    df.fillna(0, inplace=True)
    #print(df.head(1))
    position = frame_to_csv(df, "机器学习_frame.csv")
    return position

#变换了策略，故退役
def main():
    pass
    # start_date="2013-01-24"
    # end_date="2016-03-23"
    # codes = ['000001', '000002', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011',
    #          '000012', '000014', '000016', '000017', '000018', '000019', '000020', '000021', '000022', '000023',
    #          '000025', '000026', '000027', '000028', '000029', '000030', '000031', '000032', '000034', '000035',
    #          '000036', '000037', '000038', '000039', '000040', '000042', '000043', '000045', '000046', '000048',
    #          '000049', '000050', '000055', '000056', '000058', '000059', '000060', '000061', '000062', '000063']
    # path="info/{}.csv"
    # # out_market_del(codes, start_date, end_date)
    # #pos=make_price_frames_save(codes,start_date,end_date)
    # #pos=ml_data(start_date,end_date,*codes)
    # #visualize_corr(pos)
    # codes=['000001']
    # # codes=corr_code_sets(code,pos,0.01)                              #找出与目标股票，相关性高的一系列股票
    #
    #
    # # visualize_corr(position1)
    # model_path=do_ml(codes,start_date,end_date)
    #clf = joblib.load(model_path)
    # print(clf.predict(pd.read_csv(ml_data(start_date,end_date,code),index_col=0).values))
if __name__=="__main__":
    main()