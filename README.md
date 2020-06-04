# -项目思路
本项目意在从互联网金融平台，搜集股票的历史数据，以作为模拟股票操作的战场，检验与调整短线抢反弹，中线区间法，机器模型预测法三个股票量化策略，并横向比对，以凸显各自优劣，再在此基础上加以完善。
实现步骤：
1.在股城网爬取股票代码，然后通过网易财经接口获得股票历史数据，再通过pymysql存储。（股票信息收集.py）
2.对股票历史数据进行机器学习，生成模型。（机器学习.py)
3.构建模拟盘的类，并代码实现三个量化策略，令其在模拟盘中实践。并用图像，记录股票走势与策略实施情况，从而便于比对分析。（股票模拟.py)

