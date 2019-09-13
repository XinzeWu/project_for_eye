#区别于svm和svm（2）的是兼顾了网格搜索、直接从TXT文件出结果，不再中间保留文件
# 利用自己手动定义训练集和测试集的方式尝试实现得到输入数据的坐标
from sklearn.model_selection import GridSearchCV
import numpy as np
import scipy.signal as signal
from sklearn import svm
from sklearn import metrics
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def ratio_svm(number):
    """
    此函數就是為了提取特征
    :return:
    """

    all = []
    all.append(
        ["index", "datamed", "datamean", "datavar", "datamax", "datamin",
         "_datamed", "_datamean", "_datavar", "_datamax", "_datamin",
         "__datamed", "__datamean", "__datavar", "__datamax", "__datamin",
         "total", "low", "_lessmed", "_lessmean", "_lessvar", "_lessmax", "_lessmin",
         "less", "lessmed", "lessmean", "lessvar", "lessmax", "lessmin"])
    x_train, x_test, y_train, y_test = [],[],[],[]
    for m in range(5):
        m_data = []
        m_label = []
        for n in range(8):
            if (m == 2 and n == 7):
                continue
            else:
                data = []
                for line in open("D:\\data_for_net\\{}\\{}\\txt\\aspect_ratio.txt".format(m, n)):
                    data.append(float(line))
                data = np.array(data)
                lens = len(data)//number
                for i in range(lens):
                    all_data = []
                    all_data.append(m)
                    all_label = []
                    all_data.append(np.median(data[i*number:(i+1)*number]))  # 特征1 ：纵横比的中位数
                    all_data.append(np.mean(data[i * number:(i + 1) * number]))
                    all_data.append(np.var(data[i*number:(i+1)*number]))
                    all_data.append(np.max(data[i * number:(i + 1) * number]))
                    all_data.append(np.min(data[i * number:(i + 1) * number]))

                    _data = []
                    for j in range(number-1):
                        _data.append(data[i * number+1+j]-data[i * number+j])
                    all_data.append(np.median(_data))  # 特征2 ： 纵横比倒数的中位数
                    all_data.append(np.mean(_data))
                    all_data.append(np.var(_data))
                    all_data.append(np.max(_data))
                    all_data.append(np.min(_data))

                    __data = []
                    for jj in range(number - 2):
                        __data.append(_data[jj+1] - _data[jj])
                    all_data.append(np.median(__data))# 特征3 ： 纵横比二阶导的中位数
                    all_data.append(np.mean(__data))
                    all_data.append(np.var(__data))
                    all_data.append(np.max(__data))
                    all_data.append(np.min(__data))

                    threshold = 0.27
                    count = 0
                    total = 0
                    low = 0

                    for k in range(number):
                        if data[i*number+k] < threshold:
                            count += 1
                            low += 1
                        else:
                            if count > 2:
                                total += 1
                            count = 0
                    all_data.append(total)   # 特征4 ： 眨眼总数
                    all_data.append(low/number) # 特征5 ： 闭眼时长占比

                    less = signal.argrelextrema(data[i*number:(i+1)*number], np.less)

                    _less = []
                    for i in range(len(less[0])-1):
                        _less.append(less[0][i+1]-less[0][i])

                    all_data.append(np.median(_less))
                    all_data.append(np.mean(_less))
                    all_data.append(np.var(_less))
                    all_data.append(np.max(_less))
                    all_data.append(np.min(_less))

                    all_data.append(len(less[0]) / number) # 特征6： 极小值的占比
                    all_data.append(np.median(data[less])) # 特征7： 极小值的中位数
                    all_data.append(np.mean(data[less]))
                    all_data.append(np.var(data[less]))
                    all_data.append(np.max(data[less]))
                    all_data.append(np.min(data[less]))
                    all.append(all_data)

                    all_label.append(m)
                    '''用来做第一步划分
                    if  m == 1 or m == 2:
                        all_label.append(0)
                    if m == 0 or m == 3 or m == 4:
                        all_label.append(1)'''

                    m_data.append(all_data)
                    m_label.append(all_label)
        mlen = len(m_label)

        #取每个标签的前25%作为测试集
        test = int(mlen*0.25)
        for i in range(test):
            x_test.append(m_data[i+3*test])
            y_test.append(m_label[i+3*test])
        for i in range(3*test):
            x_train.append(m_data[i])
            y_train.append(m_label[i])


    with open('D:\\Projects\\mysvm\\data\\5分类\\{}.csv'.format(number), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all)

    return all





