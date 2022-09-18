import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn import metrics
from statsmodels.tools import eval_measures
from scipy import stats
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
plt.rc('font', size=10)
pd.set_option('display.max_columns', None)


# ###########---------------set up and plot input data-----------------######################
base_value = 10  # 设置level、trend、season项的基数
steps_day, steps_week = 1, 1
n, m = 100, 2
length = [steps_day*n+steps_day, steps_week*n+steps_week, steps_week*n+steps_week]*m  # 代表周、日序列对的长度，及序列对数目3*m

weights = []
for i in range(-base_value + 1, 1):
    weights.append(0.5 ** i)  # 设置y_level项随机序列的权重呈递减指数分布，底数越小，y_level中较小值所占比例越大。
weights = np.array(weights)


##########################################################--构造乘法周期性时间序列，模拟真实销售；外层是list，内层的每一条序列是series
y_level_actual, y_trend_actual, y_season_actual, y_noise_actual, y_input_mul_actual = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_actual[i] = np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性
    y_level_actual[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(abs(y_season_actual[i])) + np.average(abs(y_season_actual[i]))  # 用指数权重分布随机数模拟水平项
    y_trend_actual[i] = (2 * max(y_season_actual[i]) + np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
        + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
        / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(y_level_actual[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_actual[i] = 3*np.random.standard_t(length[i]-1, length[i])  # 假定数据处于理想状态，并使噪音以乘法方式进入模型，则可令噪音在0附近呈学生分布。
    y_noise_actual[i][abs(y_noise_actual[i]) < max(y_noise_actual[i])*0.9] = 1  # 保留随机数中的离群值，将非离群值置为1
    y_input_mul_actual[i] = (y_level_actual[i] + y_trend_actual[i]) * y_season_actual[i] * y_noise_actual[i]  # 假定周期项以乘法方式组成输入数据

    print(f'第{i}条真实序列中水平项的极差：{max(y_level_actual[i]) - min(y_level_actual[i])}，均值：{np.mean(y_level_actual[i])}')
    print(f'第{i}条真实序列中趋势项的极差：{max(y_trend_actual[i]) - min(y_trend_actual[i])}，均值：{np.mean(y_trend_actual[i])}')
    print(f'第{i}条真实序列中周期项的极差：{max(y_season_actual[i]) - min(y_season_actual[i])}，均值：{np.mean(y_season_actual[i])}')
    print(f'第{i}条真实序列中噪音项的极差：{max(y_noise_actual[i]) - min(y_noise_actual[i])}，均值：{np.mean(y_noise_actual[i])}')
    print(f'第{i}条真实乘法性序列最终极差：{max(y_input_mul_actual[i]) - min(y_input_mul_actual[i])}，均值：{np.mean(y_input_mul_actual[i])}', '\n')

    y_level_actual[i] = pd.Series(np.around(y_level_actual[i], decimals=2)).rename('y_level_actual')
    y_trend_actual[i] = pd.Series(np.around(y_trend_actual[i], decimals=2)).rename('y_trend_actual')
    y_season_actual[i] = pd.Series(np.around(y_season_actual[i], decimals=2)).rename('y_season_actual')
    y_noise_actual[i] = pd.Series(np.around(y_noise_actual[i], decimals=2)).rename('y_noise_actual')
    y_input_mul_actual[i] = pd.Series(np.around(y_input_mul_actual[i], decimals=2)).rename('y_input_mul_actual')
    # y_input_mul_actual[i][y_input_mul_actual[i] < 0.011] = 0.011  # 将series中小于0.011的数置为0.011；因为后续regression_accuracy，regression_evaluation会将series中小于0的置为0.01，若此处不将小于0.011的置为0.011，则画出的图可能与后续两个综合评估函数中所使用的序列不一致。
    print('第{0}条真实序列的初始生成值：'.format(i))
    print(y_input_mul_actual[i], '\n')

##########################################################--构造乘法周期性时间序列，模拟预测销售；外层是list，内层的每一条序列是series
y_level_pred, y_trend_pred, y_season_pred, y_noise_pred, y_input_mul_pred = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_pred[i] = 1/2 * np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性，使预测销售的波动振幅比真实销售小
    y_level_pred[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(abs(y_season_pred[i])) + np.average(abs(y_season_pred[i]))  # 用指数权重分布随机数模拟水平项，使其相对于真实销售有所偏移
    y_trend_pred[i] = (2 * max(y_season_pred[i]) + np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
        + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
        / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(y_level_pred[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_pred[i] = np.random.standard_t(length[i]-1, length[i])  # 假定数据处于理想状态，并使噪音以乘法方式进入模型，则可令噪音在0附近呈学生分布；使其比真实销售的噪音小。
    y_noise_pred[i][abs(y_noise_pred[i]) < max(y_noise_pred[i])*0.9] = 1  # 保留随机数中的离群值，将非离群值置为1
    y_input_mul_pred[i] = (y_level_pred[i] + y_trend_pred[i]) * y_season_pred[i] * y_noise_pred[i]  # 假定周期项以乘法方式组成输入数据

    print(f'第{i}条预测序列中水平项的极差：{max(y_level_pred[i]) - min(y_level_pred[i])}，均值：{np.mean(y_level_pred[i])}')
    print(f'第{i}条预测序列中趋势项的极差：{max(y_trend_pred[i]) - min(y_trend_pred[i])}，均值：{np.mean(y_trend_pred[i])}')
    print(f'第{i}条预测序列中周期项的极差：{max(y_season_pred[i]) - min(y_season_pred[i])}，均值：{np.mean(y_season_pred[i])}')
    print(f'第{i}条预测序列中噪音项的极差：{max(y_noise_pred[i]) - min(y_noise_pred[i])}，均值：{np.mean(y_noise_pred[i])}')
    print(f'第{i}条预测乘法性序列最终极差：{max(y_input_mul_pred[i]) - min(y_input_mul_pred[i])}，均值：{np.mean(y_input_mul_pred[i])}', '\n')

    y_level_pred[i] = pd.Series(np.around(y_level_pred[i], decimals=2)).rename('y_level_pred')
    y_trend_pred[i] = pd.Series(np.around(y_trend_pred[i], decimals=2)).rename('y_trend_pred')
    y_season_pred[i] = pd.Series(np.around(y_season_pred[i], decimals=2)).rename('y_season_pred')
    y_noise_pred[i] = pd.Series(np.around(y_noise_pred[i], decimals=2)).rename('y_noise_pred')
    y_input_mul_pred[i] = pd.Series(np.around(y_input_mul_pred[i], decimals=2)).rename('y_input_mul_pred')
    # y_input_mul_pred[i][y_input_mul_pred[i] < 0.011] = 0.011  # 将series中小于0.011的数置为0.011；因为后续regression_accuracy，regression_evaluation会将series中小于0的置为0.01，若此处不将小于0.011的置为0.011，则画出的图可能与后续两个综合评估函数中所使用的序列不一致。
    print('第{0}条预测序列的初始生成值：'.format(i))
    print(y_input_mul_pred[i], '\n')

# 绘制真实值和对应的预测值序列
for i in range(len(y_input_mul_actual)):
    plt.figure('origin true and predict series {0}'.format(i), figsize=(5,10))
    ax1 = plt.subplot(5,1,1)
    ax2 = plt.subplot(5,1,2)
    ax3 = plt.subplot(5,1,3)
    ax4 = plt.subplot(5,1,4)
    ax5 = plt.subplot(5,1,5)
    y_input_mul_actual[i].plot(ax=ax1, legend=True)
    y_level_actual[i].plot(ax=ax2, legend=True)
    y_trend_actual[i].plot(ax=ax3, legend=True)
    y_season_actual[i].plot(ax=ax4, legend=True)
    y_noise_actual[i].plot(ax=ax5, legend=True)
    y_input_mul_pred[i].plot(ax=ax1, legend=True)
    y_level_pred[i].plot(ax=ax2, legend=True)
    y_trend_pred[i].plot(ax=ax3, legend=True)
    y_season_pred[i].plot(ax=ax4, legend=True)
    y_noise_pred[i].plot(ax=ax5, legend=True)
    plt.show()


def print_execute_time(func):
    from time import time
    # 定义嵌套函数，用来打印出装饰的函数的执行时间
    def wrapper(*args, **kwargs):
        # 定义开始时间和结束时间，将func夹在中间执行，取得其返回值
        start = time()
        func_return = func(*args, **kwargs)
        end = time()
        # 打印方法名称和其执行时间
        print(f'{func.__name__}() execute time: {end - start}s')
        # 返回func的返回值
        return func_return

    # 返回嵌套的内层函数
    return wrapper


def geo_zscore(samples, bias=1, ddof=1):
    """
    gzscore = log(a/gmu) / log(gsigma), where gmu (resp. gsigma) is the geometric mean (resp. standard deviation).
    """
    # The geometric standard deviation is defined for strictly positive values only, because of log.
    # 特别重要：不要写成samples += abs(min(samples)) + bias，否则当同一个序列第二次调用geo_zscore时，这条序列已经被改变，不是原序列了，
    # 体现为函数对输入变量的"全局"赋值作用；或者说每调用一次函数，输入序列samples就被"全局"改变一次。
    samp_shift = samples + abs(min(samples)) + bias  # 使samp_shift>0，满足下面np.log的要求
    # gstd = exp(std(log(a)))
    geo_std = np.exp(np.std(np.log(samp_shift), ddof=ddof))
    # let degrees of freedom correction in the calculation of the standard deviation to be 0.
    gzscore = np.log(samp_shift / stats.gmean(samp_shift)) / np.log(geo_std)

    return gzscore


def s_curve_interp(n, x=(1, 10, 20, 30), y=(1e-5, 0.1, 0.9, 1)):
    """
    n：需要根据构造的插值函数得到对应y值的x坐标
    x：用于构造插值函数的点的x坐标，n值最好在x的范围内，因为插值函数不合适做外推
    y：用于构造插值函数的点的y坐标；x和y是成对的坐标，遵循奥卡姆剃刀原则，最少只需四个点，即三段插值函数，就可以构造任意大致规律的全局函数；若点数越多，构造出的函数形态就可以控制得越细致。
    return: 构造出的插值函数的x坐标为n时，对应的一个y坐标值
    """
    if x[0] <= n < x[1]:
        cs1 = CubicSpline(x[:2], y[:2], bc_type=((1, y[1] / x[1]**2), (1, y[1] / x[1]**0.5)), extrapolate=False)
        r = cs1(n)
        if r < 0:
            r = cs1(x[0]+1)
    elif x[1] <= n < x[2]:
        cs2 = CubicSpline(x[1:3], y[1:3], bc_type=((1, y[1] / x[1]**0.5), (1, (y[2]-y[1]) / (x[2]-x[1])**2)), extrapolate=False)
        r = cs2(n)
    else:
        cs3 = CubicSpline(x[-2:], y[-2:], bc_type=((1, (y[2] - y[1]) / (x[2] - x[1]) ** 2), (1, (y[3] - y[2]) / (x[3] - x[2]) ** 2)), extrapolate=False)
        r = cs3(n)
        if r > 1:
            r = cs3(x[-1]-1)
    return float(r)


def dyn_seri_weighted(seri, type=None, w=None, initial=1, r=2, d=1, low=0, up=1, critical_y=(1e-10, 0.15, 0.5, 1)):
    """
    :param seri: 需要进行加权平均变成一个值的一维数组，可以是series,array,list,tuple；在调用不同type时需注意可能有不同的顺序。
    :param type: 选择序列seri加权的类型，'amean_geo', 'amean_arith', 'amean_trim', 'amean_sigmoid', 'gmean', 'hmean',
    'smean', 'normean', None。
     其中，
    'amean_geo'是权重从左至右呈几何级数递减的算术平均；
    'amean_arith'是权重从左至右呈算术级数递减的算术平均；
    'amean_trim'是两侧截尾简单算术平均，是按小于临界值和大于临界值去截断，而不是按顺序的索引截断；
    'amean_sigmoid'是权重从左至右呈S型升高的算术平均，即加权结果不太受序列左侧点的影响，而受右侧点影响比较大；
    'normean'是基于偏斜正态概率的加权算术平均，受两侧离群值影响小，因为假定数据出现离群值的概率小；
     None是对序列做简单算术平均或权重为w的加权算术平均。
    'gmean'是简单几何平均或者权重为w的加权几何平均，比算数平均更接近较小值；
    'hmean'是简单调和平均，结果比'gmean'更趋近较小值；
    'smean'是均方根，比算术平均更接近较大值；
    :param w: 一维的权重系数，可以是series,array,list,tuple；若手动输入，其长度必须和一维数组seri（即序列点数）相等
    :param r: type='amean_geo'时，指定几何级数分母的公比
    :param d: type='amean_arith'时，指定算数级数分母的公差
    :param initial: type='amean_arith'时，指定算数级数分母的初始值
    :param low: type='amean_trim'，截尾简单算数平均中，从小到大，截断比第low个值更小的那些值，但不包括第low个较小值
    :param up: type='amean_trim'，截尾简单算数平均中，从大到小，截断比第up个值更大的那些值，但不包括第up个较大值
    :param critical_y: type='amean_sigmoid'中，设置S型曲线在四个临界点处的y值，相邻两个y值间的差值δy越大，则在该区间内曲线上升越快；δy越小，上升越慢。
    :return: seri各点与权重w相乘再相加，返回的一个加权后的最终值
    """
    seri = np.array(seri)
    if type not in ['amean_geo', 'amean_arith', 'amean_trim', 'amean_sigmoid', 'gmean', 'hmean', 'smean', 'normean', None]:
        raise Exception('type must be one of the \'amean_geo\', \'amean_arith\', \'amean_trim\', \'amean_sigmoid\', '
                        '\'gmean\', \'hmean\', \'smean\', \'normean\', or \'None\'')
    elif w is None:
        w = np.ones(len(seri)) / sum(np.ones(len(seri)))  # 生成均等权重
    if len(w) != len(seri):
        raise Exception('len(w) != len(seri)')
    # weighted arithmetic average, weights are geometric series or arithmetic series
    elif type in ['amean_geo', 'amean_arith']:
        w = list()
        if type == 'amean_geo':
            for i in range(len(seri)):
                w.append(initial * (1 / r) ** i)  # 生成首项是initial，公比是(1/r)的几何级数作权重；权重从左至右呈指数型降低
        else:
            for i in range(len(seri)):
                w.append(1 / (initial + d * i))  # 生成首项是initial，公差是d的算术级数，再做倒数作为权重；权重从左至右呈指数型降低
        w = np.array(w) / sum(w)
        return np.dot(np.array(seri), w)
    elif type == 'amean_trim':
        if low < 0 or low > len(seri) - 1 or up < 1 or up > len(seri) or (not isinstance(low, int)) \
                or (not isinstance(up, int)):
            raise Exception('low is index from the start, up is index from the end, and must be \'int\'')
        seri.sort()
        return stats.tmean(seri, (seri[low], seri[-up]), inclusive=(True, True))
    elif type == 'amean_sigmoid':  # 权重从左至右呈S型升高，即加权结果不太受左侧点的影响，而受右侧点影响比较大
        if len(seri)<4:
            # if length of series < 4, use equal weights, i.e. simple algorithem average
            return np.dot(np.array(seri), w)
        else:
            # 用于构造函数的坐标点
            critical_x = (1, len(seri) / 3, len(seri) * 2 / 3, len(seri))  # 将x轴上的定义域分为均等的三段
            critical_y = critical_y
            xnew = np.arange(critical_x[0], critical_x[-1] + 1, 1)  # 设置每个需要计算权重的x坐标
            ynew = [s_curve_interp(i, x=critical_x, y=critical_y) for i in xnew]  # 计算这些x坐标在S曲线上对应的y值
            # 根据构造的函数生成归一化的权重w。因为每个w的分子与构造曲线的每个y值完全相同，而每个w的分母都是sum(ynew)，
            # 所以w的分布完全由其分子确定，而其分子的分布与构造曲线y值的分布相同，所以w的分布特征与构造曲线的分布特征完全相同。
            w = [i / sum(ynew) for i in ynew]
            return np.dot(np.array(seri), w)
    elif type == 'gmean':
        return stats.gmean(seri, weights=w)  # simple geometric average, or weighted geometric average
    elif type == 'hmean':
        return stats.hmean(seri)  # simple harmonic average
    elif type == 'smean':
        return metrics.mean_squared_error(seri, np.zeros(len(seri)), sample_weight=w, squared=False)  # RMSE with 0
    elif type == 'normean':
        samp_geo = geo_zscore(seri)  # 对序列seri做几何标准化变换，将接近幂律分布的序列转化为接近正态分布
        samp_yj, lmda_yj = stats.yeojohnson(seri)  # 对序列seri做幂变换，增强序列的正态性
        samp_geo_yj, lmda_geo_yj = stats.yeojohnson(samp_geo)  # 对序列samp_geo做幂变换，增强其正态性
        # 计算normalize后序列的正态性指标，k2和p-value，k2=skewness^2+kurtosis^2，越接近0正态性越强；p越接近1表明越有可能是从正态总体中抽样得到的序列samp_yj或samp_geo_yj
        nt_samp_yj = stats.normaltest(samp_yj)
        nt_samp_geo_yj = stats.normaltest(samp_geo_yj)
        if (nt_samp_yj[0] < nt_samp_geo_yj[0]) and (nt_samp_yj[1] > nt_samp_geo_yj[1]):  # 表示序列nt_samp_yj的正态性大于nt_samp_geo_yj
            skew_yj = stats.skew(samp_yj)  # 计算序列samp_yj的偏度
            # 最关键的一步，得到变换后的序列samp_yj的概率密度函数pdf，而samp_yj的顺序与传入序列seri的顺序完全相同，而与samp_yj的pdf函数图形上各自变量x的位置无关，即使自变量x就是samp_yj中的各个元素；
            # 则用samp_yj算出的pdf的y值，就可作为原序列seri对应各个点出现的概率，归一化后就可作为原序列seri的权重。
            pdf_y = stats.skewnorm.pdf(samp_yj, skew_yj, loc=np.mean(samp_yj), scale=np.std(samp_yj, ddof=0))
            w = np.array(pdf_y) / sum(pdf_y)
        else:
            skew_geo_yj = stats.skew(samp_geo_yj)
            pdf_y = stats.skewnorm.pdf(samp_geo_yj, skew_geo_yj, loc=np.mean(samp_geo_yj), scale=np.std(samp_geo_yj, ddof=0))
            w = np.array(pdf_y) / sum(pdf_y)
        return np.dot(np.array(seri), w)
    elif type is None:  # 简单或加权算术平均
        w = np.array(w) / sum(w)  # 自定义权重
        return np.dot(np.array(seri), w)


def dyn_df_weighted(df, type=None, w=None, initial=1*2, r=2, d=1/2):
    """
    传入二维数组df；若type='geometric'或'arithmetic'，且输入了w，则w不起作用；若不输入权重，则根据df的列数动态计算基于几何级数或算数级数再作归一化的权重，再做算术平均；
    也可人为输入权重做算术平均；若不输入type和w，则进行简单算数平均；因为使用np.matmul，则df.columns的索引越小，权重越大；将df的各列与权重相乘再相加，得到一条最终的序列。
    :param df: 需要进行加权变成一条序列的二维数组，df的每列代表一条需要进行加权的序列
    :param type: 采用几何级数或算数级数进行加权，或人为指定权重，或默认权重相等，type = 'geometric'或'arithmetic'或None；若type='geometric'或'arithmetic'，且输入了w，则w不起作用。
    :param w: 一维的权重系数，可以是series,array,list,tuple；若手动输入，其长度必须和一维数组seri（即序列点数）相等
    :param r: 指定几何级数分母的公比
    :param d: 指定算数级数分母的公差
    :param initial: 指定算数级数分母的初始值
    :return: df各列与权重w相乘再相加，返回一条最终的序列
    """
    if type not in ['geometric', 'arithmetic', None]:
        raise Exception('type must be one of geometric, arithmetic or None')
    if type is not None:
        w = list()
        if type == 'geometric':
            for i in range(len(df.columns)):
                w.append(initial * (1 / r) ** i)  # 生成首项是initial，公比是(1/r)的几何级数作权重
        else:
            for i in range(len(df.columns)):
                w.append(1 / (initial + d * i))  # 生成首项是initial，公差是d的算术级数，再做倒数作为权重
        w = np.array(w) / sum(w)
    elif (type is None) and (w is None):
        w = np.ones(len(df.columns)) / sum(np.ones(len(df.columns)))  # 生成均等权重
    elif (type is None) and (w is not None) and (len(w) == len(df.columns)):
        w = np.array(w) / sum(w)  # 自定义权重
    else:
        raise Exception('手动输入的权重长度必须和一维数组长度（即序列点数）相等')
    if abs(sum(w)-1) > 0.001:
        raise Exception('weights are not useable')
    return np.matmul(df.values, w)


def train_check(score):
    """
    :param score: 某一模型每周得分组成的一维数组
    :return: 若为'no-train'，则不重新训练；若为're-train'，则重新训练
    """
    score = np.array(score)
    if len(score) <= 4:
        return 'no-train'
    else:
        if score[-1] > score[-13:-1].mean() + score[-13:-1].std():
            return 're-train'
        else:
            return 'no-train'


# y_true, y_pred无限制条件
def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


# y_true, y_pred无限制条件
def emlae(y_true, y_pred):
    """
    :param y_true: 一条真实值序列，array,series,list,tuple均可，长度要与预测值长度相等
    :param y_pred: 一条预测值序列，array,series,list,tuple均可，长度要与真实值长度相等
    :return: EMLAE，将残差离群值压缩后的一次绝对性指标，对残差的离群值不如MAE敏感
    """
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    emlae = np.exp(sum(np.log(abs(y_pred - y_true) + 1)) / n) - 1
    return emlae


# y_true, y_pred无限制条件
def mape(y_true, y_pred):
    """
    y_true和y_pred顺序可变
    :param y_true: 一条真实值序列，array,series,list,tuple均可，长度要与预测值长度相等；为0的真实值和对应的预测值被剔除
    :param y_pred: 一条预测值序列，array,series,list,tuple均可，长度要与真实值长度相等
    :return: MAPE，零次的相对性指标，小数，代表预测值偏离真实值的平均程度；若>1，表示预测值偏离真实值的平均程度超过真实值的1倍，若<1，表示预测值偏离真实值的平均程度小于真实值的1倍
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    y_pred = y_pred[y_true != 0]  # 应先筛选y_pred，保持y_true为原始长度；若先筛选y_true，其会变短，再筛选y_pred时会因索引对应不上而报错
    y_true = y_true[y_true != 0]
    n = len(y_true)
    return round((sum(abs((y_pred - y_true) / y_true)) / n), 4)


# y_true, y_pred无限制条件
def smape(y_true, y_pred):
    """
    y_true和y_pred顺序可变
    当y_true≥0，y_pred≥0时，0≤SMAPE≤2；其中包括：当y_true=0，或y_pred=0时，SMAPE=2；当y_true>0且y_pred>0时，0≤SMAPE<2；当y_true=0且y_pred=0时，剔除该样本不计算SMAPE，此种完全准确的情况可单独统计。
    :param y_true: 一条真实值序列，array,series,list,tuple均可，长度要与预测值长度相等，真实值和预测值同时为0的记录被删掉
    :param y_pred: 一条预测值序列，array,series,list,tuple均可，长度要与真实值长度相等，真实值和预测值同时为0的记录被删掉
    :return: SMAPE，零次的相对性指标，小数，代表预测值偏离真实值的平均程度；不以某类特定的量为基准分母，而是将两类值均作为分母，第一具有对称性，
    第二不会因某类量数值的大小而影响评价结果，所以受离群值影响也比MAPE小；若>1，表示预测值偏离真实值的平均程度超过真实值与预测值均值的1倍，若<1，表示预测值偏离真实值的平均程度小于真实值与预测值均值的1倍。
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    y_pred_f = y_pred[abs(y_pred) + abs(y_true) != 0]
    y_true_f = y_true[abs(y_pred) + abs(y_true) != 0]
    n = len(y_true_f)
    return round(sum(abs(2 * (y_pred_f - y_true_f) / (abs(y_pred_f) + abs(y_true_f)))) / n, 4)


# y_true, y_pred无限制条件
def male(y_true, y_pred):
    """
    param：
        Y:原始序列，array,series,list,tuple均可
        y:拟合序列，array,series,list,tuple均可
    return：
        对数MAE值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true[y_true < 0] = 0
    y_pred[y_pred == -1] = -0.99
    male = sum(abs(np.log(abs(y_true+1)) - np.log(abs(y_pred+1)))) / len(y_true)
    return male


@print_execute_time
def regression_accuracy_pairs(y_true, y_pred, w=(3,2,2,1, 1,1,1,3,1,1, 1,1)):
    """
    :param y_true: 若干条真实序列组成的一个二维list或array或series，其中的每条真实序列必须是带索引的series，为了能对>0的数值的索引取交集；并与y_pred中的预测序列按顺序一一对应
    :param y_pred: 若干条预测序列组成的一个二维list或array或series，其中的每条预测序列必须是带索引的series，为了能对>0的数值的索引取交集；并与y_true中的真实序列按顺序一一对应
    :return: 精度指标，按顺序分别是：最终精度指标，MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE

    测试几种常用的评价序列对的精度指标
    原则：1.带平方项的指标会放大在正负1之外的残差的影响，而压缩在正负1之内的残差的影响，由于各指标越接近零越好，则会惩罚正负1之外的残差，偏离正负1越远，越受到惩罚；而奖励正负1之内的残差。
    2.作对数变换的指标会降低离群值的影响，相对于不带对数项的指标，会惩罚非离群值。因为在(1,+∞)区间内，y=lnx的一阶导数比y=x的一阶导数小，则y=lnx比y=x递增慢。
    3.ln(1/x)+len(x)=0，即对一个数（如x）取对数，与其倒数（1/x）取对数，互为相反数；当x>0，ln(1/x)+x-1≥0，lnx+(1/x)-1≥0，可由求导证明；在(0,4]的区间内，lnx的增长速度快于x**(1/2)，在[4,+∞)区间内，lnx的增长速度慢于x**(1/2)，可由求导证明。
    """

    MAPE, SMAPE, RMSPE, MTD_p2 = [], [], [], []  # 零次的相对性指标
    EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1 = [], [], [], [], [], []  # 一次的绝对性指标
    MSE, MSLE = [], []  # 二次的绝对性指标

    y_true_trun, y_pred_trun = [], []
    for i in range(len(y_true)):
        # 为了统一下列12个精度指标的条件，在y_true和y_pred的序列对中，取大于0的对应点，即排除≤0的对应点；但不应取>0，可以取>0.01，否则若序列中存在大于0但非常接近0的数做分母，可能产生很大的值，不利于得到有效可用的精度值
        judge = (y_true[i] > 0.01) & (y_pred[i] > 0.01)
        if sum(judge):
            y_true_trun.append(y_true[i][judge])
            y_pred_trun.append(y_pred[i][judge])
        else: continue

    if (len(y_true_trun) != len(y_pred_trun)) or (len(y_true_trun) < 2):
        raise Exception('y_true_trun与y_pred_trun中序列条数必须相等且≥2')  # 若序列对的数目小于2，则数值变换后的指标均为1

    plt.figure('finall inputs of regression_accuracy_pairs')
    for i in range(len(y_true_trun)):
        ax = plt.subplot(len(y_true_trun), 1, i+1)
        xlim = plt.gca().set_xlim(0, length[i]-1)  # xlim使图形按x轴上的点数充满横坐标
        y_true_trun[i].plot(ax=ax, legend=True)
        y_pred_trun[i].plot(ax=ax, legend=True)
        print('第{0}组实际输入的序列对：'.format(i))
        print(y_true_trun[i], '\n', y_pred_trun[i], '\n')
    plt.show()

    for i in range(len(y_true_trun)):
        # 第一组，零次的相对性指标：
        MAPE.append(mape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true != 0; no bias
        SMAPE.append(smape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true + y_pred != 0; symmetric MAPE, no bias and more general, less susceptible to outliers than MAPE.
        RMSPE.append(eval_measures.rmspe(np.array(y_true_trun[i]), np.array(y_pred_trun[i])) / 10)  # y_true != 0; susceptible to outliers of deviation ratio, if more, RMSPE will be larger than MAPE.
        MTD_p2.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=2)) # y_pred > 0, y_true > 0; less susceptible to outliers than MAPE when y_pred[i] / y_true[i] > 1, nevertheless, more susceptible to outliers than MAPE when y_pred[i] / y_true[i] < 1

        # 第二组，一次的绝对性指标：
        EMLAE.append(emlae(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件; less susceptible to outliers of error than MAE, so this will penalize small deviation and award large deviation relative to MAE.
        MALE.append(male(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件;
        MAE.append(metrics.mean_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric has no penalty, no bias
        RMSE.append(eval_measures.rmse(np.array(y_true_trun[i]), np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件；susceptible to outliers of error than MAE, so this will penalize large deviation and award small deviation relative to MAE.
        MedAE.append(metrics.median_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； if len(y) is slightly large; won't be affected by outliers completely
        MTD_p1.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=1))  # y_pred > 0, y_true ≥ 0; The higher `p` the less weight is given to extreme deviations between true and predicted targets.

        # 第三组，二次的绝对性指标：
        MSE.append(metrics.mean_squared_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric penalizes a large residual greater than a small residual because of square
        MSLE.append(metrics.mean_squared_log_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true≥0, y_pred≥0； this metric penalizes an under-predicted estimate greater than an over-predicted estimate because of logarithm

    print('判断前的真实（及预测）序列对数:', len(y_true), '  判断后的真实（及预测）序列对数:', len(y_true_trun), '\n')
    print('原始的评估指标（越接近0越好）：')
    print('第一组，零次的相对性指标：', '\n', 'MAPE:', MAPE, '\n', 'SMAPE:', SMAPE, '\n', 'RMSPE:', RMSPE, '\n', 'MTD_p2:', MTD_p2)
    print('第二组，一次的绝对性指标：', '\n', 'EMLAE:', EMLAE, '\n', 'MALE:', MALE, '\n', 'MAE:', MAE, '\n', 'RMSE:', RMSE, '\n', 'MedAE:', MedAE, '\n', 'MTD_p1:', MTD_p1)
    print('第三组，二次的绝对性指标：', '\n', 'MSE:', MSE, '\n', 'MSLE:', MSLE, '\n')

    # 将各序列对的若干精度指标整合成各序列对的最终单一评价指标；序列对的数目必须≥2，否则归一化后各指标值均为1。
    # 将各精度指标在各自维度内进行数值变换：1.对各指标除以其均值，将任意数量级的指标转化为在1上下波动的数值。
    # 2.再对抹平数量级后的指标取幂函数，进一步缩小指标内数值的差距，保留代表优劣的方向性即可；原始指标内数值差异越大，所开次方根数越大，反之越小，可以避免指标间离群值的出现。
    # 3.再对list作归一化，将所有结果都转化为(0,1)之间的数，越趋近0越好，代表预测列越趋近真实序列；最终精度presion经过有偏向的加权后，也是(0,1)之间的数值。
    MAPE_1 = (MAPE / np.mean(MAPE)) / sum(MAPE / np.mean(MAPE))
    SMAPE_1 = (SMAPE / np.mean(SMAPE)) / sum(SMAPE / np.mean(SMAPE))
    RMSPE_1 = (RMSPE / np.mean(RMSPE)) / sum(RMSPE / np.mean(RMSPE))
    MTD_p2_1 = np.sqrt(MTD_p2 / np.mean(MTD_p2)) / sum(np.sqrt(MTD_p2 / np.mean(MTD_p2)))

    EMLAE_1 = np.sqrt(EMLAE / np.mean(EMLAE)) / sum(np.sqrt(EMLAE / np.mean(EMLAE)))
    MALE_1 = (MALE / np.mean(MALE)) / sum(MALE / np.mean(MALE))
    MAE_1 = np.sqrt(MAE / np.mean(MAE)) / sum(np.sqrt(MAE / np.mean(MAE)))
    RMSE_1 = np.sqrt(RMSE / np.mean(RMSE)) / sum(np.sqrt(RMSE / np.mean(RMSE)))
    MedAE_1 = np.sqrt(MedAE / np.mean(MedAE)) / sum(np.sqrt(MedAE / np.mean(MedAE)))
    MTD_p1_1 = np.sqrt(MTD_p1 / np.mean(MTD_p1)) / sum(np.sqrt(MTD_p1 / np.mean(MTD_p1)))

    MSE_1 = (MSE / np.mean(MSE))**(1/4) / sum((MSE / np.mean(MSE))**(1/4))
    MSLE_1 = np.sqrt(MSLE / np.mean(MSLE)) / sum(np.sqrt(MSLE / np.mean(MSLE)))

    print('数值变换后的评估指标（越接近0越好）：')
    print('第一组，零次的相对性指标：', '\n', 'MAPE:', MAPE_1, '\n', 'SMAPE:', SMAPE_1, '\n', 'RMSPE:', RMSPE_1, '\n', 'MTD_p2:', MTD_p2_1)
    print('第二组，一次的绝对性指标：', '\n', 'EMLAE:', EMLAE_1, '\n', 'MALE:', MALE_1, '\n', 'MAE:', MAE_1, '\n', 'RMSE:', RMSE_1, '\n', 'MedAE:', MedAE_1, '\n', 'MTD_p1:', MTD_p1_1)
    print('第三组，二次的绝对性指标：', '\n', 'MSE:', MSE_1, '\n', 'MSLE:', MSLE_1, '\n')

    precision = []
    for i in range(len(y_true_trun)):
        # 不用调和平均、几何平均，避免结果向极小值趋近；不用均方根，避免结果向极大值趋近；使用算术平均加权，权重可根据实际需求手动调整。
        precision.append(dyn_seri_weighted([MAPE_1[i], SMAPE_1[i], RMSPE_1[i], MTD_p2_1[i],
                                      EMLAE_1[i], MALE_1[i], MAE_1[i], RMSE_1[i], MedAE_1[i], MTD_p1_1[i],
                                      MSE_1[i], MSLE_1[i]], w=w))
    print('各序列对的最终精度（越接近0越好）：', '\n', np.array(precision), '\n')

    # 注意返回的各分量精度指标是未归一化前的数值，而最终precision是由各分量精度指标归一化后的数值算出的
    return precision, MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE, y_true_trun, y_pred_trun


def regression_accuracy_single(y_true, y_pred):
    """
    :param y_true: 一条真实序列，带索引的series，为了能对>0的数值的索引取交集；并与y_pred中的预测序列按顺序一一对应
    :param y_pred: 一条预测序列，带索引的series，为了能对>0的数值的索引取交集；并与y_true中的真实序列按顺序一一对应
    :return: 精度指标，按顺序分别是：MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE

    测试几种常用的评价序列对的精度指标
    原则：1.带平方项的指标会放大在正负1之外的残差的影响，而压缩在正负1之内的残差的影响，由于各指标越接近零越好，则会惩罚正负1之外的残差，偏离正负1越远，越受到惩罚；而奖励正负1之内的残差。
    2.作对数变换的指标会降低离群值的影响，相对于不带对数项的指标，会惩罚非离群值。因为在(1,+∞)区间内，y=lnx的一阶导数比y=x的一阶导数小，则y=lnx比y=x递增慢。
    3.ln(1/x)+len(x)=0，即对一个数（如x）取对数，与其倒数（1/x）取对数，互为相反数；当x>0，ln(1/x)+x-1≥0，lnx+(1/x)-1≥0，可由求导证明；在(0,4]的区间内，lnx的增长速度快于x**(1/2)，在[4,+∞)区间内，lnx的增长速度慢于x**(1/2)，可由求导证明。
    """

    # 为了统一下列12个精度指标的条件，在y_true和y_pred的序列对中，取大于0的对应点，即排除≤0的对应点；但不应取>0，可以取>0.01，否则若序列中存在大于0但非常接近0的数做分母，可能产生很大的值，不利于得到有效可用的精度值
    judge = (y_true > 0.01) & (y_pred > 0.01)
    # if sum(judge):
    y_true_trun = y_true[judge]
    y_pred_trun = y_pred[judge]
    if len(y_true_trun) < 2:
        raise Exception('y_true_trun与y_pred_trun中序列条数必须相等且≥2')  # 若序列对的数目小于2，则数值变换后的指标均为1

    plt.figure('finall inputs of regression_accuracy_single')
    ax = plt.subplot(1, 1, 1)
    # xlim = plt.gca().set_xlim(0, 1)  # xlim使图形按x轴上的点数充满横坐标
    y_true_trun.plot(ax=ax, legend=True)
    y_pred_trun.plot(ax=ax, legend=True)
    plt.show()
    print('实际输入的序列对：')
    print(y_true_trun, '\n', y_pred_trun, '\n')

    # 第一组，零次的相对性指标：
    MAPE = mape(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true != 0; no bias
    SMAPE = smape(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true + y_pred != 0; symmetric MAPE, no bias and more general, less susceptible to outliers than MAPE.
    RMSPE = eval_measures.rmspe(np.array(y_true_trun), np.array(y_pred_trun)) / 10  # y_true != 0; susceptible to outliers of deviation ratio, if more, RMSPE will be larger than MAPE.
    MTD_p2 = metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun), power=2) # y_pred > 0, y_true > 0; less susceptible to outliers than MAPE when y_pred / y_true > 1, nevertheless, more susceptible to outliers than MAPE when y_pred / y_true < 1

    # 第二组，一次的绝对性指标：
    EMLAE = emlae(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true, y_pred无限制条件; less susceptible to outliers of error than MAE, so this will penalize small deviation and award large deviation relative to MAE.
    MALE = male(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true, y_pred无限制条件;
    MAE = metrics.mean_absolute_error(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true, y_pred无限制条件； this metric has no penalty, no bias
    RMSE = eval_measures.rmse(np.array(y_true_trun), np.array(y_pred_trun))  # y_true, y_pred无限制条件；susceptible to outliers of error than MAE, so this will penalize large deviation and award small deviation relative to MAE.
    MedAE = metrics.median_absolute_error(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true, y_pred无限制条件； if len(y) is slightly large; won't be affected by outliers completely
    MTD_p1 = metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun), power=1)  # y_pred > 0, y_true ≥ 0; The higher `p` the less weight is given to extreme deviations between true and predicted targets.

    # 第三组，二次的绝对性指标：
    MSE = metrics.mean_squared_error(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true, y_pred无限制条件； this metric penalizes a large residual greater than a small residual because of square
    MSLE = metrics.mean_squared_log_error(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true≥0, y_pred≥0； this metric penalizes an under-predicted estimate greater than an over-predicted estimate because of logarithm

    print('判断前的真实（及预测）序列的点数:', len(y_true), '  判断后的真实（及预测）序列的点数:', len(y_true_trun), '\n')
    print('原始的评估指标（越接近0越好）：')
    print('第一组，零次的相对性指标：', '\n', 'MAPE:', MAPE, '\n', 'SMAPE:', SMAPE, '\n', 'RMSPE:', RMSPE, '\n', 'MTD_p2:', MTD_p2)
    print('第二组，一次的绝对性指标：', '\n', 'EMLAE:', EMLAE, '\n', 'MALE:', MALE, '\n', 'MAE:', MAE, '\n', 'RMSE:', RMSE, '\n', 'MedAE:', MedAE, '\n', 'MTD_p1:', MTD_p1)
    print('第三组，二次的绝对性指标：', '\n', 'MSE:', MSE, '\n', 'MSLE:', MSLE, '\n')

    # 无法得出最终precision，因为各指标的结果数量级不同，又没有其他序列对得出的指标结果作归一化消除数量级的影响
    return MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE, y_true_trun, y_pred_trun


@print_execute_time
def regression_correlaiton_pairs(y_true, y_pred):
    """
    :param y_true: 若干条真实序列组成的一个二维list或array或series；并与y_pred中的预测序列按顺序一一对应；y_true是历史上进模型之前的可能经过处理的真实值。
    :param y_pred: 若干条预测序列组成的一个二维list或array或series；并与y_true中的真实序列按顺序一一对应；y_pred是历史上该模型输出的预测值，或者经过补偿的预测值，总之是最终用于订货的预测值。
    y_true，y_pred也可以是需要进行相关性计算的多组序列对，其中每条序列中的元素个数是每个样本的特征数
    :return: 各个相关性指标，按顺序分别是：综合相关性指标，PR, SR, KT, WT, MGC
    """

    PR, SR, KT, WT, MGC = [], [], [], [], []  # 原始相关性指标
    PRmul, SRmul, KTmul, WTmul, MGCmul = [], [], [], [], []  # 考虑置信度的相关性指标
    y_true_trun, y_pred_trun = y_true, y_pred

    if len(y_true_trun[0]) == len(y_pred_trun[0]) > 1:  # 加上[0]，为了适应y_trun是一维和二维这两种情况

        for i in range(len(y_true_trun)):
            # 各个相关性函数使用如图所示形态的数据作为输入值
            plt.figure('finall input of regression_correlaiton_pairs, correlation fuctions use following scatters as {0}th inputs'.format(i))
            plt.scatter(y=y_true_trun[i], x=y_pred_trun[i])
            plt.xlabel('y_pred_trun[{0}]'.format(i))
            plt.ylabel('y_true_trun[{0}]'.format(i))
            plt.show()

        for i in range(len(y_true_trun)):
            if (len(y_true_trun[i]) < 5) or (len(y_pred_trun[i]) < 5):
                raise Exception('实际使用的序列对y_true_trun[{0}]与y_pred_trun[{1}]中，点数过少不具有统计意义，每条序列至少要≥5个点'.format(i, i))
            # PR当序列对在散点图中的斜率接近±1或0，各个点的斜率稍有变化时，容易识别为线性无关，此种情况应是较强的线性相关性；PR的特性表现为越趋近临界值（各点斜率趋近±1，0），鲁棒性越差
            PR.append(stats.pearsonr(x=y_true_trun[i], y=y_pred_trun[i]))  # Two-tailed p-value
            PRmul.append(PR[i][0] * (1 - PR[i][1]))
            # SR和PR有相似的上述鲁棒性问题，但鲁棒性稍好；KT和WT的鲁棒性也好于PR
            SR.append(stats.spearmanr(a=y_true_trun[i], b=y_pred_trun[i]))  # The two-sided p-value for a hypothesis test whose null hypothesis is that two sets of data are uncorrelated
            SRmul.append(SR[i][0] * (1 - SR[i][1]))
            KT.append(stats.kendalltau(x=y_true_trun[i], y=y_pred_trun[i]))  # The two-sided p-value for a hypothesis test whose null hypothesis is an absence of association, tau = 0.
            KTmul.append(KT[i][0] * (1 - KT[i][1]))
            WT.append(stats.weightedtau(x=y_true_trun[i], y=y_pred_trun[i]))
            WTmul.append(WT[i][0] * (1 - np.mean([PR[i][1], SR[i][1], KT[i][1]])))  # suppose the p-value is mean of others
            # MGC几乎没有上述鲁棒性问题，且reps越大，p-values越可信，但计算量越大
            MGC.append(
                stats.multiscale_graphcorr(x=np.array(y_true_trun[i]), y=np.array(y_pred_trun[i]), workers=1,
                                           reps=0, random_state=1)[
                :2])  # hardly affected by abnormal scatters (i.e. outliers); x and y must be ndarrays; MGC requires at least 5 samples to give reasonable results
            MGCmul.append(MGC[i][0] * (1 - np.mean([PR[i][1], SR[i][1], KT[i][1]])))  # suppose the p-value is mean of others

        print('各个原始的相关性指标（越接近1正相关性越强，越接近-1负相关性越强，越接近0相关性越弱）：', '\n', 'PR:', PR, '\n',
              'SR:', SR, '\n', 'KT:', KT, '\n', 'WT:', WT, '\n', 'MGC:', MGC, '\n')

        # 对各个相关性指标考虑置信度：p-value越大，越不能拒绝原假设（序列对无关），备择假设（序列对相关）越不可信，则相关系数乘以越小的系数，则认为序列对的实际相关性，跟计算出的相关系数比，越低
        metrics_raw = {'PRmul': PRmul, 'SRmul': SRmul, 'KTmul': KTmul, 'WTmul': WTmul,
                       'MGCmul': MGCmul}  # samples belong the row, metrics belong the colmun
        df_raw = pd.DataFrame(metrics_raw)
        # 对所有样本的各个指标取均值和中位数作为计算指标权重的基准；考虑到计算速度，没有对各列剔除最大最小值后再算均值
        df_raw['mean'] = (df_raw.mean(axis=1) + df_raw.median(axis=1)) / 2
        # 计算所有样本各种指标值与其基准值的距离
        for i in range(len(metrics_raw)):
            df_raw['D_{}'.format(df_raw.columns[i])] = abs(df_raw[df_raw.columns[i]] - df_raw['mean'])
        a = 0
        # 对所有样本的各个距离求和
        for i in range(len(metrics_raw)):
            a += df_raw['D_{}'.format(df_raw.columns[i])]
        df_raw['D_sum'] = a
        # 对距离的比例取相反数，使距离越大，其值越小，并为线性关系。再+1使距离比例为正，则归一化后为正确逻辑的权重；若不使距离比例为正，则归一化后仍是距离越大权重越大的错误逻辑
        for i in range(len(metrics_raw)):
            df_raw['w_{}_raw'.format(df_raw.columns[i])] = 1 - df_raw['D_{}'.format(df_raw.columns[i])] / df_raw[
                'D_sum']
        # 计算调整逻辑后的权重之和
        a = 0
        for i in range(len(metrics_raw)):
            a += df_raw['w_{}_raw'.format(df_raw.columns[i])]
        if a.mean() - len(metrics_raw) - 1 > 1e-10:
            raise Exception('权重变换有误')
        df_raw['w_sum'] = a
        # 计算调整逻辑后的各个权重
        for i in range(len(metrics_raw)):
            df_raw['weight_{}'.format(df_raw.columns[i])] = df_raw['w_{}_raw'.format(df_raw.columns[i])] / df_raw[
                'w_sum']
        # 用调整后的权重计算各个指标的加权平均
        a = 0
        for i in range(len(metrics_raw)):
            a += df_raw[df_raw.columns[i]] * df_raw['weight_{}'.format(df_raw.columns[i])]
        df_raw['correlation'] = a

        return df_raw[['correlation', 'PRmul', 'SRmul', 'KTmul', 'WTmul', 'MGCmul']], [y_true_trun, y_pred_trun]

    else:
        raise Exception('y_true_trun与y_pred_trun中序列条数必须相等且>1')


def regression_correlaiton_single(y_true, y_pred, type='high', w=(1,4,2,2,3)):
    """
    :param y_true: 一条真实序列，并与预测序列按顺序一一对应；y_true是历史上进模型之前的可能经过处理的真实值。
    :param y_pred: 一条预测序列，并与真实序列按顺序一一对应；y_pred是历史上该模型输出的预测值，或者经过补偿的预测值，总之是最终用于订货的预测值。
    y_true，y_pred也可以是需要进行相关性计算的多组序列对，其中每条序列中的元素个数是每个样本的特征数
    :type: 'high' includes MGC, which is the best but time costs, however, 'low' does not include.
    :return: 各个相关性指标，按顺序分别是：综合相关性指标，PR, SR, KT, WT, MGC(type='high')
    """

    y_true_trun, y_pred_trun = np.array(y_true), np.array(y_pred)
    error = 'no'

    # 各个相关性函数使用如图所示形态的数据作为输入值
    plt.figure('inputs of regression_correlaiton_single, correlation fuctions use following scatters as inputs')
    plt.scatter(y=y_true_trun, x=y_pred_trun)
    plt.xlabel('y_pred_trun')
    plt.ylabel('y_true_trun')
    plt.show()

    if type=='high' and len(y_true_trun) >= 5 and len(y_pred_trun) >= 5:
        try:
            # PR当序列对在散点图中的斜率接近±1或0，各个点的斜率稍有变化时，容易识别为线性无关，此种情况应是较强的线性相关性；PR的特性表现为越趋近临界值（各点斜率趋近±1，0），鲁棒性越差；以及对离群点的适应问题
            PR = stats.pearsonr(x=y_true_trun, y=y_pred_trun)
            PRmul = PR[0] * (1 - PR[1])
            # SR和PR有相似的上述鲁棒性问题，但鲁棒性稍好；KT和WT的鲁棒性也好于PR
            SR = stats.spearmanr(a=y_true_trun, b=y_pred_trun)
            SRmul = SR[0] * (1 - SR[1])
            KT = stats.kendalltau(x=y_true_trun, y=y_pred_trun)
            KTmul = KT[0] * (1 - KT[1])
            WT = stats.weightedtau(x=y_true_trun, y=y_pred_trun)
            WTmul = WT[0] * (1 - np.mean([PR[1], SR[1], KT[1]]))  # suppose the p-value is mean of others
            # MGC几乎没有上述鲁棒性问题，且reps越大，p-values越可信，但计算量越大
            # bacause MGC uses knn inside, if the length of series is longer, the iterations of knn will be much more, therefore the consumption of time will be much more.
            MGC = stats.multiscale_graphcorr(x=y_true_trun, y=y_pred_trun, workers=1, reps=0, random_state=1)[
                  :2]  # hardly affected by abnormal scatters (i.e. outliers); x and y must be ndarrays; MGC requires at least 5 samples to give reasonable results
            MGCmul = MGC[0] * (1 - np.mean([PR[1], SR[1], KT[1]]))  # suppose the p-value is 0.05

            # 对各个相关性指标考虑置信度：p-value越大，越不能拒绝原假设（序列对无关），备择假设（序列对相关）越不可信，则相关系数乘以越小的系数，则认为序列对的实际相关性，跟计算出的相关系数比，越低
            metrics_raw = np.array([PRmul, SRmul, KTmul, WTmul, MGCmul])
            # 对所有样本的各个指标取均值和中位数作为计算指标权重的基准；考虑到计算速度，没有对各列剔除最大最小值后再算均值
            corr_mean = (np.mean(metrics_raw) + np.median(metrics_raw)) / 2
            # 计算所有样本各种指标值与其基准值的距离
            df_distance = np.abs(metrics_raw - corr_mean)
            # 对所有样本的各个距离求和
            dist_sum = df_distance.sum()
            # 对距离的比例取相反数，使距离越大，其值越小，并为线性关系。再+1使距离比例为正，则归一化后为正确逻辑的权重；若不使距离比例为正，则归一化后仍是距离越大权重越大的错误逻辑
            df_wight = 1 - df_distance / dist_sum
            df_wight2 = df_wight * w  # 考虑各指标的权重
            # 计算调整逻辑后的权重之和
            df_wight2_sum = df_wight2.sum()
            # 计算调整逻辑后的各个权重，并用调整后的权重计算各个指标的加权平均
            wighted_corr = np.sum(metrics_raw * df_wight2 / df_wight2_sum)

            metrics_raw = {'PRmul': PRmul, 'SRmul': SRmul, 'KTmul': KTmul, 'WTmul': WTmul,
                           'MGCmul': MGCmul}  # samples belong the row, metrics belong the colmun
            df_raw = pd.DataFrame(metrics_raw, index={'corr of series'})
            df_raw['correlation'] = wighted_corr

            return df_raw[['correlation', 'PRmul', 'SRmul', 'KTmul', 'WTmul', 'MGCmul']], [y_true_trun, y_pred_trun]

        except Exception as e:
            error = e
            print('MGC error: ', error)
            metrics_raw = {'correlation': 0, 'PRmul': np.nan, 'SRmul': np.nan, 'KTmul': np.nan, 'WTmul': np.nan}  # samples belong the row, metrics belong the colmun
            df_raw = pd.DataFrame(metrics_raw, index={'corr of series'})
            return df_raw[['correlation', 'PRmul', 'SRmul', 'KTmul', 'WTmul']], [y_true_trun, y_pred_trun]

    if error != 'no' or type=='low' or (type=='high' and (len(y_true_trun) < 5 or len(y_pred_trun) < 5)):
        try:
            # PR当序列对在散点图中的斜率接近±1或0，各个点的斜率稍有变化时，容易识别为线性无关，此种情况应是较强的线性相关性；PR的特性表现为越趋近临界值（各点斜率趋近±1，0），鲁棒性越差
            PR = stats.pearsonr(x=y_true_trun, y=y_pred_trun)
            PRmul = PR[0] * (1 - PR[1])
            # SR和PR有相似的上述鲁棒性问题，但鲁棒性稍好；KT和WT的鲁棒性也好于PR
            SR = stats.spearmanr(a=y_true_trun, b=y_pred_trun)
            SRmul = SR[0] * (1 - SR[1])
            KT = stats.kendalltau(x=y_true_trun, y=y_pred_trun)
            KTmul = KT[0] * (1 - KT[1])
            WT = stats.weightedtau(x=y_true_trun, y=y_pred_trun)
            WTmul = WT[0] * (1 - np.mean([PR[1], SR[1], KT[1]]))  # suppose the p-value is mean of others

            # 对各个相关性指标考虑置信度：p-value越大，越不能拒绝原假设（序列对无关），备择假设（序列对相关）越不可信，则相关系数乘以越小的系数，则认为序列对的实际相关性，跟计算出的相关系数比，越低
            metrics_raw = np.array([PRmul, SRmul, KTmul, WTmul])
            # 对所有样本的各个指标取均值和中位数作为计算指标权重的基准；考虑到计算速度，没有对各列剔除最大最小值后再算均值
            corr_mean = (np.mean(metrics_raw) + np.median(metrics_raw)) / 2
            # 计算所有样本各种指标值与其基准值的距离
            df_distance = np.abs(metrics_raw - corr_mean)
            # 对所有样本的各个距离求和
            dist_sum = df_distance.sum()
            # 对距离的比例取相反数，使距离越大，其值越小，并为线性关系。再+1使距离比例为正，则归一化后为正确逻辑的权重；若不使距离比例为正，则归一化后仍是距离越大权重越大的错误逻辑
            df_wight = 1 - df_distance / dist_sum
            df_wight2 = df_wight * w[:-1]  # 考虑各指标的权重
            # 计算调整逻辑后的权重之和
            df_wight2_sum = df_wight2.sum()
            # 计算调整逻辑后的各个权重，并用调整后的权重计算各个指标的加权平均
            wighted_corr = np.sum(metrics_raw * df_wight2 / df_wight2_sum)

            # 对各个相关性指标考虑置信度：p-value越大，越不能拒绝原假设（序列对无关），备择假设（序列对相关）越不可信，则相关系数乘以越小的系数，则认为序列对的实际相关性，跟计算出的相关系数比，越低
            metrics_raw = {'PRmul': PRmul, 'SRmul': SRmul, 'KTmul': KTmul, 'WTmul': WTmul}  # samples belong the row, metrics belong the colmun
            df_raw = pd.DataFrame(metrics_raw, index={'corr of series'})
            df_raw['correlation'] = wighted_corr

            return df_raw[['correlation', 'PRmul', 'SRmul', 'KTmul', 'WTmul']], [y_true_trun, y_pred_trun]

        except Exception as e:
            print('sample error: ', e)
            metrics_raw = {'correlation': np.nan, 'PRmul': np.nan, 'SRmul': np.nan, 'KTmul': np.nan, 'WTmul': np.nan}  # samples belong the row, metrics belong the colmun
            df_raw = pd.DataFrame(metrics_raw, index={'corr of series'})
            return df_raw[['correlation', 'PRmul', 'SRmul', 'KTmul', 'WTmul']], [y_true_trun, y_pred_trun]
    else:
        raise Exception('type must be either low or high')


@print_execute_time
def correlation_population(pop1, pop2):
    """
    x,y: ndarray，每一行代表一个样本，每一列代表一个特征。
    return: 返回这两个ndarray的综合相关性和p-value，代表两个总体pop1和pop2间的相关程度。
    计算来自两个总体pop1和pop2的n个样本，所组成的两个ndarray间的综合相关性，每个ndarray有n行m列，其中n是从总体中随机抽取的样本数，m是每个样本的特征数；
    当workers=-1，每次会从两个ndarray中抽取k对样本，传到cpu的k个线程中计算每对样本各自的相关性，直到将ndarray中所有样本对计算完。
    """
    corr = stats.multiscale_graphcorr(x=pd.DataFrame(pop1, columns=list(pop1[0].index)).values
    , y=pd.DataFrame(pop2, columns=list(pop2[0].index)).values, workers=-1, reps=1000, random_state=1)[:2]  # hardly affected by abnormal scatters (i.e. outliers); x and y must be ndarrays; MGC requires at least 5 samples to give reasonable results

    return corr


@print_execute_time
def regression_evaluation_pairs(y_true, y_pred, w=(3,2,2,1, 1,1,1,3,1,1, 1,1, 1/2,1/3,1/2,2,1,1,2)):
    """
    :param y_true: 若干条真实序列组成的一个二维list或array或series，其中的每条真实序列必须是带索引的series，为了能对>0的数值的索引取交集；
    并与y_pred中的预测序列按顺序一一对应；y_true是历史上进模型之前的可能经过处理的真实值。
    :param y_pred: 若干条预测序列组成的一个二维list或array或series，其中的每条预测序列必须是带索引的series，为了能对>0的数值的索引取交集；
    并与y_true中的真实序列按顺序一一对应；y_pred是历史上该模型输出的预测值，或者经过补偿的预测值，总之是最终用于订货的预测值。
    :return: 精度指标，按顺序分别是：最终精度指标，MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE, VAR

    测试几种常用的评价序列对的精度指标
    原则：1.带平方项的指标会放大在正负1之外的残差的影响，而压缩在正负1之内的残差的影响，由于各指标越接近零越好，则会惩罚正负1之外的残差，偏离正负1越远，越受到惩罚；而奖励正负1之内的残差。
    2.作对数变换的指标会降低离群值的影响，相对于不带对数项的指标，会惩罚非离群值。因为在(1,+∞)区间内，y=lnx的一阶导数比y=x的一阶导数小，则y=lnx比y=x递增慢。
    3.ln(1/x)+len(x)=0，即对一个数（如x）取对数，与其倒数（1/x）取对数，互为相反数；当x>0，ln(1/x)+x-1≥0，lnx+(1/x)-1≥0，可由求导证明；在(0,4]的区间内，lnx的增长速度快于x**(1/2)，在[4,+∞)区间内，lnx的增长速度慢于x**(1/2)，可由求导证明。
    """

    MAPE, SMAPE, RMSPE, MTD_p2  = [], [], [], []  # 零次的相对性指标
    EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1 = [], [], [], [], [], []  # 一次的绝对性指标
    MSE, MSLE = [], []  # 二次的绝对性指标
    VAR, R2, PR, SR, KT, WT, MGC = [], [], [], [], [], [], []  # 相关性指标

    y_true_trun, y_pred_trun = [], []
    for i in range(len(y_true)):
        # 为了统一下列精度指标的条件，在y_true和y_pred的序列对中，取大于0的对应点，即排除≤0的对应点；但不应取>0，可以取>0.01，否则若序列中存在大于0但非常接近0的数做分母，可能产生很大的值，不利于得到有效可用的精度值
        judge = (y_true[i] > 0.01) & (y_pred[i] > 0.01)
        if sum(judge):
            y_true_trun.append(y_true[i][judge])
            y_pred_trun.append(y_pred[i][judge])
        else:
            continue

    if (len(y_true_trun) != len(y_pred_trun)) or (len(y_true_trun) < 2):
        raise Exception('y_true_trun与y_pred_trun中序列条数必须相等且≥2')  # 若序列对的数目小于2，则数值变换后的指标均为1

    plt.figure('finall inputs of accuracy functions (first three groups in regression_evaluation_pairs)')
    for i in range(len(y_true_trun)):
        # 前三组精度函数使用如图所示形态的数据作为输入值
        ax = plt.subplot(len(y_true_trun), 1, i+1)
        xlim = plt.gca().set_xlim(0, length[i]-1)  # xlim使图形按x轴上的点数充满横坐标
        y_true_trun[i].plot(ax=ax, legend=True)
        y_pred_trun[i].plot(ax=ax, legend=True)
        print('第{0}组实际输入的序列对：'.format(i))
        print(y_true_trun[i], '\n', y_pred_trun[i], '\n')
    plt.show()

    for i in range(len(y_true_trun)):
        # 第四组相关性函数使用如图所示形态的数据作为输入值。此for循环不能与上一个for循环合并，否则会错误调用plt。
        plt.figure('correlation fuctions (the 4th group in regression_evaluation_pairs) use following scatters as {0}th inputs'.format(i))
        plt.scatter(y=y_true_trun[i], x=y_pred_trun[i])
        plt.xlabel('y_pred_trun[{0}]'.format(i))
        plt.ylabel('y_true_trun[{0}]'.format(i))
        plt.show()

    for i in range(len(y_true_trun)):
        if (len(y_true_trun[i]) < 5) or (len(y_pred_trun[i]) < 5):
            raise Exception('实际使用的序列对y_true_trun[{0}]与y_pred_trun[{1}]中，点数过少不具有统计意义，每条序列至少要≥5个点'.format(i, i))
        # 第一组，零次的相对性指标：
        MAPE.append(mape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true != 0; no bias
        SMAPE.append(smape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true + y_pred != 0; symmetric MAPE, no bias and more general, less susceptible to outliers than MAPE.
        RMSPE.append(eval_measures.rmspe(np.array(y_true_trun[i]), np.array(y_pred_trun[i])) / 10)  # y_true != 0; susceptible to outliers of deviation ratio, if more, RMSPE will be larger than MAPE.
        MTD_p2.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=2)) # y_pred > 0, y_true > 0; less susceptible to outliers than MAPE when y_pred[i] / y_true[i] > 1, nevertheless, more susceptible to outliers than MAPE when y_pred[i] / y_true[i] < 1

        # 第二组，一次的绝对性指标：
        EMLAE.append(emlae(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件; less susceptible to outliers of error than MAE, so this will penalize small deviation and award large deviation relative to MAE.
        MALE.append(male(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件;
        MAE.append(metrics.mean_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric has no penalty, no bias
        RMSE.append(eval_measures.rmse(np.array(y_true_trun[i]), np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件；susceptible to outliers of error than MAE, so this will penalize large deviation and award small deviation relative to MAE.
        MedAE.append(metrics.median_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； if len(y) is slightly large; won't be affected by outliers completely
        MTD_p1.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=1))  # y_pred > 0, y_true ≥ 0; The higher `p` the less weight is given to extreme deviations between true and predicted targets.

        # 第三组，二次的绝对性指标：
        MSE.append(metrics.mean_squared_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric penalizes a large residual greater than a small residual because of square
        MSLE.append(metrics.mean_squared_log_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true≥0, y_pred≥0； this metric penalizes an under-predicted estimate greater than an over-predicted estimate because of logarithm

        # 第四组，相关性指标：
        # VAR作为相关性评价指标是最"僵硬的"，每个点的残差都相同时，VAR=1
        VAR.append(metrics.explained_variance_score(y_true=y_true_trun[i], y_pred=y_pred_trun[i]))  # y_true, y_pred无限制条件；但explained_variance_score为极大化目标函数，值域为(-∞, 1]，越趋近1越好；与其余的极小化目标函数相反，它们的因变量是越小越好。
        # R2主要是用来评估拟合程度，而不是预测准确度或相关性；当每个拟合值与真实值都相同时，R2=1
        R2.append(metrics.r2_score(y_true=y_true_trun[i], y_pred=y_pred_trun[i]))  # y_true, y_pred的series中，至少要有≥2个点，否则会返回nan；r2_score也为极大化目标函数，值域为(-∞, 1]，越趋近1越好；与其余的极小化目标函数相反，它们的因变量是越小越好。
        PR.append(stats.pearsonr(x=y_true_trun[i], y=y_pred_trun[i])[0])
        SR.append(stats.spearmanr(a=y_true_trun[i], b=y_pred_trun[i])[0])
        KT.append(stats.kendalltau(x=y_true_trun[i], y=y_pred_trun[i])[0])
        WT.append(stats.weightedtau(x=y_true_trun[i], y=y_pred_trun[i])[0])
        MGC.append(stats.multiscale_graphcorr(x=np.array(y_true_trun[i]), y=np.array(y_pred_trun[i]), reps=0, workers=1, random_state=1)[0])  # hardly affected by abnormal scatters (i.e. outliers); x and y must be ndarrays; MGC requires at least 5 samples to give reasonable results

    print('判断前的真实（及预测）序列对数:', len(y_true), '  判断后的真实（及预测）序列对数:', len(y_true_trun), '\n')
    print('原始的评估指标：')
    print('第一组，零次的相对性指标：', '\n', 'MAPE:', MAPE, '\n', 'SMAPE:', SMAPE, '\n', 'RMSPE:', RMSPE, '\n', 'MTD_p2:', MTD_p2)
    print('第二组，一次的绝对性指标：', '\n', 'EMLAE:', EMLAE, '\n', 'MALE:', MALE, '\n', 'MAE:', MAE, '\n', 'RMSE:', RMSE, '\n', 'MedAE:', MedAE, '\n', 'MTD_p1:', MTD_p1)
    print('第三组，二次的绝对性指标：', '\n', 'MSE:', MSE, '\n', 'MSLE:', MSLE)
    print('第四组，相关性指标：', '\n', 'VAR:', VAR, '\n', 'R2:', R2, '\n', 'PR:', PR, '\n', 'SR:', SR, '\n', 'KT:', KT, '\n', 'WT:', WT, '\n', 'MGC:', MGC, '\n')

    # 将各序列对的若干精度指标整合成各序列对的最终单一评价指标；序列对的数目必须≥2，否则归一化后各指标值均为1。
    # 将各精度指标在各自维度内进行数值变换：1.对各指标除以其均值，将任意数量级的指标转化为在1上下波动的数值。
    # 2.再对抹平数量级后的指标取幂函数，进一步缩小指标内数值的差距，保留代表优劣的方向性即可；原始指标内数值差异越大，所开次方根数越大，反之越小，可以避免指标间离群值的出现。
    # 3.再对list作归一化，将所有结果都转化为(0,1)之间的数，越趋近0越好，代表预测列越趋近真实序列；最终精度presion经过有偏向的加权后，也是(0,1)之间的数值。
    MAPE_1 = (MAPE / np.mean(MAPE)) / sum(MAPE / np.mean(MAPE))
    SMAPE_1 = (SMAPE / np.mean(SMAPE)) / sum(SMAPE / np.mean(SMAPE))
    RMSPE_1 = (RMSPE / np.mean(RMSPE)) / sum(RMSPE / np.mean(RMSPE))
    MTD_p2_1 = np.sqrt(MTD_p2 / np.mean(MTD_p2)) / sum(np.sqrt(MTD_p2 / np.mean(MTD_p2)))

    EMLAE_1 = np.sqrt(EMLAE / np.mean(EMLAE)) / sum(np.sqrt(EMLAE / np.mean(EMLAE)))
    MALE_1 = (MALE / np.mean(MALE)) / sum(MALE / np.mean(MALE))
    MAE_1 = np.sqrt(MAE / np.mean(MAE)) / sum(np.sqrt(MAE / np.mean(MAE)))
    RMSE_1 = np.sqrt(RMSE / np.mean(RMSE)) / sum(np.sqrt(RMSE / np.mean(RMSE)))
    MedAE_1 = np.sqrt(MedAE / np.mean(MedAE)) / sum(np.sqrt(MedAE / np.mean(MedAE)))
    MTD_p1_1 = np.sqrt(MTD_p1 / np.mean(MTD_p1)) / sum(np.sqrt(MTD_p1 / np.mean(MTD_p1)))

    MSE_1 = (MSE / np.mean(MSE))**(1/4) / sum((MSE / np.mean(MSE))**(1/4))
    MSLE_1 = np.sqrt(MSLE / np.mean(MSLE)) / sum(np.sqrt(MSLE / np.mean(MSLE)))

    VAR_1 = (-np.array(VAR)+1.01 + np.mean(-np.array(VAR)+1.01)) / sum(-np.array(VAR)+1.01 + np.mean(-np.array(VAR)+1.01))  # 因为VAR的取值范围是(-∞, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    VAR_1 = (VAR_1 / np.mean(VAR_1))**(1/1) / sum((VAR_1 / np.mean(VAR_1))**(1/1))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始VAR均为1时，sum(-np.array(VAR)+1 + np.mean(-np.array(VAR)+1))就会为0，则VAR_1就会为nan。
    R2_1 = (-np.array(R2)+1.01 + np.mean(-np.array(R2)+1.01)) / sum(-np.array(R2)+1.01 + np.mean(-np.array(R2)+1.01))  # 因为R2的取值范围是(-∞, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    R2_1 = (R2_1 / np.mean(R2_1))**(1/1) / sum((R2_1 / np.mean(R2_1))**(1/1))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始VAR均为1时，sum(-np.array(R2)+1 + np.mean(-np.array(R2)+1))就会为0，则R2_1就会为nan。
    PR_1 = (-np.array(PR)+1.01 + np.mean(-np.array(PR)+1.01)) / sum(-np.array(PR)+1.01 + np.mean(-np.array(PR)+1.01))  # 因为PR的取值范围是[-1, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    PR_1 = (PR_1 / np.mean(PR_1))**(1/1) / sum((PR_1 / np.mean(PR_1))**(1/1))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始PR均为1时，sum(-np.array(PR)+1 + np.mean(-np.array(PR)+1))就会为0，则PR_1就会为nan。
    SR_1 = (-np.array(SR)+1.01 + np.mean(-np.array(SR)+1.01)) / sum(-np.array(SR)+1.01 + np.mean(-np.array(SR)+1.01))  # 因为SR的取值范围是[-1, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    SR_1 = (SR_1 / np.mean(SR_1))**(1/1) / sum((SR_1 / np.mean(SR_1))**(1/1))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始SR均为1时，sum(-np.array(SR)+1 + np.mean(-np.array(SR)+1))就会为0，则SR_1就会为nan。
    KT_1 = (-np.array(KT)+1.01 + np.mean(-np.array(KT)+1.01)) / sum(-np.array(KT)+1.01 + np.mean(-np.array(KT)+1.01))  # 因为KT的取值范围是[-1, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    KT_1 = (KT_1 / np.mean(KT_1))**(1/1) / sum((KT_1 / np.mean(KT_1))**(1/1))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始KT均为1时，sum(-np.array(KT)+1 + np.mean(-np.array(KT)+1))就会为0，则KT_1就会为nan。
    WT_1 = (-np.array(WT)+1.01 + np.mean(-np.array(WT)+1.01)) / sum(-np.array(WT)+1.01 + np.mean(-np.array(WT)+1.01))  # 因为WT的取值范围是[-1, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    WT_1 = (WT_1 / np.mean(WT_1))**(1/1) / sum((WT_1 / np.mean(WT_1))**(1/1))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始WT均为1时，sum(-np.array(WT)+1 + np.mean(-np.array(WT)+1))就会为0，则WT_1就会为nan。
    MGC_1 = (-np.array(MGC)+1.01 + np.mean(-np.array(MGC)+1.01)) / sum(-np.array(MGC)+1.01 + np.mean(-np.array(MGC)+1.01))  # 因为MGC的取值范围是[-1, 1]，越趋近1越好，可看作极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数。
    MGC_1 = (MGC_1 / np.mean(MGC_1))**(1/1) / sum((MGC_1 / np.mean(MGC_1))**(1/1))                                      # 但不能+1，可以加比1多一点点的任何数，如1.01，否则当原始MGC均为1时，sum(-np.array(MGC)+1 + np.mean(-np.array(MGC)+1))就会为0，则MGC_1就会为nan。

    print('数值变换后的评估指标：')
    print('第一组，零次的相对性指标：', '\n', 'MAPE:', MAPE_1, '\n', 'SMAPE:', SMAPE_1, '\n', 'RMSPE:', RMSPE_1, '\n', 'MTD_p2:', MTD_p2_1)
    print('第二组，一次的绝对性指标：', '\n', 'EMLAE:', EMLAE_1, '\n', 'MALE:', MALE_1, '\n', 'MAE:', MAE_1, '\n', 'RMSE:', RMSE_1, '\n', 'MedAE:', MedAE_1, '\n', 'MTD_p1:', MTD_p1_1)
    print('第三组，二次的绝对性指标：', '\n', 'MSE:', MSE_1, '\n', 'MSLE:', MSLE_1)
    print('第四组，相关性指标：', '\n', 'VAR:', VAR_1, '\n', 'R2:', R2_1, '\n', 'PR:', PR_1, '\n', 'SR:', SR_1, '\n', 'KT:', KT_1, '\n', 'WT:', WT_1, '\n', 'MGC:', MGC_1, '\n')

    evaluation = []
    for i in range(len(y_true_trun)):
        # 不用调和平均、几何平均，避免结果向极小值趋近；不用均方根，避免结果向极大值趋近；使用算术平均加权，权重可根据实际需求手动调整。
        evaluation.append(dyn_seri_weighted([MAPE_1[i], SMAPE_1[i], RMSPE_1[i], MTD_p2_1[i],
                                      EMLAE_1[i], MALE_1[i], MAE_1[i], RMSE_1[i], MedAE_1[i], MTD_p1_1[i],
                                      MSE_1[i], MSLE_1[i],
                                      VAR_1[i], R2_1[i], PR_1[i], SR_1[i], KT_1[i], WT_1[i], MGC_1[i]],
                                      w=w))
    print('各序列对的最终精度：', '\n', np.array(evaluation), '\n')

    # 注意返回的各分量指标是未数值变换前的结果，而最终precision是由各分量指标经数值变换后的结果加权算出的
    return evaluation, MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE, VAR, R2, PR, SR, KT, WT, MGC, y_true_trun, y_pred_trun


def regression_evaluation_single(y_true, y_pred):
    """
    :param y_true: 若干条真实序列组成的一个二维list或array或series，其中的每条真实序列必须是带索引的series，为了能对>0的数值的索引取交集；
    并与y_pred中的预测序列按顺序一一对应；y_true是历史上进模型之前的可能经过处理的真实值。
    :param y_pred: 若干条预测序列组成的一个二维list或array或series，其中的每条预测序列必须是带索引的series，为了能对>0的数值的索引取交集；
    并与y_true中的真实序列按顺序一一对应；y_pred是历史上该模型输出的预测值，或者经过补偿的预测值，总之是最终用于订货的预测值。
    :return: 精度指标，按顺序分别是：最终精度指标，MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE, VAR

    测试几种常用的评价序列对的精度指标
    原则：1.带平方项的指标会放大在正负1之外的残差的影响，而压缩在正负1之内的残差的影响，由于各指标越接近零越好，则会惩罚正负1之外的残差，偏离正负1越远，越受到惩罚；而奖励正负1之内的残差。
    2.作对数变换的指标会降低离群值的影响，相对于不带对数项的指标，会惩罚非离群值。因为在(1,+∞)区间内，y=lnx的一阶导数比y=x的一阶导数小，则y=lnx比y=x递增慢。
    3.ln(1/x)+len(x)=0，即对一个数（如x）取对数，与其倒数（1/x）取对数，互为相反数；当x>0，ln(1/x)+x-1≥0，lnx+(1/x)-1≥0，可由求导证明；在(0,4]的区间内，lnx的增长速度快于x**(1/2)，在[4,+∞)区间内，lnx的增长速度慢于x**(1/2)，可由求导证明。
    """

    # 为了统一下列精度指标的条件，在y_true和y_pred的序列对中，取大于0的对应点，即排除≤0的对应点；但不应取>0，可以取>0.01，否则若序列中存在大于0但非常接近0的数做分母，可能产生很大的值，不利于得到有效可用的精度值
    judge = (y_true > 0.01) & (y_pred > 0.01)
    # if sum(judge):
    y_true_trun = y_true[judge]
    y_pred_trun = y_pred[judge]
    if len(y_true_trun) < 2:
        raise Exception('y_true_trun与y_pred_trun中序列条数必须相等且≥2')  # 若序列对的数目小于2，则数值变换后的指标均为1

    plt.figure('finall inputs of accuracy functions (first three groups in regression_evaluation_single)')
    # 前三组精度函数使用如图所示形态的数据作为输入值
    ax = plt.subplot(1, 1, 1)
    # xlim = plt.gca().set_xlim(0, length-1)  # xlim使图形按x轴上的点数充满横坐标
    y_true_trun.plot(ax=ax, legend=True)
    y_pred_trun.plot(ax=ax, legend=True)
    plt.show()
    print('实际输入的序列对：')
    print(y_true_trun, '\n', y_pred_trun, '\n')

    # 第四组相关性函数使用如图所示形态的数据作为输入值。此for循环不能与上一个for循环合并，否则会错误调用plt。
    plt.figure('correlation fuctions (the 4th group in regression_evaluation_single) use following scatters as {0}th inputs'.format(i))
    plt.scatter(y=y_true_trun, x=y_pred_trun)
    plt.xlabel('y_pred_trun')
    plt.ylabel('y_true_trun')
    plt.show()

    if (len(y_true_trun) < 5) or (len(y_pred_trun) < 5):
        raise Exception('实际使用的序列对y_true_trun与y_pred_trun中，点数过少不具有统计意义，每条序列至少要≥5个点')
    # 第一组，零次的相对性指标：
    MAPE = mape(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true != 0; no bias
    SMAPE = smape(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true + y_pred != 0; symmetric MAPE, no bias and more general, less susceptible to outliers than MAPE.
    RMSPE = eval_measures.rmspe(np.array(y_true_trun), np.array(y_pred_trun)) / 10  # y_true != 0; susceptible to outliers of deviation ratio, if more, RMSPE will be larger than MAPE.
    MTD_p2 = metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun), power=2) # y_pred > 0, y_true > 0; less susceptible to outliers than MAPE when y_pred / y_true > 1, nevertheless, more susceptible to outliers than MAPE when y_pred / y_true < 1

    # 第二组，一次的绝对性指标：
    EMLAE = emlae(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true, y_pred无限制条件; less susceptible to outliers of error than MAE, so this will penalize small deviation and award large deviation relative to MAE.
    MALE = male(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true, y_pred无限制条件;
    MAE = metrics.mean_absolute_error(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true, y_pred无限制条件； this metric has no penalty, no bias
    RMSE = eval_measures.rmse(np.array(y_true_trun), np.array(y_pred_trun))  # y_true, y_pred无限制条件；susceptible to outliers of error than MAE, so this will penalize large deviation and award small deviation relative to MAE.
    MedAE = metrics.median_absolute_error(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true, y_pred无限制条件； if len(y) is slightly large; won't be affected by outliers completely
    MTD_p1 = metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun), power=1)  # y_pred > 0, y_true ≥ 0; The higher `p` the less weight is given to extreme deviations between true and predicted targets.

    # 第三组，二次的绝对性指标：
    MSE = metrics.mean_squared_error(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true, y_pred无限制条件； this metric penalizes a large residual greater than a small residual because of square
    MSLE = metrics.mean_squared_log_error(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true≥0, y_pred≥0； this metric penalizes an under-predicted estimate greater than an over-predicted estimate because of logarithm

    # 第四组，相关性指标：
    # VAR作为相关性评价指标是最"僵硬的"，每个点的残差都相同时，VAR=1
    VAR = metrics.explained_variance_score(y_true=y_true_trun, y_pred=y_pred_trun)  # y_true, y_pred无限制条件；但explained_variance_score为极大化目标函数，值域为(-∞, 1]，越趋近1越好；与其余的极小化目标函数相反，它们的因变量是越小越好。
    # R2主要是用来评估拟合程度，而不是预测准确度或相关性；当每个拟合值与真实值都相同时，R2=1
    R2 = metrics.r2_score(y_true=y_true_trun, y_pred=y_pred_trun)  # y_true, y_pred的series中，至少要有≥2个点，否则会返回nan；r2_score也为极大化目标函数，值域为(-∞, 1]，越趋近1越好；与其余的极小化目标函数相反，它们的因变量是越小越好。
    PR = stats.pearsonr(x=y_true_trun, y=y_pred_trun)[0]
    SR = stats.spearmanr(a=y_true_trun, b=y_pred_trun)[0]
    KT = stats.kendalltau(x=y_true_trun, y=y_pred_trun)[0]
    WT = stats.weightedtau(x=y_true_trun, y=y_pred_trun)[0]
    MGC = stats.multiscale_graphcorr(x=np.array(y_true_trun), y=np.array(y_pred_trun), reps=0, workers=1, random_state=1)[0]  # hardly affected by abnormal scatters (i.e. outliers); x and y must be ndarrays; MGC requires at least 5 samples to give reasonable results

    print('判断前的真实（及预测）序列对数:', len(y_true), '  判断后的真实（及预测）序列对数:', len(y_true_trun), '\n')
    print('原始的评估指标：')
    print('第一组，零次的相对性指标：', '\n', 'MAPE:', MAPE, '\n', 'SMAPE:', SMAPE, '\n', 'RMSPE:', RMSPE, '\n', 'MTD_p2:', MTD_p2)
    print('第二组，一次的绝对性指标：', '\n', 'EMLAE:', EMLAE, '\n', 'MALE:', MALE, '\n', 'MAE:', MAE, '\n', 'RMSE:', RMSE, '\n', 'MedAE:', MedAE, '\n', 'MTD_p1:', MTD_p1)
    print('第三组，二次的绝对性指标：', '\n', 'MSE:', MSE, '\n', 'MSLE:', MSLE)
    print('第四组，相关性指标：', '\n', 'VAR:', VAR, '\n', 'R2:', R2, '\n', 'PR:', PR, '\n', 'SR:', SR, '\n', 'KT:', KT, '\n', 'WT:', WT, '\n', 'MGC:', MGC, '\n')

    # 无法得出最终precision，因为各指标的结果数量级不同，又没有其他序列对得出的指标结果作归一化消除数量级的影响
    return MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE, VAR, R2, PR, SR, KT, WT, MGC, y_true_trun, y_pred_trun


def accuracy_single(y_true, y_pred):
    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    if sum(y_true) == 0:
        return None
    elif np.count_nonzero(y_true) < 2 / 3 * len(y_true) or len(y_true) == 1:
        y_true, y_pred = sum(y_true), sum(y_pred)
        if y_true >= y_pred:  # 当预测值小于真实值，直接计算精度，无需通过偏差来计算精度
            accuracy = y_pred / y_true
            return accuracy
        else:
            APE = abs((y_true - y_pred) / y_true)
            SAPE = 2 * abs((y_true - y_pred) / (y_true + y_pred))
            bias = dyn_seri_weighted([APE, SAPE], type='gmean', w=[2, 1])
            if bias <= 0.4:  # 假定偏差bias<=40%为合格，此阶段偏差与精度为线性反向关系；
                accuracy = 1 - bias
            else:  # 当偏差超过40%，由于偏差也可能超过1，则将其与精度压缩为指数关系，使精度始终为正
                accuracy = np.exp(np.log(1 - 0.4) / -0.4) ** (-bias)
            return accuracy
    else:  # 当序列中非零值超过2/3时，则剔除对应值，逐点计算精度，使粒度最细
        judge = y_true > 0.01
        y_true = y_true[judge]
        y_pred = y_pred[judge]
        MAPE = sum(abs((y_true - y_pred) / y_true)) / len(y_true)
        SMAPE = sum(abs(2 * (y_true - y_pred) / (y_true + y_pred))) / len(y_true)
        RMSPE = np.sqrt(sum(((y_true - y_pred) / y_true) ** 2) / len(y_true))
        bias = dyn_seri_weighted([MAPE, SMAPE, RMSPE], type='gmean', w=[2, 1, 0.5])
        if bias <= 0.4:
            accuracy = 1 - bias
        else:
            accuracy = np.exp(np.log(1 - 0.4) / -0.4) ** (-bias)
        return accuracy


results_v1_all = regression_accuracy_pairs(y_true=y_input_mul_actual[:], y_pred=y_input_mul_pred[:])
results_v1 = pd.DataFrame(results_v1_all[:-2], index=['precision',
                                           'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
                                           'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
                                           'MSE', 'MSLE'])
print('指标个数：', len(results_v1))
print(results_v1, '\n')

results_v2_all = regression_accuracy_single(y_true=y_input_mul_actual[-1], y_pred=y_input_mul_pred[-1])
results_v2 = pd.DataFrame(results_v2_all[:-2],
                          index=[
                                'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
                                'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
                                'MSE', 'MSLE'])
print('指标个数：', len(results_v2))
print(results_v2, '\n')

results_v3_all = regression_correlaiton_pairs(y_true=y_input_mul_actual[:], y_pred=y_input_mul_pred[:])
print('指标个数：', len(results_v3_all[0].columns))
print('序列对的最终相关性指标（越接近1正相关性越强，越接近-1负相关性越强，越接近0相关性越弱）（未归一化）:', '\n', results_v3_all[0], '\n')
#
results_v4_all = regression_correlaiton_single(y_true=y_input_mul_actual[-1], y_pred=y_input_mul_pred[-1], type='high')
print('指标个数：', len(results_v4_all[0].columns))
print('序列对的最终相关性指标（越接近1正相关性越强，越接近-1负相关性越强，越接近0相关性越弱）（未归一化）:', '\n', results_v4_all[0], '\n')

results_v5_all = regression_evaluation_pairs(y_true=y_input_mul_actual[:], y_pred=y_input_mul_pred[:])
results_v5 = pd.DataFrame(results_v5_all[:-2], index=['evaluation',
                                           'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
                                           'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
                                           'MSE', 'MSLE',
                                           'VAR', 'R2', 'PR', 'SR', 'KT', 'WT', 'MGC'])
print('指标个数：', len(results_v5))
print(results_v5, '\n')
#
results_v6_all = regression_evaluation_single(y_true=y_input_mul_actual[-1], y_pred=y_input_mul_pred[-1])
results_v6 = pd.DataFrame(results_v6_all[:-2], index=[
                                           'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
                                           'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
                                           'MSE', 'MSLE',
                                           'VAR', 'R2', 'PR', 'SR', 'KT', 'WT', 'MGC'])
print('指标个数：', len(results_v6))
print(results_v6, '\n')

results_v7 = correlation_population(pop1=y_input_mul_actual[:], pop2=y_input_mul_pred[:])
print('\n''两组ndarray的综合相关性：', results_v7[0], '\n', 'p-values:', results_v7[1], '\n')

################################################################################################################
# 将评估函数结果用于动态加权的使用方法：
# 使用步骤：1.将每个门店单品，两种模型的一段历史区间，预测序列和真实序列的两组序列对，输入regression_evaluation，得到两个评估指标；
# 2.对这两个指标进行如下操作，得到每个门店单品的两个模型在下一个月所要采用的权重w；
# w1相反数权重，当results_v5.loc['evaluation']中元素个数为2时，生成的w1为对称关系，指标0.4:0.6变为权重0.6:0.4；当元素个数增加时，权重间的比例会被压缩，这对于指标到权重的变换是有益的。
w1 = (results_v5.loc['evaluation'].sum()-results_v5.loc['evaluation']) / (results_v5.loc['evaluation'].sum()-results_v5.loc['evaluation']).sum()
print('w:', w1, '\n', sum(w1))
# w2倒数权重，在整个定义域内指标和权重都是对称关系，例如指标为[0.1,0.2,0.3,0.4]，则权重为对称的[0.48,0.24,0.16,0.12]，但指标的倍数关系会大于趋近程度的倍数关系，所以是更极端的。
w2 = (results_v5.loc['evaluation'].sum() / results_v5.loc['evaluation']) / (results_v5.loc['evaluation'].sum() / results_v5.loc['evaluation']).sum()
print('w:', w2, '\n', sum(w2))
# 因为regression_evaluation返回的最终评估指标precision是(0,1)之间的值，越趋近0则模型预测结果越好，所以需要使用一个递减函数对precision做变换，才能得到真正能使用的权重。
# 这里采用y=1-x，而不用y=1/x，可以避免当x较小时，1/x被放大过多，且一点微小的扰动都会对1/x产生较大影响的不利效应。
# 3.预测下一个月的新数据时，将每个门店单品两个模型的预测序列和各自权重w输入dyn_df_weighted，得到一条预测序列，就是最终发布的预测值。
###############################################################################################################

print('\n', '预测精度：', accuracy_single(y_input_mul_actual[-1], y_input_mul_pred[-1]))
