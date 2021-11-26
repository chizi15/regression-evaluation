import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn import metrics
from statsmodels.tools import eval_measures
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
plt.rc('font', size=10)
pd.set_option('display.max_columns', None)

# ###########---------------set up and plot input data-----------------######################
base_value = 10  # 设置level、trend、season项的基数
steps_day, steps_week = 1, 1
length = [steps_day*20+steps_day, steps_week*20+steps_week, steps_week*20+steps_week]  # 代表周、日序列对的长度

weights = []
for i in range(-base_value + 1, 1):
    weights.append(0.5 ** i)  # 设置y_level项随机序列的权重呈递减指数分布，底数越小，y_level中较小值所占比例越大。
weights = np.array(weights)


# #########################################################--构造乘法周期性时间序列，模拟真实销售；外层是list，内层的每一条序列是series
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

    y_level_actual[i] = pd.Series(y_level_actual[i]).rename('y_level_actual')
    y_trend_actual[i] = pd.Series(y_trend_actual[i]).rename('y_trend_actual')
    y_season_actual[i] = pd.Series(y_season_actual[i]).rename('y_season_actual')
    y_noise_actual[i] = pd.Series(y_noise_actual[i]).rename('y_noise_actual')
    y_input_mul_actual[i] = pd.Series(y_input_mul_actual[i]).rename('y_input_mul_actual')
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

    y_level_pred[i] = pd.Series(y_level_pred[i]).rename('y_level_pred')
    y_trend_pred[i] = pd.Series(y_trend_pred[i]).rename('y_trend_pred')
    y_season_pred[i] = pd.Series(y_season_pred[i]).rename('y_season_pred')
    y_noise_pred[i] = pd.Series(y_noise_pred[i]).rename('y_noise_pred')
    y_input_mul_pred[i] = pd.Series(y_input_mul_pred[i]).rename('y_input_mul_pred')
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


def dyn_seri_weighted(seri, type=None, w=None, initial=1, r=2, d=1):
    """
    传入一维数组seri，可以是series,array,list,tuple；若type='geometric'或'arithmetic'，且输入了w，则w不起作用；若不输入权重，则根据seri的长度动态计算基于几何级数或算数级数再作归一化的权重，再做算术平均；
    也可人为输入权重做算术平均；若不输入type和w，则进行简单算数平均；因为使用np.dot，则seri索引越小，权重越大；将seri各点与权重相乘再相加，得到一个最终点。
    :param seri: 需要进行加权变成一个点的一维数组
    :param type: 采用几何级数或算数级数进行加权，或人为指定权重，或默认权重相等，type = 'geometric'或'arithmetic'或None；若type='geometric'或'arithmetic'，且输入了w，则w不起作用。
    :param w: 一维的权重系数，可以是series,array,list,tuple；若手动输入，其长度必须和一维数组seri（即序列点数）相等
    :param r: 指定几何级数分母的公比
    :param d: 指定算数级数分母的公差
    :param initial: 指定算数级数分母的初始值
    :return: seri各点与权重w相乘再相加，返回的一个加权后的最终点
    """
    if type not in ['geometric', 'arithmetic', None]:
        raise Exception('type must be one of geometric, arithmetic or None')
    if type is not None:
        w = list()
        if type == 'geometric':
            for i in range(len(seri)):
                w.append(initial * (1 / r) ** i)  # 生成首项是initial，公比是(1/r)的几何级数作权重
        else:
            for i in range(len(seri)):
                w.append(1 / (initial + d * i))  # 生成首项是initial，公差是d的算术级数，再做倒数作为权重
        w = np.array(w) / sum(w)
    elif (type is None) and (w is None):
        w = np.ones(len(seri)) / sum(np.ones(len(seri)))  # 生成均等权重
    elif (type is None) and (w is not None) and (len(w) == len(seri)):
        w = np.array(w) / sum(w)  # 自定义权重
    else:
        raise Exception('手动输入的权重长度必须和一维数组长度（即序列点数）相等')
    if abs(sum(w)-1) > 0.001:
        raise Exception('weights are not useable')
    return np.dot(np.array(seri), w)


def dyn_df_weighted(df, type=None, w=None, initial=1, r=2, d=1):
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


def regression_accuracy_pairs(y_true, y_pred):
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
                                      MSE_1[i], MSLE_1[i]], w=[3,2,2,1, 1,1,1,3,1,1, 1,1]))
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
    if sum(judge):
        y_true_trun = y_true[judge]
        y_pred_trun = y_pred[judge]

    if (len(y_true_trun) != len(y_pred_trun)) or (len(y_true_trun) < 2):
        raise Exception('y_true_trun与y_pred_trun中序列条数必须相等且≥2')  # 若序列对的数目小于2，则数值变换后的指标均为1

    plt.figure('finall inputs of regression_accuracy_single')
    ax = plt.subplot(1, 1, 1)
    # xlim = plt.gca().set_xlim(0, 1)  # xlim使图形按x轴上的点数充满横坐标
    y_true_trun.plot(ax=ax, legend=True)
    y_pred_trun.plot(ax=ax, legend=True)
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


def regression_correlaiton_pairs(y_true, y_pred):
    """
    :param y_true: 若干条真实序列组成的一个二维list或array或series，其中的每条真实序列必须是带索引的series，为了能对>0的数值的索引取交集；
    并与y_pred中的预测序列按顺序一一对应；y_true是历史上进模型之前的可能经过处理的真实值。
    :param y_pred: 若干条预测序列组成的一个二维list或array或series，其中的每条预测序列必须是带索引的series，为了能对>0的数值的索引取交集；
    并与y_true中的真实序列按顺序一一对应；y_pred是历史上该模型输出的预测值，或者经过补偿的预测值，总之是最终用于订货的预测值。
    :return: 各个相关性指标，按顺序分别是：综合相关性指标，PR, SR, KT, WT, MGC
    """

    PR, SR, KT, WT, MGC = [], [], [], [], []  # 相关性指标
    PRmul, SRmul, KTmul, WTmul, MGCmul = [], [], [], [], []
    y_true_trun, y_pred_trun = y_true, y_pred

    if len(y_true_trun[0]) == len(y_pred_trun[0]) > 1:  # 加上[0]，为了适应y_trun是一维和二维这两种情况

        for i in range(len(y_true_trun)):
            # 第四组相关性函数使用如图所示形态的数据作为输入值。此for循环不能与上一个for循环合并，否则会错误调用plt。
            plt.figure('finall input of regression_correlaiton_pairs, correlation fuctions use following scatters as {0}th inputs'.format(i))
            plt.scatter(y=y_true_trun[i], x=y_pred_trun[i])
            plt.xlabel('y_pred_trun[{0}]'.format(i))
            plt.ylabel('y_true_trun[{0}]'.format(i))

        for i in range(len(y_true_trun)):
            if (len(y_true_trun[i]) < 5) or (len(y_pred_trun[i]) < 5):
                raise Exception('实际使用的序列对y_true_trun[{0}]与y_pred_trun[{1}]中，点数过少不具有统计意义，每条序列至少要≥5个点'.format(i, i))
            # 各个相关性指标，考虑置信度，p-value越大，越不可信，则相关系数乘以越小的系数，则认为序列对越不相关：
            PR.append(stats.pearsonr(x=y_input_mul_actual[i], y=y_input_mul_pred[i]))
            PRmul.append(PR[i][0] * (1 - PR[i][1]))
            SR.append(stats.spearmanr(a=y_input_mul_actual[i], b=y_input_mul_pred[i]))
            SRmul.append(SR[i][0] * (1 - SR[i][1]))
            KT.append(stats.kendalltau(x=y_input_mul_actual[i], y=y_input_mul_pred[i]))
            KTmul.append(KT[i][0] * (1 - KT[i][1]))
            WT.append(stats.weightedtau(x=y_input_mul_actual[i], y=y_input_mul_pred[i]))
            WTmul.append(WT[i][0] * 0.95)  # suppose the p-value is 0.05
            MGC.append(
                stats.multiscale_graphcorr(x=y_input_mul_actual[i].values, y=y_input_mul_pred[i].values, workers=-1,
                                           reps=0)[
                :2])  # hardly affected by abnormal scatters (i.e. outliers); x and y must be ndarrays; MGC requires at least 5 samples to give reasonable results
            MGCmul.append(MGC[i][0] * 0.95)  # suppose the p-value is 0.05

        print('判断前的真实（及预测）序列对数:', len(y_true), '  判断后的真实（及预测）序列对数:', len(y_true_trun), '\n')
        print('原始相关性指标：')
        print('各个相关性指标（越接近1正相关性越强，越接近-1负相关性越强，越接近0相关性越弱）：', '\n', 'PR:', PR, '\n',
              'SR:', SR, '\n', 'KT:', KT, '\n', 'WT:', WT, '\n', 'MGC:', MGC, '\n')

        metrics_raw = {'PRmul': PRmul, 'SRmul': SRmul, 'KTmul': KTmul, 'WTmul': WTmul,
                       'MGCmul': MGCmul}  # samples belong row, metrics belong colmun
        df_raw = pd.DataFrame(metrics_raw)
        df_raw['mean'] = (df_raw.mean(axis=1) + df_raw.median(axis=1)) / 2
        for i in range(len(metrics_raw)):
            df_raw['D_{}'.format(df_raw.columns[i])] = abs(df_raw[df_raw.columns[i]] - df_raw['mean'])
        a = 0
        for i in range(len(metrics_raw)):
            a += df_raw['D_{}'.format(df_raw.columns[i])]
        df_raw['D_sum'] = a
        for i in range(len(metrics_raw)):
            df_raw['w_{}_raw'.format(df_raw.columns[i])] = 1 - df_raw['D_{}'.format(df_raw.columns[i])] / df_raw[
                'D_sum']
        a = 0
        for i in range(len(metrics_raw)):
            a += df_raw['w_{}_raw'.format(df_raw.columns[i])]
        if a.mean() - len(metrics_raw) - 1 > 1e-10:
            raise Exception('权重变换有误')
        df_raw['w_sum'] = a
        for i in range(len(metrics_raw)):
            df_raw['weight_{}'.format(df_raw.columns[i])] = df_raw['w_{}_raw'.format(df_raw.columns[i])] / df_raw[
                'w_sum']
        a = 0
        for i in range(len(metrics_raw)):
            a += df_raw[df_raw.columns[i]] * df_raw['weight_{}'.format(df_raw.columns[i])]
        df_raw['correlation'] = a

        return df_raw[['correlation', 'PRmul', 'SRmul', 'KTmul', 'WTmul', 'MGCmul']], [y_true_trun, y_pred_trun]

    else:
        raise Exception('y_true_trun与y_pred_trun中序列条数必须相等且>1')


def regression_correlaiton_single(y_true, y_pred):
    """
    :param y_true: 一条真实序列，并与y_pred中的预测序列按顺序一一对应；
    :param y_pred: 一条预测序列，并与y_true中的真实序列按顺序一一对应；
    :return: 各个相关性指标，按顺序分别是：综合相关性指标，PR, SR, KT, WT, MGC
    """

    y_true_trun, y_pred_trun = y_true, y_pred
    # 相关性函数使用如图所示形态的数据作为输入值
    plt.figure('finall input of regression_correlaiton_single, correlation fuctions use following scatters as inputs')
    plt.scatter(y=y_true_trun, x=y_pred_trun)
    plt.xlabel('y_pred_trun')
    plt.ylabel('y_true_trun')

    if (len(y_true_trun) < 5) or (len(y_pred_trun) < 5):
        raise Exception('实际使用的序列对y_true_trun与y_pred_trun中，点数过少不具有统计意义，每条序列至少要≥5个点')
    # 各个相关性指标：
    PR = stats.pearsonr(x=y_true_trun, y=y_pred_trun)[0]
    SR = stats.spearmanr(a=y_true_trun, b=y_pred_trun)[0]
    KT = stats.kendalltau(x=y_true_trun, y=y_pred_trun)[0]
    WT = stats.weightedtau(x=y_true_trun, y=y_pred_trun)[0]
    MGC = stats.multiscale_graphcorr(x=np.array(y_true_trun), y=np.array(y_pred_trun))[0]  # hardly affected by abnormal scatters (i.e. outliers); x and y must be ndarrays; MGC requires at least 5 samples to give reasonable results

    print('原始的评估指标：')
    print('各个相关性指标（越接近1正相关性越强，越接近-1负相关性越强，越接近0相关性越弱）：', '\n', 'PR:', PR, '\n', 'SR:', SR, '\n', 'KT:', KT, '\n', 'WT:', WT, '\n', 'MGC:', MGC, '\n')
    # 不用调和平均、几何平均，避免结果向极小值趋近；不用均方根，避免结果向极大值趋近；使用算术平均加权，权重可根据实际需求手动调整。
    correlation= dyn_seri_weighted([PR, SR, KT, WT, MGC], w=[1,1,1,1,1])
    print('各序列对的最终相关性（越接近1正相关性越强，越接近-1负相关性越强，越接近0相关性越弱）：', '\n', np.array(correlation), '\n')

    # 注意返回的各分量指标是未数值变换前的结果，而最终precision是由各分量指标经数值变换后的结果加权算出的
    return correlation, PR, SR, KT, WT, MGC, y_true_trun, y_pred_trun


def regression_evaluation_pairs(y_true, y_pred):
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

    for i in range(len(y_true_trun)):
        # 第四组相关性函数使用如图所示形态的数据作为输入值。此for循环不能与上一个for循环合并，否则会错误调用plt。
        plt.figure('correlation fuctions (the 4th group in regression_evaluation_pairs) use following scatters as {0}th inputs'.format(i))
        plt.scatter(y=y_true_trun[i], x=y_pred_trun[i])
        plt.xlabel('y_pred_trun[{0}]'.format(i))
        plt.ylabel('y_true_trun[{0}]'.format(i))

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
        MGC.append(stats.multiscale_graphcorr(x=np.array(y_true_trun[i]), y=np.array(y_pred_trun[i]))[0])  # hardly affected by abnormal scatters (i.e. outliers); x and y must be ndarrays; MGC requires at least 5 samples to give reasonable results

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
                                      w=[3,2,2,1, 1,1,1,3,1,1, 1,1, 1/2,1/10,1,1,1,1,1]))
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
    if sum(judge):
        y_true_trun = y_true[judge]
        y_pred_trun = y_pred[judge]

    if (len(y_true_trun) != len(y_pred_trun)) or (len(y_true_trun) < 2):
        raise Exception('y_true_trun与y_pred_trun中序列条数必须相等且≥2')  # 若序列对的数目小于2，则数值变换后的指标均为1

    plt.figure('finall inputs of accuracy functions (first three groups in regression_evaluation_single)')
    # 前三组精度函数使用如图所示形态的数据作为输入值
    ax = plt.subplot(1, 1, 1)
    # xlim = plt.gca().set_xlim(0, length-1)  # xlim使图形按x轴上的点数充满横坐标
    y_true_trun.plot(ax=ax, legend=True)
    y_pred_trun.plot(ax=ax, legend=True)
    print('实际输入的序列对：')
    print(y_true_trun, '\n', y_pred_trun, '\n')

    # 第四组相关性函数使用如图所示形态的数据作为输入值。此for循环不能与上一个for循环合并，否则会错误调用plt。
    plt.figure('correlation fuctions (the 4th group in regression_evaluation_single) use following scatters as {0}th inputs'.format(i))
    plt.scatter(y=y_true_trun, x=y_pred_trun)
    plt.xlabel('y_pred_trun')
    plt.ylabel('y_true_trun')

    if (len(y_true_trun) < 5) or (len(y_pred_trun) < 5):
        raise Exception('实际使用的序列对y_true_trun与y_pred_trun中，点数过少不具有统计意义，每条序列至少要≥5个点')
    # 第一组，零次的相对性指标：
    MAPE=mape(y_true=np.array(y_true_trun), y_pred=np.array(y_pred_trun))  # y_true != 0; no bias
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
    MGC = stats.multiscale_graphcorr(x=np.array(y_true_trun), y=np.array(y_pred_trun))[0]  # hardly affected by abnormal scatters (i.e. outliers); x and y must be ndarrays; MGC requires at least 5 samples to give reasonable results

    print('判断前的真实（及预测）序列对数:', len(y_true), '  判断后的真实（及预测）序列对数:', len(y_true_trun), '\n')
    print('原始的评估指标：')
    print('第一组，零次的相对性指标：', '\n', 'MAPE:', MAPE, '\n', 'SMAPE:', SMAPE, '\n', 'RMSPE:', RMSPE, '\n', 'MTD_p2:', MTD_p2)
    print('第二组，一次的绝对性指标：', '\n', 'EMLAE:', EMLAE, '\n', 'MALE:', MALE, '\n', 'MAE:', MAE, '\n', 'RMSE:', RMSE, '\n', 'MedAE:', MedAE, '\n', 'MTD_p1:', MTD_p1)
    print('第三组，二次的绝对性指标：', '\n', 'MSE:', MSE, '\n', 'MSLE:', MSLE)
    print('第四组，相关性指标：', '\n', 'VAR:', VAR, '\n', 'R2:', R2, '\n', 'PR:', PR, '\n', 'SR:', SR, '\n', 'KT:', KT, '\n', 'WT:', WT, '\n', 'MGC:', MGC, '\n')

    # 不用调和平均、几何平均，避免结果向极小值趋近；不用均方根，避免结果向极大值趋近；使用算术平均加权，权重可根据实际需求手动调整。
    evaluation = dyn_seri_weighted([MAPE, SMAPE, RMSPE, MTD_p2,
                                  EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1,
                                  MSE, MSLE,
                                  VAR, R2, PR, SR, KT, WT, MGC],
                                  w=[3,2,2,1, 1,1,1,3,1,1, 1,1, 1/2,1/10,1,1,1,1,1])

    # 无法得出最终precision，因为各指标的结果数量级不同，又没有其他序列对得出的指标结果作归一化消除数量级的影响
    return MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE, VAR, R2, PR, SR, KT, WT, MGC, y_true_trun, y_pred_trun


# results_v1_all = regression_accuracy_pairs(y_true=y_input_mul_actual[:], y_pred=y_input_mul_pred[:])
# results_v1 = pd.DataFrame(results_v1_all[:-2], index=['precision',
#                                            'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
#                                            'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
#                                            'MSE', 'MSLE'])
# print('指标个数：', len(results_v1))
# print(results_v1, '\n')
#
# results_v2_all = regression_accuracy_single(y_true=y_input_mul_actual[0], y_pred=y_input_mul_pred[0])
# results_v2 = pd.DataFrame(results_v2_all[:-2],
#                           index=[
#                                 'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
#                                 'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
#                                 'MSE', 'MSLE'])
# print('指标个数：', len(results_v2))
# print(results_v2, '\n')

results_v3_all = regression_correlaiton_pairs(y_true=y_input_mul_actual[:], y_pred=y_input_mul_pred[:])
# results_v3 = pd.DataFrame(results_v3_all[:-2], index=['correlaiton', 'PR', 'SR', 'KT', 'WT', 'MGC'])
print('指标个数：', len(results_v3_all[0].columns))
print('序列对的最终相关性指标（越接近1正相关性越强，越接近-1负相关性越强，越接近0相关性越弱）（未归一化）:', '\n', results_v3_all[0], '\n')
# print(results_v3_all[1], '\n')
#
# results_v4_all = regression_correlaiton_single(y_true=y_input_mul_actual[0], y_pred=y_input_mul_pred[0])
# results_v4 = pd.DataFrame(results_v4_all[:-2], index=['correlaiton', 'PR', 'SR', 'KT', 'WT', 'MGC'])
# print('指标个数：', len(results_v4))
# print(results_v4, '\n')
#
# results_v5_all = regression_evaluation_pairs(y_true=y_input_mul_actual[:], y_pred=y_input_mul_pred[:])
# results_v5 = pd.DataFrame(results_v5_all[:-2], index=['evaluation',
#                                            'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
#                                            'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
#                                            'MSE', 'MSLE',
#                                            'VAR', 'R2', 'PR', 'SR', 'KT', 'WT', 'MGC'])
# print('指标个数：', len(results_v5))
# print(results_v5, '\n')
#
# results_v6_all = regression_evaluation_single(y_true=y_input_mul_actual[0], y_pred=y_input_mul_pred[0])
# results_v6 = pd.DataFrame(results_v6_all[:-2], index=[
#                                            'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
#                                            'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
#                                            'MSE', 'MSLE',
#                                            'VAR', 'R2', 'PR', 'SR', 'KT', 'WT', 'MGC'])
# print('指标个数：', len(results_v6))
# print(results_v6, '\n')
#
# ################################################################################################################
# # 将评估函数结果用于动态加权的使用方法：
# # 使用步骤：1.将每个门店单品，两种模型的一段历史区间，预测序列和真实序列的两组序列对，输入regression_evaluation，得到两个评估指标；
# # 2.对这两个指标进行如下操作，得到每个门店单品的两个模型在下一个月所要采用的权重w；
# w = (1-results_v5.loc['evaluation']) / (1-results_v5.loc['evaluation']).sum()
# print('w:', w, '\n', sum(w))
# 因为regression_evaluation返回的最终评估指标precision是(0,1)之间的值，越趋近0则模型预测结果越好，所以需要使用一个递减函数对precision做变换，才能得到真正能使用的权重。
# 这里采用y=1-x，而不用y=1/x，可以避免当x较小时，1/x被放大过多，且一点微小的扰动都会对1/x产生较大影响的不利效应。
# 3.预测下一个月的新数据时，将每个门店单品两个模型的预测序列和各自权重w输入dyn_df_weighted，得到一条预测序列，就是最终发布的预测值。
################################################################################################################
