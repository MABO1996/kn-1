# coding=utf-8
# Define functions and operators
# All time-serial windows are default to 48 as each day has 48 5-minute windows
# All the following inputs of "d" represnet the number of days.

import numpy as np
import pandas as pd
import os
import math
import itertools
import pdb
from scipy.interpolate import UnivariateSpline

"""-----------------------------------------------------------------------------------------------------------------"""

# Global data for running helper function in utilities without specifying
# parameters
if os.path.exists("/shared/JY_Data/eod/ticker_names.npy"):
    tickers = np.load('/shared/JY_Data/eod/ticker_names.npy')
    dates = pd.to_datetime(np.load("/shared/JY_Data/eod/dates.npy"))
else:
    tickers = np.load('/mnt/ssd/eod/ticker_names.npy')
    dates = pd.to_datetime(np.load("/mnt/ssd/eod/dates.npy"))
min_dates = pd.to_datetime(
    np.load("/shared/FactorBank/NewMinData/mindata_concat/min_dates.npy"))
uni_dates = pd.to_datetime(np.unique(min_dates))
# Last five-minute window of each day
idx = np.arange(47, uni_dates.size * 48, 48)

"""-----------------------------------------------------------------------------------------------------------------"""


def floor(float_num):
    return np.int(np.floor(float_num))


"""-----------------------------------------------------------------------------------------------------------------"""


def isOvernight(d):
    if d > 48:
        return np.remainder(d, 48)
    else:
        return d


"""-----------------------------------------------------------------------------------------------------------------"""


def RANK(df):
    return df.rank(axis=1)  # pandas


"""-----------------------------------------------------------------------------------------------------------------"""


def DELAY(df, d):
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
        return np.array(df.shift(periods=isOvernight(d), axis=0))
    else:
        return df.shift(periods=isOvernight(d), axis=0)


"""-----------------------------------------------------------------------------------------------------------------"""


def CORR(df_1, df_2, idx=idx, d=1):  # 与过去n天的相关系数
    corr_res = df_1.rolling(window=48 * d).corr(df_2)
    return corr_res.iloc[idx, :]


"""-----------------------------------------------------------------------------------------------------------------"""


def COV(df_1, df_2, idx=idx, d=1):  # 与过去n天的协方差
    cov_res = df_1.rolling(window=48 * d).cov(df_2)
    return cov_res.iloc[idx, :]


"""-----------------------------------------------------------------------------------------------------------------"""


def SCALE(df, scaled_sum=1):
    scaler = df.abs().sum(axis=0) / scaled_sum
    return df.div(scaler)  # pandas


"""-----------------------------------------------------------------------------------------------------------------"""


def DELTA(df, d):
    return df.diff(periods=isOvernight(d))  # pandas


"""-----------------------------------------------------------------------------------------------------------------"""


def signedpower(df, a):
    return df.pow(a)  # pandas


"""-----------------------------------------------------------------------------------------------------------------"""


def HIGHDAY(df, d):
    win = isOvernight(d)
    mp = floor(win / 2)
    return df.rolling(
        window=win,
        min_periods=mp).apply(
        lambda x: x.shape[0] -
        np.argmax(
            x,
            axis=0) -
        1)


"""-----------------------------------------------------------------------------------------------------------------"""


def LOWDAY(df, d):
    win = isOvernight(d)
    mp = floor(win / 2)
    return df.rolling(
        window=win,
        min_periods=mp).apply(
        lambda x: x.shape[0] -
        np.argmin(
            x,
            axis=0) -
        1)


"""-----------------------------------------------------------------------------------------------------------------"""


def TSMIN(df, d):
    win = isOvernight(d)
    mp = floor(win / 2)
    return df.rolling(window=win, min_periods=mp).min()


"""-----------------------------------------------------------------------------------------------------------------"""


def TSMAX(df, d):
    win = isOvernight(d)
    mp = floor(win / 2)
    return df.rolling(window=win, min_periods=mp).max()


"""-----------------------------------------------------------------------------------------------------------------"""


def MIN(df, d, mp=5):
    return df.rolling(window=d * 48, min_periods=mp).min()


"""-----------------------------------------------------------------------------------------------------------------"""


def MAX(df, d, mp=5):
    return df.rolling(window=d * 48, min_periods=mp).max()


"""-----------------------------------------------------------------------------------------------------------------"""


def ARGMAX(df, d, mp=5):
    return df.rolling(
        window=d * 48,
        min_periods=mp).apply(
        lambda x: x.argmax()) + 1


"""-----------------------------------------------------------------------------------------------------------------"""


def ARGMIN(df, d, mp=5):
    return df.rolling(
        window=d * 48,
        min_periods=mp).apply(
        lambda x: x.argmin()) + 1


"""-----------------------------------------------------------------------------------------------------------------"""


def TSRANK(df, d):
    win = isOvernight(d)
    mp = floor(win / 2)
    return df.rolling(window=win, min_periods=mp).apply(
        lambda x: pd.DataFrame(x).rank(method="min").iloc[-1])


"""-----------------------------------------------------------------------------------------------------------------"""


def SUM(df, d):
    win = isOvernight(d)
    mp = floor(win / 2)
    return df.rolling(window=win, min_periods=mp).sum()


"""-----------------------------------------------------------------------------------------------------------------"""


def SUMAC(df, d):
    win = isOvernight(d)
    mp = floor(win / 2)
    return df.rolling(window=win, axis=1, min_periods=mp).sum()


"""-----------------------------------------------------------------------------------------------------------------"""


def SUMIF(df, d, cond):
    win = isOvernight(d)
    mp = floor(win / 2)
    df[~cond] = 0
    return df.rolling(window=win, min_periods=mp).sum()


"""-----------------------------------------------------------------------------------------------------------------"""


def PROD(df, d):
    win = isOvernight(d)
    mp = floor(win / 2)
    return df.rolling(window=win, min_periods=mp).apply(lambda x: np.prod(x))


"""-----------------------------------------------------------------------------------------------------------------"""


def STD(df, d):
    win = isOvernight(d)
    mp = floor(win / 2)
    return df.rolling(window=win, min_periods=mp).std()


"""-----------------------------------------------------------------------------------------------------------------"""


def MEAN(df, d):
    win = isOvernight(d)
    mp = floor(win / 2)
    return df.rolling(window=win, min_periods=mp).mean()


"""-----------------------------------------------------------------------------------------------------------------"""


def COUNT(cond, d):
    return cond.rolling(window=d).sum()


"""-----------------------------------------------------------------------------------------------------------------"""

def ELEMENT_MIN(df1, df2):
    al = df1 < df2 + 0
    return al * df1 + (1 - al) * df2



def DECAYLINEAR(df, d):
    window = isOvernight(d)
    if df.isnull().values.any():
        df.fillna(value=0, inplace=True)

    lwma = np.zeros_like(df)
    lwma[:window, :] = df.ix[:window, :]
    divider = window * (window + 1) / 2
    wgt = np.arange(window) + 1
    wgt = np.true_divide(wgt, divider)
    series = df.as_matrix()

    for row in range(window - 1, df.shape[0]):
        x = series[(row - window + 1):(row + 1), :]
        lwma[row, :] = (np.dot(x.T, wgt))
    lwma = pd.DataFrame(lwma, index=df.index, columns=df.columns)
    scaler = lwma.abs().sum(axis=0) / 1
    return lwma.div(scaler)


"""-----------------------------------------------------------------------------------------------------------------"""


def WMA(df, d):
    window = isOvernight(d)
    if df.isnull().values.any():
        df.fillna(value=0, inplace=True)

    lwma = np.zeros_like(df)
    lwma[:window, :] = df.ix[:window, :]
    wgt = (5 - np.arange(start=isOvernight(d) - 1, stop=-1, step=-1)) * 0.9
    divider = np.sum(wgt)
    series = df.as_matrix()

    for row in range(window - 1, df.shape[0]):
        x = series[(row - window + 1):(row + 1), :]
        lwma[row, :] = (np.dot(x.T, wgt))
    lwma = pd.DataFrame(lwma, index=df.index, columns=df.columns)
    return lwma.div(divider)


"""-----------------------------------------------------------------------------------------------------------------"""


def interExtraPolation(df):
    # NaN positions in original dataframe
    allNaN = np.isnan(np.array(df))

    # Interpolate
    interpolated = df.reset_index(
        drop=True).interpolate(
        method="spline",
        order=3)

    # Extrapolate
    extraTickers = np.isnan(interpolated).sum()
    extraTickers = np.where(
        np.logical_and(
            extraTickers > 0,
            extraTickers < df.shape[0]))[0]
    series = np.arange(df.shape[0])
    interpolated = np.array(interpolated)
    for i in extraTickers:
        ticker = interpolated[:, i]
        nanPos = np.isnan(ticker)  # logical
        # Fitting spline
        extrapolator = UnivariateSpline(series[~nanPos], ticker[~nanPos])
        # Prediction and replacing NaNs
        ticker[nanPos] = extrapolator(np.where(nanPos)[0])
        interpolated[:, i] = ticker

    return interpolated, allNaN


def SMA(df, n, m, alpha_dates, tickers=tickers):
    df, nanPos = interExtraPolation(df)
    df = pd.DataFrame(df, index=alpha_dates, columns=tickers)
    y_i, y_last = 0, 0
    new_df = np.full_like(df, np.nan)
    df = np.array(df).astype("float")
    for row in range(df.shape[0]):
        if row == 0:
            y_i = df[row, :]
        else:
            y_i = (df[row - 1, :] * m + y_last * (n - m)) / n
        new_df[row, :] = y_i
        y_last = y_i

    new_df[nanPos] = np.nan
    new_df = pd.DataFrame(new_df, index=alpha_dates, columns=tickers)
    return new_df


"""-----------------------------------------------------------------------------------------------------------------"""


def SM_AVG(df, d, mp=5):
    return df.rolling(window=d * 48, min_periods=mp).mean()


"""-----------------------------------------------------------------------------------------------------------------"""


def DECAY_LINEAR(df, d):
    window = d * 48
    if df.isnull().values.any():
        df.fillna(value=0, inplace=True)

    lwma = np.zeros_like(df)
    lwma[:window, :] = df.ix[:window, :]
    divider = window * (window + 1) / 2
    wgt = np.arange(window) + 1
    wgt = np.true_divide(wgt, divider)
    series = df.as_matrix()

    for row in range(window - 1, df.shape[0]):
        x = series[(row - window + 1):(row + 1), :]
        lwma[row, :] = (np.dot(x.T, wgt))
    lwma = pd.DataFrame(lwma, index=df.index, columns=df.columns)
    scaler = lwma.abs().sum(axis=0) / 1
    return lwma.div(scaler)


"""-----------------------------------------------------------------------------------------------------------------"""


def REGBETA(df_1, df_2, d):
    df_1, df_2 = np.array(df_1), np.array(df_2)
    betas = np.full_like(df_1, np.nan)
    iterations = list(
        itertools.product(
            np.arange(
                d - 1,
                df_1.shape[0]),
            np.arange(
                df_1.shape[1])))

    for curr_iter in iterations:
        row = curr_iter[0]
        col = curr_iter[1]
        x = df_1[(row - d + 1):(row + 1), col]
        y = df_2[(row - d + 1):(row + 1), col]
        index = np.logical_and(~np.isnan(x), ~np.isnan(y))
        index = np.logical_and(index, ~np.isinf(x))
        index = np.logical_and(index, ~np.isinf(y))
        if index.sum() <= 1:
            betas[row, col] = np.nan
        else:
            x = x[index]
            y = y[index]
            if np.logical_or(
                    np.count_nonzero(x) <= 1,
                    np.count_nonzero(y) <= 1):
                betas[row, col] = np.nan
            else:
                betas[row, col] = np.polyfit(x, y, 1)[0]
    return betas


"""-----------------------------------------------------------------------------------------------------------------"""


def REGBETA_seq(df_1, seq_n, d):
    df_1 = np.array(df_1)
    betas = np.full_like(df_1, np.nan)
    iterations = list(
        itertools.product(
            np.arange(
                d - 1,
                df_1.shape[0]),
            np.arange(
                df_1.shape[1])))

    for curr_iter in iterations:
        row = curr_iter[0]
        col = curr_iter[1]
        x = df_1[(row - d + 1):(row + 1), col]
        y = np.arange(1, seq_n + 1)
        index = np.logical_and(~np.isnan(x), ~np.isnan(y))
        index = np.logical_and(index, ~np.isinf(x))
        index = np.logical_and(index, ~np.isinf(y))
        if index.sum() <= 1:
            betas[row, col] = np.nan
        else:
            x = x[index]
            y = y[index]
            if np.logical_or(
                    np.count_nonzero(x) <= 1,
                    np.count_nonzero(y) <= 1):
                betas[row, col] = np.nan
            else:
                betas[row, col] = np.polyfit(x, y, 1)[0]
    return betas


"""-----------------------------------------------------------------------------------------------------------------"""


def REGRESID(df_1, df_2, d):
    df_1, df_2 = np.array(df_1), np.array(df_2)
    resid = np.full_like(df_1, np.nan)
    iterations = list(
        itertools.product(
            np.arange(
                d - 1,
                df_1.shape[0]),
            np.arange(
                df_1.shape[1])))

    for curr_iter in iterations:
        row = curr_iter[0]
        col = curr_iter[1]
        x = df_1[(row - d + 1):(row + 1), col]
        y = df_2[(row - d + 1):(row + 1), col]
        index = np.logical_and(~np.isnan(x), ~np.isnan(y))
        index = np.logical_and(index, ~np.isinf(x))
        index = np.logical_and(index, ~np.isinf(y))
        if index.sum() <= 1:
            resid[row, col] = np.nan
        else:
            x = x[index]
            y = y[index]
            if np.logical_or(
                    np.count_nonzero(x) <= 1,
                    np.count_nonzero(y) <= 1):
                resid[row, col] = np.nan
            else:
                resid[row, col] = np.polyfit(x, y, 1, full=True)[1][0]
    return resid


"""-----------------------------------------------------------------------------------------------------------------"""


def wrapping(df, dates, idx, trade_status):
    df = pd.DataFrame(df)
    df[np.isinf(df)] = np.nan
    # capturing info within a day
    # if df.shape[0] > len(idx):
    # df = df.iloc[idx, :]
    # capturing info within a day and between 2 excessive days
    if df.shape[0] > len(idx):
        df = df.groupby(level=0).mean()

    df = df.reindex(dates)
    df = np.array(df)
    df[np.logical_and(np.isnan(df), trade_status == 1)] = 0
    df[trade_status != 1] = np.nan
    df = np.transpose(np.array(df))
    return df


"""-----------------------------------------------------------------------------------------------------------------"""


def my_wrapping(df, dates, trade_status):
    # pdb.set_trace()
    df = pd.DataFrame(df)
    df = df.astype("float")
    df[np.isinf(df)] = np.nan
    if df.shape[0] > 100:  # 100 is the day of self.max_window you defined in Alpha101.py and Alpha191.py
        df = df.groupby(level=0).mean()
    df.index = df.index.map(lambda x: x.strftime("%Y%m%d"))
    df = df.reindex(dates)
    res = df.values.astype("f8")
    res[np.logical_and(np.isnan(res), trade_status == 1)] = 0
    res[trade_status != 1] = np.nan
    res = pd.DataFrame(res, index=df.index, columns=df.columns)
    return res
