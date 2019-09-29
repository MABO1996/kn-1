import numpy as np
import pandas as pd
import os
import math
from scipy.stats import rankdata

# Define functions and operators
## All time-serial windows are default to 48 as each day has 48 5-minute windows
## All the following inputs of "d" represnet the number of days.



def RANK(df):
    return df.rank(axis=1)  # pandas


"""--------------------------------------------------------------------------"""


def DELAY(df, d):
    return df.shift(periods=d * 48, axis=0)


"""--------------------------------------------------------------------------"""


def CORR(df_1, df_2, d):
    correlation = df_1.rolling(window=48 * d, min_periods=1).corr(df_2)
    return correlation


"""--------------------------------------------------------------------------"""


def COVIANCE(df_1, df_2, d):
    covariance = df_1.rolling(window=48 * d, min_periods=1).cov(df_2)
    return covariance


"""--------------------------------------------------------------------------"""


def scale(df, scaled_sum=1):
    scaler = df.abs().sum(axis=0) / scaled_sum
    return df.div(scaler)  # pandas


"""--------------------------------------------------------------------------"""


def DELTA(df, d):
    return df.diff(periods=d * 48)  # pandas


"""--------------------------------------------------------------------------"""


def signedpower(df, a):
    return df.pow(a)  # pandas


"""--------------------------------------------------------------------------"""


def DECAYLINEAR(df, d):
    window = d * 48
    if df.isnull().values.any():
        df.fillna(value=0, inplace=True)
    lwma = np.zeros_like(df)
    lwma[:window, :] = df.values[:window, :]
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


"""--------------------------------------------------------------------------"""


def TSMIN(df, d):
    return df.rolling(window=d * 48, min_periods=5).min()


"""--------------------------------------------------------------------------"""

def my_wrapping(df, dates, trade_status):
	df = pd.DataFrame(df)
	df[np.isinf(df)] = np.nan
	if df.shape[0] > 100: # 100 is the day of self.max_window you defined in Alpha101.py and Alpha191.py
		df = df.groupby(level = 0).mean()
	df = df.reindex(dates)
	df = np.array(df)
	df[np.logical_and(np.isnan(df), trade_status == 1)] = 0
	df[trade_status != 1] = np.nan
	return df[-1, :]

def TSMAX(df, d):
    return df.rolling(window=d * 48, min_periods=5).max()


"""--------------------------------------------------------------------------"""


def ts_argmax(df, d):
    return df.rolling(window=d * 48, min_periods=5).apply(lambda x: x.argmax()) + 1


"""--------------------------------------------------------------------------"""


def ts_argmin(df, d):
    return df.rolling(window=d * 48, min_periods=5).apply(lambda x: x.argmin()) + 1


"""--------------------------------------------------------------------------"""


def TSRANK(df, d):
    return df.rolling(window=d * 48, min_periods=5).apply(lambda x: rankdata(x)[-1])


"""--------------------------------------------------------------------------"""


def SUM(df, d):
    return df.rolling(window=d * 48, min_periods=5).sum()


"""--------------------------------------------------------------------------"""


def product(df, d):
    return df.rolling(window=d * 48).apply(lambda x: np.prod(x))


"""--------------------------------------------------------------------------"""


def MEAN(df, d):
    return df.rolling(window=d * 48).mean()


"""--------------------------------------------------------------------------"""


def STD(df, d):
    return df.rolling(window=d * 48).std()


"""--------------------------------------------------------------------------"""


def LOG(df):
    return np.log(df)


"""--------------------------------------------------------------------------"""


def MIN(df1, df2):
    al = df1 < df2 + 0
    return al * df1 + (1 - al) * df2


"""--------------------------------------------------------------------------"""


def MAX(df1, df2):
    al = df1 > df2 + 0
    return al * df1 + (1 - al) * df2


def SIGN(df):
    return np.sign(df)


def  choose(cho, df1, df2):
    al = cho + 0
    return al * df1 + (1 - al) * df2


def COUNT(con, n):
    con = con + 0
    return con.rolling(n).sum()


def ABS(df):
    return abs(df)

def SMA(df, n, m):
	y_i, y_last = 0, 0
	new_df = np.full_like(df, np.nan)
	for row in range(df.shape[0]):
		if row == 0:
			y_i = df.iloc[row, :]
		else:
			y_i = (df.iloc[row-1, :] * m + y_last * (n - m)) / n
		new_df[row, :] = y_i
		y_last = y_i
	return pd.DataFrame(new_df)


def wrapping(df, dates, idx, trade_status):
	if df.shape[0] > len(idx):
		df = df.groupby(level = 0).mean()
	df = df.reindex(dates)
	df[(np.isnan(df)) & (trade_status == 1)] = 0
	df[(np.isnan(trade_status)) | (trade_status == 0)] = np.nan
	df = np.transpose(np.array(df))
	return df

orderList = [
    7,
    90,
    99,
    105,
    114,
    123,
    125,
    133,
    141,
    146,
    158,
    168,
    176,
    ]



class Alphas():

    def dtm(self):
        al = self.OPEN <= DELAY(self.OPEN, 1)
        m = MAX((self.HIGH - self.OPEN), (self.OPEN - DELAY(self.OPEN, 1)))
        return choose(al, 0, m)

    def dbm(self):
        al = self.OPEN >= DELAY(self.OPEN, 1)
        m = MAX((self.OPEN - self.LOW), (self.OPEN - DELAY(self.OPEN, 1)))
        return choose(al, 0, m)

    def TR(self):
        return MAX(MAX(self.HIGH - self.LOW, abs(self.HIGH - DELAY(self.CLOSE, 1))),
                   abs(self.LOW - DELAY(self.CLOSE, 1)))

    def HD(self):
        return self.HIGH - DELAY(self.HIGH, 1)

    def LD(self):
        return DELAY(self.LOW, 1) - self.LOW

    def alpha1(self):
        rk1 = RANK(DELTA(LOG(self.VOLUME), 1))
        rk2 = RANK(((self.CLOSE - self.OPEN) / self.OPEN))
        cor = CORR(rk1, rk2, 6)
        return wrapping((-1) * cor)

    def alpha2(self):
        return wrapping((-1) * DELTA((((self.CLOSE - self.LOW) - (self.HIGH - self.CLOSE)) / ((self.HIGH - self.LOW))), 1))

    def alpha3(self):
        al1 = self.CLOSE == DELAY(self.CLOSE, 1) + 0
        choose1 = (1 - al1) * self.CLOSE
        al2 = self.CLOSE > DELAY(self.CLOSE, 1) + 0
        choose2 = (al2 * MIN(self.LOW, DELAY(self.CLOSE, 1)) + (1 - al2) * MAX(self.HIGH, DELAY(self.CLOSE, 1)))
        return wrapping(SUM((choose1 - choose2), 6))

    def alpha4(self):
        al1 = ((SUM(self.CLOSE, 2) / 2) < ((SUM(self.CLOSE, 8) / 8) - STD(self.CLOSE, 8))) + 0
        al21 = (1 < (self.VOLUME / MEAN(self.VOLUME, 20))) + 0
        al22 = ((self.VOLUME / MEAN(self.VOLUME, 20)) == 1) + 0
        al2 = np.sign(al21 + al22)
        ch2 = 2 * al2 - 1
        ch1 = al1 + (1 - al1) * ch2
        alp = (((SUM(self.CLOSE, 8) / 8) + STD(self.CLOSE, 8)) < (SUM(self.CLOSE, 2) / 2)) + 0
        return wrapping(alp * (-1) + (1 - alp) * ch1)

    def alpha6(self):
        return wrapping(RANK(SIGN(DELTA((((self.OPEN * 0.85) + (self.HIGH * 0.15))), 4))))

    def alpha7(self):
        return wrapping((RANK(MAX((self.VWAP - self.CLOSE), 3)) + RANK(MIN((self.VWAP - self.CLOSE), 3))) * RANK(
            DELTA(self.VOLUME, 3)))

    def alpha8(self):
        return wrapping(RANK(DELTA(((((self.HIGH + self.LOW) / 2) * 0.2) + (self.VWAP * 0.8)), 4) - 1))

    def alpha11(self):
        return wrapping(SUM(((self.CLOSE - self.LOW) - (self.HIGH - self.CLOSE)) / (self.HIGH - self.LOW) * self.VOLUME, 6))

    def alpha12(self):
        return wrapping(RANK((self.OPEN - (SUM(self.VWAP, 10) / 10)))) * ((-1) * (RANK(abs((self.CLOSE - self.VWAP)))))

    def alpha13(self):
        inn = (self.HIGH * self.LOW)
        return wrapping((signedpower(inn, 0.5)) - self.VWAP)

    def alpha14(self):
        return wrapping(self.CLOSE - DELAY(self.CLOSE, 5))

    def alpha15(self):
        return wrapping(self.OPEN / DELAY(self.CLOSE, 1) - 1)

    def alpha16(self):
        return wrapping((-1) * TSMAX(RANK(CORR(RANK(self.VOLUME), RANK(self.VWAP), 5)), 5))

    def alpha17(self):
        inn1 = RANK((self.VWAP - MAX(self.VWAP, 15)))
        inn2 = DELTA(self.CLOSE, 5)
        return wrapping(pow(inn1, inn2))

    def alpha18(self):
        return wrapping(self.CLOSE / DELAY(self.CLOSE, 5))

    def alpha19(self):
        df = choose(self.CLOSE == DELAY(self.CLOSE, 5), 0, (self.CLOSE - DELAY(self.CLOSE, 5)) / self.CLOSE)
        return wrapping(choose(self.CLOSE < DELAY(self.CLOSE, 5), (self.CLOSE - DELAY(self.CLOSE, 5)) / DELAY(self.CLOSE, 5), df))

    def alpha20(self):
        return wrapping((self.CLOSE - DELAY(self.CLOSE, 6)) / DELAY(self.CLOSE, 6) * 100)

    def alpha25(self):
        RET = np.load('RET.npy')
        return wrapping(((-1) * RANK(
            (DELTA(self.CLOSE, 7) * (1 - RANK(DECAYLINEAR((self.VOLUME / MEAN(self.VOLUME, 20)), 9)))))) * (
                            1 + RANK(SUM(RET, 250))))

    def alpha26(self):
        return wrapping((((SUM(self.CLOSE, 7) / 7) - self.CLOSE)) + ((CORR(self.VWAP, DELAY(self.CLOSE, 5), 230))))

    def alpha29(self):
        return wrapping((self.CLOSE - DELAY(self.CLOSE, 6)) / DELAY(self.CLOSE, 6) * self.VOLUME)

    def alpha31(self):
        return wrapping((self.CLOSE - MEAN(self.CLOSE, 12)) / MEAN(self.CLOSE, 12) * 100)

    def alpha32(self):
        return wrapping((-1) * SUM(RANK(CORR(RANK(self.HIGH), RANK(self.VOLUME), 3)), 3))

    def alpha34(self):
        return wrapping(MEAN(self.CLOSE, 12) / self.CLOSE)

    def alpha35(self):
        rk1 = RANK(DECAYLINEAR(DELTA(self.OPEN, 1), 15))
        rk2 = RANK(DECAYLINEAR(CORR((self.VOLUME), ((self.OPEN * 0.65) + (self.OPEN * 0.35)), 17), 7))
        return wrapping(MIN(rk1, rk2) * (-1))

    def alpha36(self):
        return wrapping(RANK(SUM(CORR(RANK(self.VOLUME), RANK(self.VWAP), 6), 2)))

    def alpha38(self):
        al = ((SUM(self.HIGH, 20) / 20) < self.HIGH)
        df1 = (-1 * DELTA(self.HIGH, 2))
        return wrapping(choose(al, df1, 0))

    def alpha40(self):
        a = SUM(choose(self.CLOSE > DELAY(self.CLOSE, 1), self.VOLUME, 0), 26)
        b = SUM(choose(self.CLOSE <= DELAY(self.CLOSE, 1), self.VOLUME, 0), 26)
        return wrapping(a / b * 100)

    def alpha41(self):
        return wrapping(RANK(MAX(DELTA((self.VWAP), 3), 5)) * (-1))

    def alpha42(self):
        return wrapping(((-1) * RANK(STD(self.HIGH, 10))) * CORR(self.HIGH, self.VOLUME, 10))

    def alpha43(self):
        cho1 = self.CLOSE < DELAY(self.CLOSE, 1)
        df = choose(cho1, -self.VOLUME, 0)
        al1 = self.CLOSE > DELAY(self.CLOSE, 1)
        cho2 = choose(al1, self.VOLUME, df)
        return wrapping(SUM(cho2, 6))

    def alpha45(self):
        return wrapping(RANK(DELTA((((self.CLOSE * 0.6) + (self.OPEN * 0.4))), 1)) * RANK(
            CORR(self.VWAP, MEAN(self.VOLUME, 150), 15)))

    def alpha46(self):
        return wrapping((MEAN(self.CLOSE, 3) + MEAN(self.CLOSE, 6) + MEAN(self.CLOSE, 12) + MEAN(self.CLOSE, 24)) / (
                    4 * self.CLOSE))

    def alpha48(self):
        return wrapping((-1) * ((RANK(((SIGN((self.CLOSE - DELAY(self.CLOSE, 1))) + SIGN(
            (DELAY(self.CLOSE, 1) - DELAY(self.CLOSE, 2)))) + SIGN(
            (DELAY(self.CLOSE, 2) - DELAY(self.CLOSE, 3)))))) * SUM(self.VOLUME, 5)) / SUM(self.VOLUME, 20))

    def alpha49(self):
        condition1 = (self.HIGH + self.LOW) <= (DELAY(self.HIGH, 1) + DELAY(self.LOW, 1))
        condition2 = (self.HIGH + self.LOW) >= (DELAY(self.HIGH, 1) + DELAY(self.LOW, 1))
        m = MAX(ABS(self.HIGH - DELAY(self.HIGH, 1)), ABS(self.LOW - DELAY(self.LOW, 1)))
        c = SUM(choose(condition1, 0, m), 12)
        d = (SUM(choose(condition2, 0, m), 12) + SUM(choose(condition1, 0, m), 12))
        return wrapping(c / d)

    def alpha50(self):
        condition1 = (self.HIGH + self.LOW) <= (DELAY(self.HIGH, 1) + DELAY(self.LOW, 1))
        condition2 = (self.HIGH + self.LOW) >= (DELAY(self.HIGH, 1) + DELAY(self.LOW, 1))
        m = MAX(ABS(self.HIGH - DELAY(self.HIGH, 1)), ABS(self.LOW - DELAY(self.LOW, 1)))
        a = SUM(choose(condition1, 0, m), 12)
        b = (SUM(choose(condition1, 0, m), 12) + SUM(choose(condition2, 0, m), 12))
        c = SUM(choose(condition2, 0, m), 12)
        d = (SUM(choose(condition2, 0, m), 12) + SUM(choose(condition1, 0, m), 12))
        return wrapping(a / b - c / d)

    def alpha51(self):
        condition1 = (self.HIGH + self.LOW) <= (DELAY(self.HIGH, 1) + DELAY(self.LOW, 1))
        condition2 = (self.HIGH + self.LOW) >= (DELAY(self.HIGH, 1) + DELAY(self.LOW, 1))
        return wrapping(SUM(choose(condition1, 0, MAX(ABS(self.HIGH - DELAY(self.HIGH, 1)), ABS(self.LOW - DELAY(self.LOW, 1)))),
                   12) / (SUM(
            choose(condition1, 0, MAX(ABS(self.HIGH - DELAY(self.HIGH, 1)), ABS(self.LOW - DELAY(self.LOW, 1)))),
            12) + SUM(
            choose(condition2, 0, MAX(ABS(self.HIGH - DELAY(self.HIGH, 1)), ABS(self.LOW - DELAY(self.LOW, 1)))), 12)))

    def alpha53(self):
        return wrapping(COUNT(self.CLOSE > DELAY(self.CLOSE, 1), 12) / 12 * 100)

    def alpha58(self):
        con = self.CLOSE > DELAY(self.CLOSE, 1)
        return wrapping(COUNT(con, 20) / 20 * 100)

    def alpha59(self):
        al1 = self.CLOSE == DELAY(self.CLOSE, 1)
        al = self.CLOSE > DELAY(self.CLOSE, 1)
        cho = choose(al, MIN(self.LOW, DELAY(self.CLOSE, 1)), MAX(self.HIGH, DELAY(self.CLOSE, 1)))
        df2 = self.CLOSE - cho
        return wrapping(SUM(choose(al1, 0, df2), 20))

    def alpha60(self):
        return wrapping(SUM(((self.CLOSE - self.LOW) - (self.HIGH - self.CLOSE)) / (self.HIGH - self.LOW) * self.VOLUME, 20))

    def alpha61(self):
        return wrapping(MAX(RANK(DECAYLINEAR(DELTA(self.VWAP, 1), 12)),
                    RANK(DECAYLINEAR(RANK(CORR((self.LOW), MEAN(self.VOLUME, 80), 8)), 17))) * (-1))

    def alpha62(self):
        return wrapping((-1) * CORR(self.HIGH, RANK(self.VOLUME), 5))

    def alpha64(self):
        return wrapping(MAX(RANK(DECAYLINEAR(CORR(RANK(self.VWAP), RANK(self.VOLUME), 4), 4)),
                    RANK(DECAYLINEAR(MAX(CORR(RANK(self.CLOSE), RANK(MEAN(self.VOLUME, 60)), 4), 13), 14))) * (-1))

    def alpha65(self):
        return wrapping(MEAN(self.CLOSE, 6) / self.CLOSE)

    def alpha66(self):
        return wrapping((self.CLOSE - MEAN(self.CLOSE, 6)) / MEAN(self.CLOSE, 6) * 100)

    def alpha71(self):
        return wrapping((self.CLOSE - MEAN(self.CLOSE, 24)) / MEAN(self.CLOSE, 24) * 100)

    def alpha74(self):
        return wrapping(RANK(CORR(SUM(((self.LOW * 0.35) + (self.VWAP * 0.65)), 20), SUM(MEAN(self.VOLUME, 40), 20), 7)) + RANK(
            CORR(RANK(self.VWAP), RANK(self.VOLUME), 6)))

    def alpha76(self):
        return wrapping(STD(abs((self.CLOSE / DELAY(self.CLOSE, 1) - 1)) / self.VOLUME, 20) / MEAN(
            abs((self.CLOSE / DELAY(self.CLOSE, 1) - 1)) / self.VOLUME, 20))

    def alpha80(self):
        return wrapping((self.VOLUME - DELAY(self.VOLUME, 5)) / DELAY(self.VOLUME, 5) * 100)

    def alpha83(self):
        return wrapping((-1) * RANK(COVIANCE(RANK(self.HIGH), RANK(self.VOLUME), 5)))

    def alpha84(self):
        al = self.CLOSE > DELAY(self.CLOSE, 1)
        all = self.CLOSE < DELAY(self.CLOSE, 1)
        choo = choose(all, -self.VOLUME, 0)
        return wrapping(SUM(choose(al, self.VOLUME, choo), 20))

    def alpha86(self):
        cho = (0.25 < (((DELAY(self.CLOSE, 20) - DELAY(self.CLOSE, 10)) / 10) - (
                    (DELAY(self.CLOSE, 10) - self.CLOSE) / 10)))
        cho1 = ((((DELAY(self.CLOSE, 20) - DELAY(self.CLOSE, 10)) / 10) - (
                    (DELAY(self.CLOSE, 10) - self.CLOSE) / 10)) < 0)
        df2 = choose(cho1, 1, ((-1) * (self.CLOSE - DELAY(self.CLOSE, 1))))
        return wrapping(choose(cho, -1, df2))

    def alpha88(self):
        return wrapping((self.CLOSE - DELAY(self.CLOSE, 20)) / DELAY(self.CLOSE, 20) * 100)

    def alpha90(self):
        return wrapping(RANK(CORR(RANK(self.VWAP), RANK(self.VOLUME), 5)) * -1)

    def alpha91(self):
        return wrapping((RANK((self.CLOSE - MAX(self.CLOSE, 5))) * RANK(CORR((MEAN(self.VOLUME, 40)), LOW, 5))) * (-1))

    def alpha92(self):
        return wrapping(MAX(RANK(DECAYLINEAR(DELTA(((self.CLOSE * 0.35) + (self.VWAP * 0.65)), 2), 3)),
                    TSRANK(DECAYLINEAR(ABS(CORR((MEAN(self.VOLUME, 180)), self.CLOSE, 13)), 5), 15)) * (-1))

    def alpha93(self):
        al = self.OPEN >= DELAY(self.OPEN, 1)
        return wrapping(SUM(choose(al, 0, MAX((self.OPEN - self.LOW), (self.OPEN - DELAY(self.OPEN, 1)))), 20))

    def alpha94(self):
        al = self.CLOSE > DELAY(self.CLOSE, 1)
        al1 = self.CLOSE < DELAY(self.CLOSE, 1)
        df = choose(al1, -self.VOLUME, 0)
        return wrapping(SUM(choose(al, self.VOLUME, df), 30))

    def alpha69(self):
        DTM = np.load('DTM.npy')
        DTM=pd.DataFrame(DTM)
        DBM = np.load('DBM.npy')
        DBM = pd.DataFrame(DBM)
        df1 = (SUM(DTM, 20) - SUM(DBM, 20)) / SUM(DTM, 20)
        all = SUM(DTM, 20) == SUM(DBM, 20)
        ch2 = (SUM(DTM, 20) - SUM(DBM, 20)) / SUM(DBM, 20)
        df2 = choose(all, 0, ch2)
        al = SUM(DTM, 20) > SUM(DBM, 20)
        return wrapping(choose(al, df1, df2))

    def alpha10(self):
        RET=np.load('/home/hwzhang/RET.npy')
        RET=pd.DataFrame(RET)
        al=(RET<0)+0
        cho=choose(al,STD(RET,20),self.CLOSE)
        return wrapping(RANK(MAX(cho*cho,5)))

    def alpha39(self):
        return wrapping(RANK(DECAYLINEAR(DELTA((self.CLOSE), 2), 8)) - RANK(
            DECAYLINEAR(CORR(((self.VWAP * 0.3) + (self.OPEN * 0.7)), SUM(MEAN(self.VOLUME, 180), 37), 14), 12)))

    def alpha52(self):
        LD =DELAY(self.LOW, 1) - self.LOW
        return wrapping(SUM(MAX(0, self.HIGH - DELAY((self.HIGH + self.LOW + self.CLOSE) / 3, 1)), 26) / SUM(MAX(0, DELAY((self.HIGH + self.LOW + self.CLOSE) / 3, 1) - LD), 26) * 100)

    def alpha54(self):
        std = STD(ABS(self.CLOSE - self.OPEN), 12)
        cor = CORR(self.CLOSE, self.OPEN, 10)
        rk = RANK((std + (self.CLOSE - self.OPEN)) + cor)
        return wrapping((-1) * rk)

    def alpha56(self):
        ran = RANK(CORR(SUM(((self.HIGH + self.LOW) / 2), 19), SUM(MEAN(self.VOLUME, 40), 19), 13))
        return wrapping((RANK((self.OPEN - TSMIN(self.OPEN, 12))) < RANK(pow(ran, 5))) + 0)


    def alpha37(self):
        RET=np.load('/home/hwzhang/RET.npy')
        RET=pd.DataFrame(RET)
        return wrapping((-1)*RANK(((SUM(self.OPEN,5)*SUM(RET,5))-DELAY((SUM(self.OPEN,5)*SUM(RET,5)),10))))

    def alpha87(self):
        dec=DECAYLINEAR(((((self.LOW*0.9)+(self.LOW*0.1))-self.VWAP)/(self.OPEN-((self.HIGH+self.LOW)/2))),11)
        ts=TSRANK(dec,7)
        return wrapping((RANK(DECAYLINEAR(DELTA(self.VWAP,4),7))+ts)*(-1))

    def alpha77(self):
        return wrapping(MIN(RANK(DECAYLINEAR(((((self.HIGH+self.LOW)/2)+self.HIGH)-(self.VWAP+self.HIGH)),20)),RANK(DECAYLINEAR(CORR(((self.HIGH+self.LOW)/2),MEAN(self.VOLUME,40),3),6))))

    def alpha70(self):
        return wrapping(STD(self.VALUE,6))

    def alpha33(self):
        RET = np.load('/home/hwzhang/RET.npy')
        RET=pd.DataFrame(RET)
        inn=((((-1) * TSMIN(self.LOW, 5)) + DELAY(TSMIN(self.LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220)))

        v=TSRANK(self.volume,5)
        return wrapping(inn * v)

    def alpha73(self):
        dec=DECAYLINEAR(DECAYLINEAR(CORR((self.CLOSE), self.VOLUME, 10), 16), 4)
        ts=TSRANK(dec, 5)
        rk=RANK(DECAYLINEAR(CORR(self.VWAP, MEAN(self.VOLUME, 30), 4), 3))
        return wrapping((ts - rk) * (-1))



    def alpha9(self):
        v = ((self.HIGH + self.LOW) / 2 - (DELAY(self.HIGH, 1) + DELAY(self.LOW, 1)) * (self.HIGH - self.LOW) / self.VOLUME)
        v=v.fillna(0)
        return wrapping(SMA(v, 7, 2))

    def alpha22(self):
        v=((self.CLOSE - MEAN(self.CLOSE, 6)) / MEAN(self.CLOSE, 6) - DELAY((self.CLOSE - MEAN(self.CLOSE, 6)) / MEAN(self.CLOSE, 6), 3))
        v=v.fillna(0)
        return wrapping(SMA(v, 12, 1))

    def alpha23(self):
        std=STD(self.CLOSE,20)
        al=self.CLOSE > DELAY(self.CLOSE, 1)
        alt=self.CLOSE <=DELAY(self.CLOSE, 1)
        v1=choose(al,std, 0)
        v1=v1.fillna(0)
        s1=SMA(v1, 20, 1)
        v2=choose(alt,std,0)
        v2=v2.fillna(0)
        ss2=SMA(v2, 20, 1)
        s2=(s1+ss2)
        return wrapping(s1/s2*100)




    def alpha24(self):
        v=self.CLOSE-DELAY(self.CLOSE,5)
        v=v.fillna(0)
        return wrapping(SMA(v,5,1))

    def alpha28(self):
        sm1=((self.CLOSE - TSMIN(self.LOW, 9)) / (TSMAX(self.HIGH, 9) - TSMIN(self.LOW, 9)) * 100).fillna(0)
        sm2=((self.CLOSE - TSMIN(self.LOW, 9)) / (MAX(self.HIGH, 9) - TSMAX(self.LOW, 9)) * 100).fillna(0)
        sm3=(SMA(sm2, 3, 1)).fillna(0)
        return wrapping(3 * SMA(sm1, 3, 1) - 2 * SMA(sm3, 3, 1))

    def alpha47(self):
        v=(TSMAX(self.HIGH,6)-self.CLOSE)/(TSMAX(self.HIGH,6)-TSMIN(self.LOW,6))*100
        v=v.fillna(0)
        return wrapping(SMA(v,9,1))

    def alpha57(self):
        ts1=(self.CLOSE-TSMIN(self.LOW,9))
        ts2=TSMAX(self.HIGH,9)
        ts3=TSMIN(self.LOW,9)
        v=ts1/(ts2-ts3)*100
        v=v.fillna(0)
        return wrapping(SMA(v,3,1))

    def alpha63(self):
        sm1=MAX(self.CLOSE - DELAY(self.CLOSE, 1), 0)
        s1=SMA(sm1.fillna(0), 6, 1)
        sm2=ABS(self.CLOSE - DELAY(self.CLOSE, 1))
        s2=SMA(sm2.fillna(0), 6, 1)
        return wrapping(s1/s2 * 100)

    def alpha67(self):
        sm1 = MAX(self.CLOSE - DELAY(self.CLOSE, 1), 0)
        s1 = SMA(sm1.fillna(0), 24, 1)
        sm2 = ABS(self.CLOSE - DELAY(self.CLOSE, 1))
        s2 = SMA(sm2.fillna(0), 24, 1)
        return wrapping(s1/s2*100)

    def alpha68(self):
        v=((self.HIGH+self.LOW)/2-(DELAY(self.HIGH,1)+DELAY(self.LOW,1))/2)*(self.HIGH-self.LOW)/self.VOLUME
        return wrapping(SMA(v.fillna(0),15,2))

    def alpha72(self):
        ts1=(TSMAX(self.HIGH,6)-self.CLOSE)
        ts2=(TSMAX(self.HIGH,6)-TSMIN(self.LOW,6))
        v=ts1/ts2*100
        return wrapping(SMA(v.fillna(0),15,1))

    def alpha81(self):
        v=self.VOLUME
        v=v.fillna(0)
        return wrapping(SMA(v,21,2))


    def alpha89(self):
        s1=SMA(self.CLOSE.fillna(0),27,2)
        s2=SMA(self.CLOSE.fillna(0),13,2)
        s3=SMA((s2-s1).fillna(0),10,2)
        return wrapping(2*(s2-s1-s3))

    def alpha78(self):
        a=(self.HIGH + self.LOW + self.CLOSE) / 3
        b=MEAN(a, 12)
        c=MEAN(ABS(self.CLOSE - b), 12)
        return wrapping((a- b) / (0.015 * c))

    def alpha82(self):
        ts1 = (TSMAX(self.HIGH, 6) - self.CLOSE)
        ts2 = (TSMAX(self.HIGH, 6) - TSMIN(self.LOW, 6))
        v = ts1 / ts2 * 100
        return wrapping(SMA(v.fillna(0), 20, 1))


    def alpha79(self):
        sm1 = MAX(self.CLOSE - DELAY(self.CLOSE, 1), 0)
        s1 = SMA(sm1.fillna(0), 12, 1)
        sm2 = ABS(self.CLOSE - DELAY(self.CLOSE, 1))
        s2 = SMA(sm2.fillna(0), 12, 1)
        return wrapping(s1 / s2 * 100)


    def alpha27(self):
        A=(self.CLOSE-DELAY(self.CLOSE,3))
        B=DELAY(self.CLOSE,3)
        C=(self.CLOSE-DELAY(self.CLOSE,6))
        D=DELAY(self.CLOSE,6)
        return wrapping(DECAYLINEAR(A/B*100+C/D*100,12))


    def alpha55(self):
        inn = (self.CLOSE - DELAY(self.CLOSE, 1) + (self.CLOSE - self.OPEN) / 2 + DELAY(self.CLOSE, 1) - DELAY(self.OPEN, 1))
        a1=ABS(self.HIGH - DELAY(self.CLOSE, 1)) > ABS(self.LOW - DELAY(self.CLOSE, 1))
        a1=a1+0
        a2= ABS(self.HIGH - DELAY(self.CLOSE, 1)) > ABS(self.HIGH - DELAY(self.LOW, 1))
        a2=a2+0
        condition1 = np.sign(a1 +a2)
        res1 = ABS(self.HIGH - DELAY(self.CLOSE, 1)) + ABS(self.LOW - DELAY(self.CLOSE, 1)) / 2 + ABS(
            DELAY(self.CLOSE, 1) - DELAY(self.OPEN, 1)) / 4
        condition2 = np.sign(((ABS(self.LOW - DELAY(self.CLOSE, 1)) > ABS(self.HIGH - DELAY(self.LOW, 1))) + 0) + (
            (ABS(self.LOW - DELAY(self.CLOSE, 1)) > ABS(self.HIGH - DELAY(self.CLOSE, 1))) + 0))
        re1 = ABS(self.LOW - DELAY(self.CLOSE, 1)) + ABS(self.HIGH - DELAY(self.CLOSE, 1)) / 2 + ABS(
            DELAY(self.CLOSE, 1) - DELAY(self.OPEN, 1)) / 4
        re2 = ABS(self.HIGH - DELAY(self.LOW, 1)) + ABS(DELAY(self.CLOSE, 1) - DELAY(self.OPEN, 1)) / 4
        res2 = choose(condition2, re1, re2)
        return wrapping(SUM(16 * inn / (choose(condition1, res1, res2)) * MAX(ABS(self.HIGH - DELAY(self.CLOSE, 1)), ABS(self.LOW - DELAY(self.CLOSE, 1))),20))

    def alpha109(self):
        v=SMA((self.HIGH - self.LOW).fillna(0), 10, 2)
        a=SMA((self.HIGH - self.LOW).fillna(0), 10, 2)
        return wrapping(a/SMA(v.fillna(0), 10, 2))



    """------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

    def alpha5(self):
        v=np.load('/home/hwzhang/pyfiles/tsvolume5.npy')
        v=pd.DataFrame(v)
        return wrapping((-1) * TSMAX(CORR(v, TSRANK(self.HIGH, 5), 5), 3))

    def alpha44(self):
        return wrapping(TSRANK(DECAYLINEAR(CORR(((self.LOW)), MEAN(self.VOLUME, 10), 7), 6), 4) + TSRANK(
            DECAYLINEAR(DELTA((self.VWAP), 3), 10), 15))

    def alpha85(self):
        return wrapping(TSRANK((self.VOLUME / MEAN(self.VOLUME, 20)), 20) * TSRANK(((-1) * DELTA(self.CLOSE, 7)), 8))

    def alpha132(self):
        return wrapping(MEAN(self.VALUE,20))

    def alpha95(self):
        return wrapping(STD(self.VALUE, 20))

    def alpha110(self):
        a=SUM(MAX(0,self.HIGH-DELAY(self.CLOSE,1)),20)
        b=SUM(MAX(0,DELAY(self.CLOSE,1)-self.LOW),20)
        return wrapping(a/b*100)

    def alpha126(self):
        return wrapping((self.CLOSE+self.HIGH+self.LOW)/3)


    def alpha161(self):
        m=MAX((self.HIGH - self.LOW), ABS(DELAY(self.CLOSE, 1) - self.HIGH))
        n=ABS(DELAY(self.CLOSE, 1) - self.LOW)
        return wrapping(MEAN(MAX(m, n), 12))

    def alpha158(self):
        return wrapping(((self.HIGH - SMA(self.CLOSE, 15, 2)) - (self.LOW - SMA(self.CLOSE, 15, 2))) / self.CLOSE)

    def alpha96(self):
        a=(self.CLOSE-TSMIN(self.LOW,9))/(TSMAX(self.HIGH,9)-TSMIN(self.LOW,9))*100
        a=a.fillna(0)
        v=SMA(a,3,1)
        return wrapping(SMA(v.fillna(0),3,1))

    def alpha159(self):
        aa=MIN(self.LOW, DELAY(self.CLOSE, 1))
        a=(self.CLOSE - SUM(aa, 6))
        bb=MAX(self.HIGH, DELAY(self.CLOSE, 1)) - MIN(self.LOW, DELAY(self.CLOSE, 1))
        b=SUM(bb,6)
        c=(self.CLOSE - SUM(aa, 12))
        d=SUM(bb, 12)
        e=(self.CLOSE - SUM(aa, 24))
        f=SUM(bb, 24)
        return wrapping((a/ b * 12 * 24 + c / d*6 * 24 + e / f*6 * 24)*100 / (6 * 12 + 6 * 24 + 12 * 24))


    def alpha125(self):
        cor=CORR((self.VWAP),MEAN(self.VOLUME,80),17)
        rk1=RANK(DECAYLINEAR(cor,20))
        dt=DELTA(((self.CLOSE*0.5)+(self.VWAP*0.5)),3)
        rk2=RANK(DECAYLINEAR(dt,16))
        return wrapping(rk1/rk2)

    def alpha165(self):
        RET=np.load('/home/hwzhang/RET.npy')
        RET=pd.DataFrame(RET)
        mea=(((-1) * RET)*MEAN(self.VOLUME, 20))
        inn=(mea * self.VWAP)*(self.HIGH - self.CLOSE)
        return wrapping(RANK(inn))

    def alpha133(self):
        HD=self.HIGH-DELAY(self.HIGH,1)
        LD=DELAY(self.LOW,1)-self.LOW
        return wrapping(((20 - HD*(self.HIGH, 20)) / 20) * 100 - ((20 - LD*(self.LOW, 20)) / 20) * 100)


    def alpha103(self):
        LD = DELAY(self.LOW, 1) - self.LOW
        return ((20-LD*(self.LOW,20))/20)*100

    def alpha172(self):
        aa = pd.DataFrame(np.zeros_like(self.ld), index=self.ld.index, columns=self.ld.columns)
        aa[(self.ld > 0) & (self.ld > self.hd)] = self.ld[(self.ld > 0) & (self.ld > self.hd)]
        aa = SUM(aa, 14) * 100 / SUM(self.tr, 14)
        bb = pd.DataFrame(np.zeros_like(self.hd), index=self.ld.index, columns=self.ld.columns)
        bb[(self.hd > 0) & (self.hd > self.ld)] = self.hd[(self.hd > 0) & (self.hd > self.ld)]
        bb = SUM(bb, 14) * 100 / SUM(self.tr, 14)
        res = MEAN((aa.abs() - bb) / (aa + bb) * 100, 6)
        return wrapping(res)

    def alpha128(self):
        aa = pd.DataFrame(np.zeros_like(self.high), index=self.high.index, columns=self.high.columns)
        bb = pd.DataFrame(np.zeros_like(self.high), index=self.high.index, columns=self.high.columns)
        aa[((self.high + self.low + self.close) / 3) > DELAY((self.high + self.low + self.close) / 3, 1)] = ((self.high + self.low + self.close) / 3) * self.volume[((self.high + self.low + self.close) / 3) > DELAY((self.high + self.low + self.close) / 3,1)]
        bb[((self.high + self.low + self.close) / 3) < DELAY((self.high + self.low + self.close) / 3, 1)] = ((self.high + self.low + self.close) / 3) * self.volume[((self.high + self.low + self.close) / 3) < DELAY((self.high + self.low + self.close) / 3,1)]
        cc = 1 + SUM(aa, 14) / SUM(bb, 14)
        res = 100 - (100 / cc)
        return wrapping(res)

    def alpha107(self):
        return wrapping((((-1) * RANK((self.OPEN - DELAY(self.HIGH, 1)))) * RANK((self.OPEN - DELAY(self.CLOSE, 1)))) * RANK((self.OPEN - DELAY(self.LOW, 1))))

    def alpha118(self):
        return wrapping(SUM(self.HIGH-self.OPEN,20)/SUM(self.OPEN-self.LOW,20)*100)

    def alpha120(self):
        return wrapping(RANK((self.VWAP - self.CLOSE)) / RANK((self.VWAP + self.CLOSE)))

    def alpha124(self):

        b=TSMAX(self.CLOSE, 30)
        c=RANK(b)
        d=DECAYLINEAR(c, 2)

        a = (self.CLOSE - self.VWAP)
        return  wrapping(a/d)

    def alpha150(self):
        return wrapping((self.CLOSE + self.HIGH + self.LOW) / 3 * self.VOLUME)

    def alpha170(self):
        return wrapping((((RANK((1 / self.CLOSE))*self.VOLUME) / MEAN(self.VOLUME, 20))*((self.HIGH * RANK((self.HIGH - self.CLOSE))) / (SUM(self.HIGH, 5) / 5))) - RANK((self.VWAP - DELAY(self.VWAP, 5))))

    def alpha171(self):
        a=signedpower(self.OPEN,5)
        return wrapping(((-1) * ((self.LOW - self.CLOSE)*a)) / ((self.CLOSE - self.HIGH)*a))

    @assignOrder(1)
    def alpha191(self):
        res = my_wrapping(CORR(MEAN(self.volume, 20), self.low, self.idx, 5) + (self.high + self.low) / 2 - self.close,
                          self.dates, self.TD)
        return res

    @assignOrder(2)
    def alpha190(self):
        aa = self.close / DELAY(self.close, 1) - 1
        bb = signedpower(self.close / DELAY(self.close, 19), 0.05) - 1
        res = my_wrapping(np.log(COUNT(aa > bb, 20) * SUMIF(signedpower(aa - bb, 2), 20, aa > bb) / (
                    COUNT(aa < bb, 20) * SUMIF(signedpower(aa - bb, 2), 20, aa > bb))), self.dates, self.TD)
        return res

    @assignOrder(3)
    def alpha189(self):
        res = my_wrapping(MEAN((self.close - MEAN(self.close, 6)).abs(), 6), self.dates, self.TD)
        return res

    @assignOrder(4)
    def alpha188(self):
        res = my_wrapping(pd.DataFrame((np.true_divide(np.array(self.high - self.low), np.array(
            SMA((self.high - self.low), 11, 2, alpha_dates, self.tickers))) - 1) * 100, index=alpha_dates,
                                       columns=self.tickers), self.dates, self.TD)
        return res

    @assignOrder(5)
    def alpha187(self):
        res = np.maximum(self.high - self.low, self.opens - DELAY(self.opens, 1))
        res[self.opens < DELAY(self.opens, 1)] = 0
        res = my_wrapping(SUM(res, 20), self.dates, self.TD)
        return res

    @assignOrder(6)
    def alpha186(self):
        aa = pd.DataFrame(np.zeros_like(self.ld), index=self.ld.index, columns=self.ld.columns)
        aa[(self.ld > 0) & (self.ld > self.hd)] = self.ld[(self.ld > 0) & (self.ld > self.hd)]
        aa = SUM(aa, 14) * 100 / SUM(self.tr, 14)
        bb = pd.DataFrame(np.zeros_like(self.hd), index=self.ld.index, columns=self.ld.columns)
        bb[(self.hd > 0) & (self.hd > self.ld)] = self.hd[(self.hd > 0) & (self.hd > self.ld)]
        bb = SUM(bb, 14) * 100 / SUM(self.tr, 14)
        cc = MEAN((aa - bb).abs() / (aa + bb) * 100, 6)
        dd = DELAY(cc, 6)
        res = my_wrapping((cc + dd) / 2, self.dates, self.TD)
        return res

    @assignOrder(7)
    def alpha185(self):
        res = my_wrapping(RANK(-1 * signedpower((1 - self.opens / self.close), 2)), self.dates, self.TD)
        return res

    @assignOrder(8)
    def alpha184(self):
        aa = DELAY(self.opens - self.close, 1)
        bb = RANK(CORR(aa, self.close, self.idx, 200))
        cc = (RANK(self.opens - self.close)).groupby(level=0).mean()
        res = my_wrapping(
            pd.DataFrame(np.nansum(np.dstack((bb, cc)), 2), index=np.unique(alpha_dates), columns=self.tickers),
            self.dates, self.TD)
        return res

    @assignOrder(9)
    def alpha182(self):
        aa = (self.close > self.opens) & (self.benchmarkclose > self.benchmarkopens)
        bb = (self.close < self.opens) & (self.benchmarkclose < self.benchmarkopens)
        res = my_wrapping(COUNT(aa | bb, 20) / 20, self.dates, self.TD)
        return res

    @assignOrder(10)
    def alpha181(self):
        res = my_wrapping(SUM(
            self.close / DELAY(self.close, 1) - 1 - MEAN(self.close / DELAY(self.close, 1) - 1, 20) - signedpower(
                self.benchmarkclose - MEAN(self.benchmarkclose, 20), 2), 20) / SUM(
            self.benchmarkclose - signedpower(MEAN(self.benchmarkclose, 20), 3), 20), self.dates, self.TD)
        return res

    @assignOrder(11)
    def alpha180(self):
        res = -1 * self.volume
        res[MEAN(self.volume, 20) < self.volume] = -1 * TSRANK(DELTA(self.close, 7).abs(), 60) * \
                                                   np.sign(DELTA(self.close, 7))[MEAN(self.volume, 20) < self.volume]
        res = my_wrapping(res, self.dates, self.TD)
        return res

    @assignOrder(12)
    def alpha179(self):
        res = my_wrapping(RANK(CORR(self.vwap, self.volume, self.idx, 4)) * RANK(
            CORR(RANK(self.low), RANK(MEAN(self.volume, 50)), self.idx, 12)), self.dates, self.TD)
        return res

    @assignOrder(13)
    def alpha178(self):
        res = my_wrapping((self.close - DELAY(self.close, 1)) / DELAY(self.close, 1) * self.volume, self.dates, self.TD)
        return res

    @assignOrder(14)
    def alpha177(self):
        res = my_wrapping(((20 - HIGHDAY(self.high, 20)) / 20) * 100, self.dates, self.TD)
        return res

    @assignOrder(15)
    def alpha176(self):
        res = my_wrapping(
            CORR(RANK(((self.close - TSMIN(self.low, 12)) / (TSMAX(self.high, 12) - TSMIN(self.low, 12)))),
                 RANK(self.volume), self.idx, 6), self.dates, self.TD)
        return res

    @assignOrder(16)
    def alpha175(self):
        res = my_wrapping(MEAN(np.maximum(np.maximum((self.high - self.low), (DELAY(self.close, 1) - self.high).abs()),
                                          (DELAY(self.close, 1) - self.low).abs()), 6), self.dates, self.TD)
        return res

    @assignOrder(17)
    def alpha174(self):
        res = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        res[self.close > DELAY(self.close, 1)] = STD(self.close, 20)[self.close > DELAY(self.close, 1)]
        res = my_wrapping(SMA(res, 20, 1, alpha_dates, self.tickers), self.dates, self.TD)
        return res

    @assignOrder(18)
    def alpha173(self):
        res = my_wrapping(3 * SMA(self.close, 13, 2, alpha_dates, self.tickers) - 2 * SMA(
            SMA(self.close, 13, 2, alpha_dates, self.tickers), 13, 2, alpha_dates, self.tickers) + SMA(
            SMA(SMA(np.log(self.close), 13, 2, alpha_dates, self.tickers), 13, 2, alpha_dates, self.tickers), 13, 2,
            alpha_dates, self.tickers), self.dates, self.TD)
        return res

    @assignOrder(19)
    def alpha172(self):
        aa = pd.DataFrame(np.zeros_like(self.ld), index=self.ld.index, columns=self.ld.columns)
        aa[(self.ld > 0) & (self.ld > self.hd)] = self.ld[(self.ld > 0) & (self.ld > self.hd)]
        aa = SUM(aa, 14) * 100 / SUM(self.tr, 14)
        bb = pd.DataFrame(np.zeros_like(self.hd), index=self.ld.index, columns=self.ld.columns)
        bb[(self.hd > 0) & (self.hd > self.ld)] = self.hd[(self.hd > 0) & (self.hd > self.ld)]
        bb = SUM(bb, 14) * 100 / SUM(self.tr, 14)
        res = my_wrapping(MEAN((aa.abs() - bb) / (aa + bb) * 100, 6), self.dates, self.TD)
        return res

    @assignOrder(20)
    def alpha171(self):
        res = my_wrapping(-1 * (self.low - self.close) * signedpower(self.opens, 5) / (
                    (self.close - self.high) * signedpower(self.close, 5)), self.dates, self.TD)
        return res

    @assignOrder(21)
    def alpha170(self):
        res = my_wrapping((((RANK((1 / self.close)) * self.volume) / MEAN(self.volume, 20)) * ((self.high * RANK((
                self.high - self.close))) / (SUM(self.high, 5) / 5))) - RANK((self.vwap - DELAY(self.vwap, 5))),
                          self.dates, self.TD)
        return res

    @assignOrder(22)
    def alpha169(self):
        res = my_wrapping(SMA(
            MEAN(DELAY(SMA(self.close - DELAY(self.close, 1), 9, 1, alpha_dates, self.tickers), 1), 12) - MEAN(
                DELAY(SMA(self.close - DELAY(self.close, 1), 9, 1, alpha_dates, self.tickers), 1), 26), 10, 1,
            alpha_dates, self.tickers), self.dates, self.TD)
        return res

    @assignOrder(23)
    def alpha168(self):
        res = (-1 * self.volume / MEAN(self.volume, 20), self.dates, self.TD)
        return res

    @assignOrder(24)
    def alpha167(self):
        res = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        res[(self.close - DELAY(self.close, 1)) > 0] = (self.close - DELAY(self.close, 1))[(self.close - DELAY(
            self.close, 1)) > 0]
        res = my_wrapping(SUM(res, 12), self.dates, self.TD)
        return res

    @assignOrder(25)
    def alpha166(self):
        aa = self.close / DELAY(self.close, 1)
        bb = SUM(aa - 1 - MEAN(aa - 1, 20), 20)
        cc = SUM(signedpower(aa, 2), 20)
        cc = signedpower(cc, 1.5)
        res = my_wrapping(-20 * 19 ** 1.5 * bb / (19 * 18 * cc), self.dates, self.TD)
        return res

    @assignOrder(26)
    def alpha164(self):
        aa = pd.DataFrame(np.ones_like(self.close), index=self.close.index, columns=self.close.columns)
        aa[self.close > DELAY(self.close, 1)] = (1 / (self.close - DELAY(self.close, 1)))[
            self.close > DELAY(self.close, 1)]
        res = my_wrapping(
            SMA((aa - np.minimum(aa, np.full_like(aa, 12))) / (self.high - self.low) * 100, 13, 2, self.alpha_dates,
                self.tickers), self.dates, self.TD)
        return res

    @assignOrder(27)
    def alpha163(self):
        res = my_wrapping(
            RANK(((((-1 * self.returns) * MEAN(self.volume, 20)) * self.vwap) * (self.high - self.close))), self.dates,
            self.TD)
        return res

    @assignOrder(28)
    def alpha162(self):
        aa = SMA(np.maximum(self.close - DELAY(self.close, 1), np.zeros_like(self.close)), 12, 1, alpha_dates,
                 self.tickers)
        bb = SMA((self.close - DELAY(self.close, 1)).abs(), 12, 1, alpha_dates, self.tickers)
        res = my_wrapping((aa / bb * 100 - np.minimum(aa / bb * 100, np.full_like(aa, 12))) / (
                    np.maximum(aa / bb * 100, np.full_like(aa, 12)) - np.minimum(aa / bb * 100, np.full_like(aa, 12))),
                          self.dates, self.TD)
        return res

    @assignOrder(29)
    def alpha161(self):
        res = my_wrapping(MEAN(np.maximum(np.maximum((self.high - self.low), (DELAY(self.close, 1).abs() - self.high)),
                                          (DELAY(self.close, 1).abs() - self.low)), 12), self.dates, self.TD)
        return res

    @assignOrder(30)
    def alpha160(self):
        res = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        res[self.close <= DELAY(self.close, 1)] = STD(self.close, 20)[self.close <= DELAY(self.close, 1)]
        res = my_wrapping(SMA(res, 20, 1, alpha_dates, self.tickers), self.dates, self.TD)
        return res

    @assignOrder(31)
    def alpha159(self):
        aa = np.maximum(self.high, DELAY(self.close, 1)) - np.minimum(self.low, DELAY(self.close, 1))
        bb = np.minimum(self.low, DELAY(self.close, 1))
        cc = ((self.close - SUM(bb, 6)) / SUM(aa, 6) * 12 * 24 + (self.close - SUM(bb, 12)) / SUM(aa, 12) * 6 * 24 + (
                    self.close - SUM(bb, 24)) / SUM(aa, 24) * 6 * 24)
        res = my_wrapping(cc * 100 / (6 * 12 + 6 * 24 + 12 * 24), self.dates, self.TD)
        return res

    @assignOrder(32)
    def alpha158(self):
        res = my_wrapping((self.high - self.low) / self.close, self.dates, self.TD)
        return res

    @assignOrder(33)
    def alpha157(self):
        res = my_wrapping(np.minimum(
            PROD(RANK(RANK(np.log(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((self.close - 1), 5))))), 2), 1)))), 1),
            np.full_like(self.close, 5)) + TSRANK(DELAY((-1 * self.returns), 6), 5), self.dates, self.TD)
        return res

    @assignOrder(34)
    def alpha156(self):
        aa = RANK(DECAYLINEAR(DELTA(self.vwap, 5), 3))
        bb = RANK(DECAYLINEAR(
            ((DELTA(((self.opens * 0.15) + (self.low * 0.85)), 2) / ((self.opens * 0.15) + (self.low * 0.85))) - 1), 3))
        res = my_wrapping(np.maximum(aa, bb) - 1, self.dates, self.TD)
        return res

    @assignOrder(35)
    def alpha155(self):
        res = my_wrapping(SMA(self.volume, 13, 2, alpha_dates, self.tickers) - SMA(self.volume, 27, 2, alpha_dates,
                                                                                   self.tickers) - SMA(
            SMA(self.volume, 13, 2, alpha_dates, self.tickers) - SMA(self.volume, 27, 2, alpha_dates, self.tickers), 10,
            2, alpha_dates, self.tickers), self.dates, self.TD)
        return res

    @assignOrder(36)
    def alpha154(self):
        aa = self.vwap - np.minimum(self.vwap, np.full_like(self.vwap, 16))
        aa = aa.groupby(level=0).mean()
        res = aa < CORR(self.vwap, MEAN(self.volume, 180), self.idx, 18)
        res = my_wrapping(res * 1, self.dates, self.TD)
        return res

    @assignOrder(37)
    def alpha153(self):
        res = my_wrapping((MEAN(self.close, 3) + MEAN(self.close, 6) + MEAN(self.close, 12) + MEAN(self.close, 24)) / 4,
                          self.dates, self.TD)
        return res

    @assignOrder(38)
    def alpha152(self):
        aa = DELAY(SMA(DELAY(self.close / DELAY(self.close, 9), 1), 9, 1, alpha_dates, self.tickers), 1)
        res = my_wrapping(SMA(MEAN(aa, 12) - MEAN(aa, 26), 9, 1, alpha_dates, self.tickers), self.dates, self.TD)
        return res

    @assignOrder(39)
    def alpha151(self):
        res = my_wrapping(SMA(self.close - DELAY(self.close, 20), 20, 1, alpha_dates, self.tickers), self.dates,
                          self.TD)
        return res

    @assignOrder(40)
    def alpha150(self):
        res = my_wrapping((self.close + self.high + self.low) / 3 * self.volume, self.dates, self.TD)
        return res

    @assignOrder(41)
    def alpha149(self):
        aa = self.close / DELAY(self.close, 1) - 1
        cc = self.benchmarkclose >= DELAY(self.benchmarkclose, 1)
        aa[cc] = np.nan
        bb = self.benchmarkclose / DELAY(self.benchmarkclose, 1) - 1
        bb[cc] = np.nan
        res = REGBETA(aa, bb, 252)
        res = my_wrapping(pd.DataFrame(res, index=self.alpha_dates, columns=self.tickers), self.dates, self.TD)
        return res

    @assignOrder(42)
    def alpha148(self):
        aa = RANK((self.opens - TSMIN(self.opens, 14)))
        aa = aa.groupby(level=0).mean()
        res = my_wrapping((RANK(CORR(self.opens, SUM(MEAN(self.volume, 60), 9), self.idx, 6)) < aa) * -1, self.dates,
                          self.TD)
        return res

    @assignOrder(43)
    def alpha147(self):
        res = REGBETA_seq(MEAN(self.close, 12), 12, 12)
        res = my_wrapping(pd.DataFrame(res, index=self.alpha_dates, columns=self.tickers), self.dates, self.TD)
        return res

    @assignOrder(44)
    def alpha146(self):
        aa = (self.close - DELAY(self.close, 1)) / DELAY(self.close, 1)
        res = my_wrapping(MEAN(aa - SMA(aa, 61, 2, alpha_dates, self.tickers), 20) * (
                    aa - SMA(aa, 61, 2, alpha_dates, self.tickers)) / SMA(
            signedpower(aa - (aa - SMA(aa, 61, 2, alpha_dates, self.tickers)), 2), 60, 2, alpha_dates, self.tickers),
                          self.dates, self.TD)
        return res

    @assignOrder(45)
    def alpha145(self):
        res = my_wrapping((MEAN(self.volume, 9) - MEAN(self.volume, 26)) / MEAN(self.volume, 12) * 100, self.dates,
                          self.TD)
        return res

    @assignOrder(46)
    def alpha144(self):
        res = my_wrapping(SUMIF((self.close / DELAY(self.close, 1) - 1).abs() / self.amount, 20,
                                self.close < DELAY(self.close, 1)) / COUNT(self.close < DELAY(self.close, 1), 20),
                          self.dates, self.TD)
        return res

    @assignOrder(47)
    def alpha142(self):
        res = my_wrapping(-1 * RANK(TSRANK(self.close, 10)) * RANK(DELTA(DELTA(self.close, 1), 1)) * RANK(
            TSRANK((self.volume / MEAN(self.volume, 20)), 5)), self.dates, self.TD)
        return res

    @assignOrder(48)
    def alpha141(self):
        res = my_wrapping(RANK(CORR(RANK(self.high), RANK(MEAN(self.volume, 15)), self.idx, 9)) * -1, self.dates,
                          self.TD)
        return res

    @assignOrder(49)
    def alpha140(self):
        aa = RANK(DECAYLINEAR(((RANK(self.opens) + RANK(self.low)) - (RANK(self.high) + RANK(self.close))), 8))
        bb = TSRANK(DECAYLINEAR(CORR(TSRANK(self.close, 8), TSRANK(MEAN(self.volume, 60), 20), self.idx, 8), 7), 3)
        res = my_wrapping(np.minimum(aa, bb), self.dates, self.TD)
        return res

    @assignOrder(50)
    def alpha139(self):
        res = my_wrapping(-1 * CORR(self.opens, self.volume, self.idx, 10), self.dates, self.TD)
        return res

    @assignOrder(51)
    def alpha138(self):
        res = my_wrapping((RANK(DECAYLINEAR(DELTA((((self.low * 0.7) + (self.vwap * 0.3))), 3), 20)) - TSRANK(
            DECAYLINEAR(TSRANK(CORR(TSRANK(self.low, 8), TSRANK(MEAN(self.volume, 60), 17), self.idx, 5), 19), 16),
            7)) * -1, self.dates, self.TD)
        return res

    @assignOrder(52)
    def alpha137(self):
        aa = (self.high - DELAY(self.close, 1)).abs()
        bb = (self.low - DELAY(self.close, 1)).abs()
        cc = (self.high - DELAY(self.low, 1)).abs()
        dd = cc + (DELAY(self.close, 1) - DELAY(self.opens, 1)).abs() / 4
        dd[(bb > cc) & (bb > aa)] = (bb + aa / 2 + (DELAY(self.close, 1) - DELAY(self.opens, 1)).abs() / 4)[
            (bb > cc) & (bb > aa)]
        dd[(aa > bb) & (aa > cc)] = (aa + bb / 2 + (DELAY(self.close, 1) - DELAY(self.opens, 1)).abs() / 4)[
            (aa > bb) & (aa > cc)]
        res = my_wrapping(16 * (
                    self.close - DELAY(self.close, 1) + (self.close - self.opens) / 2 + DELAY(self.close, 1) - DELAY(
                self.opens, 1)) / dd * (np.maximum(aa, bb)), self.dates, self.TD)
        return res

    @assignOrder(53)
    def alpha136(self):
        res = my_wrapping(-1 * RANK(DELTA(self.returns, 3)) * CORR(self.opens, self.volume, self.idx, 10), self.dates,
                          self.TD)
        return res

    @assignOrder(54)
    def alpha135(self):
        res = my_wrapping(-1 * SMA(DELAY(self.close / DELAY(self.close, 20), 1), 20, 1, alpha_dates, self.tickers),
                          self.dates, self.TD)
        return res

    @assignOrder(55)
    def alpha134(self):
        res = my_wrapping((self.close - DELAY(self.close, 12)) / DELAY(self.close, 12) * self.volume, self.dates,
                          self.TD)
        return res

    @assignOrder(56)
    def alpha133(self):
        res = my_wrapping((20 - HIGHDAY(self.high, 20)) / 20 * 100 - (20 - LOWDAY(self.low, 20)) / 20 * 100, self.dates,
                          self.TD)
        return res

    @assignOrder(57)
    def alpha132(self):
        res = my_wrapping(MEAN(self.amount, 20), self.dates, self.TD)
        return res

    @assignOrder(58)
    def alpha131(self):
        aa = RANK(DELAT(self.vwap, 1))
        aa = aa.groupby(level=0).mean()
        res = my_wrapping(signedpower(aa, TSRANK(CORR(self.close, MEAN(self.volume, 50), self.idx, 18), 18)),
                          self.dates, self.TD)
        return res

    @assignOrder(59)
    def alpha130(self):
        aa = RANK(DECAYLINEAR(CORR(((self.high + self.low) / 2), MEAN(self.volume, 40), self.idx, 9), 10))
        bb = RANK(DECAYLINEAR(CORR(RANK(self.vwap), RANK(self.volume), self.idx, 7), 3))
        res = my_wrapping(aa / bb, self.dates, self.TD)
        return res

    @assignOrder(60)
    def alpha129(self):
        res = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        res[(self.close - DELAY(self.close, 1)) < 0] = (self.close - DELAY(self.close, 1)).abs()[
            (self.close - DELAY(self.close, 1)) < 0]
        res = my_wrapping(SUM(res, 12), self.dates, self.TD)
        return res

    @assignOrder(61)
    def alpha128(self):
        aa = pd.DataFrame(np.zeros_like(self.high), index=self.high.index, columns=self.high.columns)
        bb = pd.DataFrame(np.zeros_like(self.high), index=self.high.index, columns=self.high.columns)
        aa[((self.high + self.low + self.close) / 3) > DELAY((self.high + self.low + self.close) / 3, 1)] = ((
                                                                                                                         self.high + self.low + self.close) / 3) * \
                                                                                                            self.volume[
                                                                                                                ((
                                                                                                                             self.high + self.low + self.close) / 3) > DELAY(
                                                                                                                    (
                                                                                                                                self.high + self.low + self.close) / 3,
                                                                                                                    1)]
        bb[((self.high + self.low + self.close) / 3) < DELAY((self.high + self.low + self.close) / 3, 1)] = ((
                                                                                                                         self.high + self.low + self.close) / 3) * \
                                                                                                            self.volume[
                                                                                                                ((
                                                                                                                             self.high + self.low + self.close) / 3) < DELAY(
                                                                                                                    (
                                                                                                                                self.high + self.low + self.close) / 3,
                                                                                                                    1)]
        cc = 1 + SUM(aa, 14) / SUM(bb, 14)
        res = my_wrapping(100 - (100 / cc), self.dates, self.TD)
        return res

    @assignOrder(62)
    def alpha127(self):
        res = my_wrapping(signedpower(signedpower(MEAN(
            100 * (self.close - np.maximum(self.close, np.full_like(self.close, 12))) / (
                np.maximum(self.close, np.full_like(self.close, 12))), 12), 2), 0.5), self.dates, self.TD)
        return res

    @assignOrder(63)
    def alpha126(self):
        res = my_wrapping((self.close + self.high + self.low) / 3, self.dates, self.TD)
        return res

    @assignOrder(64)
    def alpha125(self):
        res = my_wrapping(RANK(DECAYLINEAR(CORR(self.vwap, MEAN(self.volume, 80), self.idx, 17), 20)) / RANK(
            DECAYLINEAR(DELTA(((self.close * 0.5) + (self.vwap * 0.5)), 3), 16)), self.dates, self.TD)
        return res

    @assignOrder(65)
    def alpha124(self):
        res = my_wrapping((self.close - self.vwap) / DECAYLINEAR(RANK(TSMAX(self.close, 30)), 2)), self.dates, self.TD
        return res

    @assignOrder(66)
    def alpha123(self):
        res = my_wrapping(
            RANK(CORR(SUM(((self.high + self.low) / 2), 20), SUM(MEAN(self.volume, 60), 20), self.idx, 9)), self.dates,
            self.TD)
        return res

    @assignOrder(67)
    def alpha122(self):
        aa = SMA(SMA(SMA(np.log(self.close), 13, 2, alpha_dates, self.tickers), 13, 2, alpha_dates, self.tickers), 13,
                 2, alpha_dates, self.tickers)
        res = my_wrapping(aa / DELAY(aa, 1) - 1, self.dates, self.TD)
        return res

    @assignOrder(68)
    def alpha121(self):
        aa = (RANK((self.vwap - np.minimum(self.vwap, np.full_like(self.vwap, 12))))).groupby(level=0).mean()
        res = my_wrapping(signedpower(aa, TSRANK(
            CORR(TSRANK(self.vwap, 20), TSRANK(MEAN(self.volume, 60), 2), self.idx, 18), 3)) * -1, self.dates, self.TD)
        return res

    @assignOrder(69)
    def alpha120(self):
        res = my_wrapping(RANK(self.vwap - self.close) / RANK(self.vwap + self.close), self.dates, self.TD)
        return res

    @assignOrder(70)
    def alpha119(self):
        aa = RANK(DECAYLINEAR(CORR(self.vwap, SUM(MEAN(self.volume, 5), 26), self.idx, 5), 7))
        bb = RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(self.opens), RANK(MEAN(self.volume, 15)), self.idx, 21), 9), 7), 8))
        res = my_wrapping(aa - bb, self.dates, self.TD)
        return res

    @assignOrder(71)
    def alpha118(self):
        res = my_wrapping(SUM(self.high - self.opens, 20) / SUM(self.opens - self.low, 20) * 100, self.dates, self.TD)
        return res

    @assignOrder(72)
    def alpha117(self):
        res = my_wrapping(TSRANK(self.volume, 32) * (1 - TSRANK(((self.close + self.high) - self.low), 16)) * (
                    1 - TSRANK(self.returns, 32)), self.dates, self.TD)
        return res

    @assignOrder(73)
    def alpha116(self):
        """
        regress 20-each ticker against the sequence [1, 2, ..., 20]
        """
        res = REGBETA_seq(self.close, 20, 20)
        res = my_wrapping(pd.DataFrame(res, index=alpha_dates, columns=self.tickers), self.dates, self.TD)
        return res

    @assignOrder(74)
    def alpha115(self):
        res = my_wrapping(
            signedpower(RANK(CORR(((self.high * 0.9) + (self.close * 0.1)), MEAN(self.volume, 30), self.idx, 10)),
                        RANK(CORR(TSRANK(((self.high + self.low) / 2), 4), TSRANK(self.volume, 10), self.idx, 7))),
            self.dates, self.TD)
        return res

    @assignOrder(75)
    def alpha114(self):
        res = my_wrapping(
            (RANK(DELAY(((self.high - self.low) / (SUM(self.close, 5) / 5)), 2)) * RANK(RANK(self.volume))) / (
                        ((self.high - self.low) / (SUM(self.close, 5) / 5)) / (self.vwap - self.close)), self.dates,
            self.TD)
        return res

    @assignOrder(76)
    def alpha113(self):
        res = my_wrapping(-1 * (
                    (RANK((SUM(DELAY(self.close, 5), 20) / 20)) * CORR(self.close, self.volume, self.idx, 2)) * RANK(
                CORR(SUM(self.close, 5), SUM(self.close, 20), self.idx, 2))), self.dates, self.TD)
        return res

    @assignOrder(77)
    def alpha112(self):
        aa = (self.close - DELAY(self.close, 1)) > 0
        bb = (self.close - DELAY(self.close, 1)) < 0
        cc = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        dd = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        cc[aa] = (self.close - DELAY(self.close, 1))[aa]
        cc[bb] = (self.close - DELAY(self.close, 1)).abs()[bb]
        res = my_wrapping((SUM(cc, 12) - SUM(dd, 12)) / (SUM(cc, 12) + SUM(dd, 12)) * 100, self.dates, self.TD)
        return res

    @assignOrder(78)
    def alpha111(self):
        res = my_wrapping(
            SMA(self.volume * ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low), 11, 2,
                alpha_dates, self.tickers) - SMA(
                self.volume * ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low), 4, 2,
                alpha_dates, self.tickers), self.dates, self.TD)
        return res

    @assignOrder(79)
    def alpha110(self):
        res = my_wrapping(SUM(np.maximum(self.high - DELAY(self.close, 1), np.zeros_like(self.close)), 20) / SUM(
            np.maximum(DELAY(self.close, 1) - self.low, np.zeros_like(self.close)), 20) * 100, self.dates, self.TD)
        return res

    @assignOrder(80)
    def alpha109(self):
        res = my_wrapping(SMA(self.high - self.low, 10, 2, alpha_dates, self.tickers) / SMA(
            SMA(self.high - self.low, 10, 2, alpha_dates, self.tickers), 10, 2, alpha_dates, self.tickers), self.dates,
                          self.TD)
        return res

    @assignOrder(81)
    def alpha108(self):
        aa = RANK((self.high - np.minimum(self.high, np.full_like(self.high, 2))))
        aa = aa.groupby(level=0).mean()
        res = my_wrapping(signedpower(aa, RANK(CORR((self.vwap), (MEAN(self.volume, 120)), self.idx, 6))) * -1,
                          self.dates, self.TD)
        return res

    @assignOrder(82)
    def alpha107(self):
        res = my_wrapping(-1 * RANK(self.opens - DELAY(self.high, 1)) * RANK(self.opens - DELAY(self.close, 1)) * RANK(
            self.opens - DELAY(self.low, 1)), self.dates, self.TD)
        return res

    @assignOrder(83)
    def alpha106(self):
        res = my_wrapping(self.close - DELAY(self.close, 20), self.dates, self.TD)
        return res

    @assignOrder(84)
    def alpha105(self):
        res = my_wrapping(-1 * CORR(RANK(self.opens), RANK(self.volume), self.idx, 10), self.dates, self.TD)
        return res

    @assignOrder(85)
    def alpha104(self):
        res = my_wrapping(-1 * DELTA(CORR(self.high, self.volume, self.idx, 5), 5) * RANK(STD(self.close, 20)),
                          self.dates, self.TD)
        return res

    @assignOrder(86)
    def alpha103(self):
        res = my_wrapping(((20 - LOWDAY(self.low, 20)) / 20) * 100, self.dates, self.TD)
        return res

    @assignOrder(87)
    def alpha102(self):
        res = my_wrapping(
            SMA(np.maximum(self.volume - DELAY(self.volume, 1), np.zeros_like(self.volume)), 6, 1, alpha_dates,
                self.tickers) / SMA((self.volume - DELAY(self.volume, 1)).abs(), 6, 1, alpha_dates, self.tickers) * 100,
            self.dates, self.TD)
        return res

    @assignOrder(88)
    def alpha101(self):
        res = my_wrapping(RANK(CORR(self.close, SUM(MEAN(self.volume, 30), 37), self.idx, 15)), self.dates, self.TD)
        return res

    @assignOrder(89)
    def alpha100(self):
        res = my_wrapping(STD(self.volume, 20), self.dates, self.TD)
        return res

    @assignOrder(90)
    def alpha099(self):
        res = my_wrapping(-1 * RANK(COV(RANK(self.close), RANK(self.volume), self.idx, 5)), self.dates, self.TD)
        return res

    @assignOrder(91)
    def alpha098(self):
        aa = (DELTA((SUM(self.close, 100) / 100), 100) / DELAY(self.close, 100)) <= 0.05
        res = -1 * DELTA(self.close, 3)
        res[aa] = (-1 * (self.close - TSMIN(self.close, 100)))[aa]
        res = my_wrapping(res, self.dates, self.TD)
        return res

    @assignOrder(92)
    def alpha097(self):
        res = my_wrapping(STD(self.volume, 10), self.dates, self.TD)
        return res

    @assignOrder(93)
    def alpha096(self):
        return wrapping(SMA(
            SMA((self.close - TSMIN(self.low, 9)) / (TSMAX(self.high, 9) - TSMIN(self.low, 9)) * 100, 3, 1, alpha_dates,
                self.tickers), 3, 1, alpha_dates, self.tickers), self.dates, self.TD)

    def alpha095(self):
        return wrapping(STD(self.amount, 20), self.dates, self.TD)



    def alpha094(self):
        res = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        res[self.close < DELAY(self.close, 1)] = -1 * self.volume[self.close < DELAY(self.close, 1)]
        res[self.close > DELAY(self.close, 1)] = self.volume[self.close > DELAY(self.close, 1)]
        return wrapping(SUM(res, 30), self.dates, self.TD)


    def alpha093(self):
        res = np.maximum((self.opens - self.low), (self.opens - DELAY(self.opens, 1)))
        res[self.opens >= DELAY(self.opens, 1)] = 0
        return wrapping(SUM(res, 20), self.dates, self.TD)

    def alpha092(self):
        return wrapping(-1 * np.maximum(RANK(DECAYLINEAR(DELTA(((self.close * 0.35) + (self.vwap * 0.65)), 2), 3)),
                                          TSRANK(DECAYLINEAR(
                                              (CORR((MEAN(self.volume, 180)), self.close, self.idx, 13)).abs(), 5),
                                                 15)), self.dates, self.TD)



    def alpha091(self):
        return wrapping(-1 * RANK((self.close - np.maximum(self.close, np.full_like(self.close, 5)))) * RANK(
            CORR((MEAN(self.volume, 40)), self.low, self.idx, 5)), self.dates, self.TD)


    def alpha090(self):
        return wrapping(-1 * RANK(CORR(RANK(self.vwap), RANK(self.volume), self.idx, 5)), self.dates, self.TD)


    def __init__(self,update_date):
        dates=np.load('/shared/FactorBank/NewMinData/5Min_Data/alpha_dates.npy')
        date_index=dates.searchsorted(update_date)
        open = np.load('/shared/FactorBank/NewMinData/open.npy')[date_index-48*48:]
        high = np.load('/shared/FactorBank/NewMinData/high.npy')[date_index-48*48:]
        low = np.load('/shared/FactorBank/NewMinData/low.npy')[date_index-48*48:]
        close = np.load('/shared/FactorBank/NewMinData/close.npy')[date_index-48*48:]
        volume = np.load('/shared/FactorBank/NewMinData/volume.npy')[date_index-48*48:]
        value = np.load('/shared/FactorBank/NewMinData/value.npy')[date_index-48*48:]
        vwap = np.load('vwap.npy')[date_index-48*48:]
        self.OPEN = pd.DataFrame(open)
        self.VWAP = pd.DataFrame(vwap)
        self.VALUE = pd.DataFrame(value)
        self.HIGH = pd.DataFrame(high)
        self.LOW = pd.DataFrame(low)
        self.CLOSE= pd.DataFrame(close)
        self.VOLUME = pd.DataFrame(volume)

if __name__ == '__main__':
    alpha = Alphas(update_date)
    dates = np.load('/shared/FactorBank/NewMinData/5Min_Data/alpha_dates.npy')
    new_ticker = np.load('/home/hwzhang/ticker_names.npy')
    new_data=pd.DataFrame(alpha.alpha124()[-2:-1],colums=new_ticker,index=update_date)
    old_ticker = np.load('/home/hwzhang/ticker_names.npy')
    old_data=pd.DataFrame(np.load('/home/hwzhang/1.npy'),colums=old_ticker,index=dates)
    l=[np.nan]*len(dates)
    for i in new_ticker:
        if i not in old_ticker:
            old_data.insert(0,i,l)
    res=pd.concat(old_data[new_ticker],new_data)
    np.save('alpha.npy')




