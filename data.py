# coding: utf-8
sampleRate=6000
import datetime
import pandas as pd
from pykalman import KalmanFilter
import os
import numpy as np
import time


def daylight(x,y):
    if int(x) == 12:
        if int(y) > 0:
            return "NOON" #12.30
        return "DAY" #12.00
    if int(x) == 06:
        if int(y) > 0:
            return "DAY" #06.30
        return "NIGHT" #06.00
    if int(x) == 18:
        if int(y) > 0:
            return "EVENING" #18.30
        return "NOON" #18.00
    if int(x) == 0:
            return "NIGHT" #00.30
    if (int(x) < 12 and int(x) > 06):
        return "DAY"
    elif int(x) > 12 and int(x) < 18:
        return "NOON"
    elif int(x) > 18 and int(x) < 24:
        return "EVENING"
    return "NIGHT"


def driving_event(x,speed_threshold,stop_threshold):
    if x > speed_threshold:
        return "A"
    elif x < 0 and abs(x) > stop_threshold:
        return "D"
#    elif abs(x) == 0:
#        return "STOP"
    return "S"


def percentage(x,y):
    return float(x)/float(y) * 100


def latest_val_of_cusum(df_x,df_y=0,df_z=0):
    df_rms = (df_x ** 2 + (1.4*df_y )** 2 + (df_z*1.4) ** 2) ** 0.5
    return df_rms.cumsum().tail(1).values[0]


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def rolling_window(a, step):
    shape   = a.shape[:-1] + (a.shape[-1] - step + 1, step)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def get_kf_value(y_values):
    kf = KalmanFilter()
    Kc, Ke = kf.em(y_values, n_iter=1).smooth(0)
    return Kc

def filter_with_kalman(df_x,column_to_be_filtered,label_kalman_filtered_column,wsize=3):
    arr = rolling_window(df_x[column_to_be_filtered].values, wsize)
    zero_padding = np.zeros(shape=(wsize-1,wsize))
    arrst = np.concatenate((zero_padding, arr))
    arrkalman = np.zeros(shape=(len(arrst),1))

    for i in range(len(arrst)):
        arrkalman[i] = get_kf_value(arrst[i])

    kalmandf = pd.DataFrame(arrkalman, columns=[label_kalman_filtered_column], index=index)
    return pd.concat([df_x, kalmandf], axis=1)

current_milli_time = lambda: int(round(time.time() * 1000))


def millisecond_to_minute(end,start):
    return ((end - start) / 1000*60) % 60

def moving_average_1(signal, period):
    buffer = [np.nan] * period
    for i in range(period,len(signal)):
        buffer.append(signal[i-period:i].mean())
    return buffer

execution_time_start = current_milli_time()

constructed = pd.DataFrame()
pointer = 0
# experimentally setted
SPEED_THRESHOLD = 2
STOP_THRESHOLD = 2
# an article ? but which ? may be found ???
LATERAL_THRESHOLD=0.145
df_grouped_total=pd.DataFrame()
# coefficients are taken from the article named at the next line
# Ride Comfort of Passenger Cars on
# Two-Lane Mountain Highways Based on
# Tri-axial Acceleration from Field Driving Tests

IDEAL_ACC_COMFORT_COEFFICIENT=0.25
IDEAL_DEC_COMFORT_COEFFICIENT=0.325
IDEAL_LATERAL_COMFORT_COEFFICIENT=0.412
IDEAL_VERTICAL_COMFORT_COEFFICIENT=0.1575

#IDEAL_ACC_COMFORT_COEFFICIENT=0.25
#IDEAL_DEC_COMFORT_COEFFICIENT=0.325
#IDEAL_LATERAL_COMFORT_COEFFICIENT=0.412
#IDEAL_VERTICAL_COMFORT_COEFFICIENT=0.1575


#path = "/Users/firat/Documents/DATASET/transport-tipleri/otobüs"
_path = "/Users/firat/Documents/YHT"
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
available_files={}
labels={}
dict_df={}
for root, dirs, files in os.walk(_path):
    for filename in files:
        if '.csv' in filename:
            available_files[filename] = os.path.join(root, filename)
            labels[filename]=root.split("/").pop()

index=0
for filename,path in available_files.items():
        try:
            index=index+1
            print("~~~~~~~~~~~~~~~~PROGRESS\t"+str(len(available_files))+"/"+str(index)+"~~~~~~~~~~~~~~~~")
            df_grouped_transformed=pd.DataFrame(columns=["MEAN_LINX","MEAN_LINY","MEAN_LINZ"])
            print(filename)
            df = pd.read_csv(path, low_memory=False)
            length=len(df)
            print(length)
            ts = time.time()
            asd=df['SAMPLETIME']
            df["LINEARACCX"] = df["LINEARACCX"].abs()
            df["LINEARACCY"] = df["LINEARACCY"].abs()
            df["LINEARACCZ"] = df["LINEARACCZ"].abs()

            # df["LINEARACCX"] = df["LINEARACCX"].rolling(window=10).mean()
            # df["LINEARACCY"] = df["LINEARACCY"].rolling(window=10).mean()
            # df["LINEARACCZ"] = df["LINEARACCZ"].rolling(window=10).mean()

            #kalman
            # wsize = 3
            # arr = rolling_window(df.LINEARACCZ.values, wsize)
            # zero_padding = np.zeros(shape=(wsize-1,wsize))
            # arrst = np.concatenate((zero_padding, arr))
            # arrkalman = np.zeros(shape=(len(arrst),1))
            #
            # for i in range(len(arrst)):
            #     arrkalman[i] = get_kf_value(arrst[i])
            #
            # kalmandf = pd.DataFrame(arrkalman, columns=['D_LINEARACCZ'], index=index)
            # df = pd.concat([df,kalmandf], axis=1)
            #kalman

#            df = filter_with_kalman(df,"LINEARACCX","K_LINEARACCX")
#            df = filter_with_kalman(df,"LINEARACCY","K_LINEARACCY")
#            df = filter_with_kalman(df,"LINEARACCZ","K_LINEARACCZ")

            df["SPEEDINKMH"] = df["SPEEDINKMH"].fillna(0)
            df["SPEEDINKMH"] = df["SPEEDINKMH"].apply(lambda x: x * 3.6)

            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
            times = pd.to_datetime(asd,format='%Y-%m-%d %H:%M:%S:%f')
            df_grouped=df.groupby([times.dt.hour, times.dt.minute,times.dt.second])
            df_grouped=df_grouped["LINEARACCX",
                                  "LINEARACCY",
                                  "LINEARACCZ",
                                  "GYROSCOPEX",
                                  "GYROSCOPEY",
                                  "GYROSCOPEZ",
                                  "SPEEDINKMH",
                                  "LAT",
                                  "LON"].mean()
            df_grouped = df_grouped.apply(lambda x: x)
            df_grouped["DATASET"]=filename
            # mean of acceleration
            mean_linx = df_grouped["LINEARACCX"].mean()
            mean_liny = df_grouped["LINEARACCY"].mean()
            mean_linz = df_grouped["LINEARACCZ"].mean()
            # standard deviation of acceleration
            std_linx = df_grouped["LINEARACCX"].std()
            std_liny = df_grouped["LINEARACCY"].std()
            std_linz = df_grouped["LINEARACCZ"].std()
            # # acceleration jerk values
            mean_jerk_linx = df_grouped["LINEARACCX"].diff().mean()
            mean_jerk_liny = df_grouped["LINEARACCY"].diff().mean()
            mean_jerk_linz = df_grouped["LINEARACCZ"].diff().mean()
            # # mean speed
            mean_speed = df_grouped["SPEEDINKMH"].mean()
            # # duration by minute
            duration = len(df_grouped) / 60
            # # ISO 2631 compatible RMS value
#            iso2631 = np.sqrt(mean_linx**2 +
#                              (1.4*mean_liny)**2 +
#                              (1.4*mean_linz)**2)
            iso2631 = np.sqrt(mean_linx**2 +
                              (mean_liny)**2 +
                              (mean_linz)**2)


            # a = (df_grouped["LINEARACCX"]**2 +
            #           df_grouped["LINEARACCY"]**2 +
            #           df_grouped["LINEARACCZ"]**2) ** 0.5
            # a = (df_grouped["LINEARACCY"]**2 +
            #           df_grouped["LINEARACCZ"]**2 +
            #           df_grouped["LINEARACCX"]**2) ** 0.5
            # actual_load_disturbance = a.cumsum().tail(1).values[0]

            b = df_grouped["LINEARACCX"]
            actual_road_disturbance = df_grouped["LINEARACCX"].cumsum().tail(1).values[0]



            # manuever count
            manuever_count = len(df_grouped[df_grouped["GYROSCOPEZ"] > LATERAL_THRESHOLD])

            #hizlanma, yavaslama, duraksama-durma olaylarinin ortalamalari
            # "F" -> bir onceki ile bir sonraki ornekteki SPEEDINMKH sample i arasindaki fark.
            #IN_MOTION, hiz degisimi var anlaminda.
            #IN_MOTION == False ise iki durum gecerli, ya sabit hizda gidiyor ya da duruyor
            #bu durumu anlamak icin SPEEDINKMH daki veriye bakilir. STOP_THRESHOLD dan buyukse sabit hiz.
            #kucuk veya esit ise arac duruyor
            df_grouped["E"]=df_grouped["SPEEDINKMH"].shift(1)
            df_grouped["F"] = (df_grouped["E"]-df_grouped["SPEEDINKMH"])*-1
            df_grouped["IN_MOTION"]= df_grouped["F"].apply(lambda x: driving_event(x, SPEED_THRESHOLD, STOP_THRESHOLD)) # duruyor ya da hareketli
            df_grouped.loc[df_grouped["SPEEDINKMH"] == 0, 'IN_MOTION'] = "STOP"
            df_grouped["MANUEVER"]=df_grouped["GYROSCOPEZ"].apply(lambda x: True if x > LATERAL_THRESHOLD else False)

            acc_count = len(df_grouped[df_grouped["IN_MOTION"] == "A"])
            dec_count = len(df_grouped[df_grouped["IN_MOTION"] == "D"])
            stable_count = len(df_grouped[df_grouped["IN_MOTION"] == "S"])
            stop_count = len(df_grouped[df_grouped["IN_MOTION"] == "STOP"])
            total_events = acc_count + dec_count + stable_count + stop_count

            acc_percentage = percentage(acc_count, total_events)
            dec_percentage = percentage(dec_count, total_events)
            stable_percentage = percentage(stable_count, total_events)
            stop_percentage = percentage(stop_count, total_events)

            # comfort scoring
            # sliced_by_speed = df_grouped.groupby(pd.cut(df_grouped["SPEEDINKMH"], np.arange(0, 200, 10)))["LINEARACCZ"]
            sample_count=len(df_grouped)

            # longitudinal_calculated_comfort = (sliced_by_speed.size()*sliced_by_speed.mean()).sum()
            lateral_calculated_comfort = LATERAL_THRESHOLD * manuever_count
            vertical_calculated_comfort = mean_linx * sample_count


            ideal_load_disturbance = (np.sqrt(IDEAL_ACC_COMFORT_COEFFICIENT**2+IDEAL_VERTICAL_COMFORT_COEFFICIENT**2) * acc_count) + \
                                     (np.sqrt(IDEAL_DEC_COMFORT_COEFFICIENT**2+IDEAL_VERTICAL_COMFORT_COEFFICIENT**2)*dec_count) + \
                                     (np.sqrt(IDEAL_VERTICAL_COMFORT_COEFFICIENT**2+IDEAL_LATERAL_COMFORT_COEFFICIENT**2)*manuever_count) + \
                                     (np.sqrt(((IDEAL_ACC_COMFORT_COEFFICIENT+IDEAL_DEC_COMFORT_COEFFICIENT)/2)**2 + IDEAL_VERTICAL_COMFORT_COEFFICIENT**2 ) *stable_count)
            # acc ve dec lerden donus eventlarının çıkartılması gerekli.


            ideal_road_disturbance = sample_count * IDEAL_VERTICAL_COMFORT_COEFFICIENT

            ideal_road_disturbance_score = (ideal_road_disturbance * 100) / actual_road_disturbance

            # comfort_score = (sample_count * IDEAL_LONGIDUTINAL_COMFORT_COEFFICIENT * 100) / (longitudinal_calculated_comfort +
            #                                                                     lateral_calculated_comfort +
            #                                                                     vertical_calculated_comfort)
            stable_events = df_grouped[df_grouped["IN_MOTION"] == "S"]
            stop_events = df_grouped[df_grouped["IN_MOTION"] == "STOP"]
            acc_events = df_grouped[df_grouped["IN_MOTION"] == "A"]
            dec_events = df_grouped[df_grouped["IN_MOTION"] == "D"]
            manuever_events = df_grouped[df_grouped["GYROSCOPEZ"] > LATERAL_THRESHOLD]


            stable_longitudinal_mean = stable_events["LINEARACCZ"].mean()
            stop_longitudinal_mean = stop_events["LINEARACCZ"].mean()
            acc_longitudinal_mean = acc_events["LINEARACCZ"].mean()
            dec_longitudinal_mean = dec_events["LINEARACCZ"].mean()
            manuever_longitudinal_mean = manuever_events["LINEARACCZ"].mean()

            stable_longitudinal_std = stable_events["LINEARACCZ"].std()
            stop_longitudinal_std = stop_events["LINEARACCZ"].std()
            acc_longitudinal_std = acc_events["LINEARACCZ"].std()
            dec_longitudinal_std = dec_events["LINEARACCZ"].std()
            manuever_longitudinal_std = manuever_events["LINEARACCZ"].std()

            stable_vertical_mean = stable_events["LINEARACCX"].mean()
            stop_vertical_mean = stop_events["LINEARACCX"].mean()
            acc_vertical_mean = acc_events["LINEARACCX"].mean()
            dec_vertical_mean = dec_events["LINEARACCX"].mean()
            manuever_vertical_mean = manuever_events["LINEARACCX"].mean()

            stable_longitudinal_cusum = latest_val_of_cusum(stable_events["LINEARACCX"],stable_events["LINEARACCZ"])
            stop_longitudinal_cusum = latest_val_of_cusum(stop_events["LINEARACCX"],stop_events["LINEARACCZ"])
            acc_longitudinal_cusum = latest_val_of_cusum(acc_events["LINEARACCX"],acc_events["LINEARACCZ"])
            dec_longitudinal_cusum = latest_val_of_cusum(dec_events["LINEARACCX"],dec_events["LINEARACCZ"])
            manuever_cusum = latest_val_of_cusum(manuever_events["LINEARACCX"],manuever_events["LINEARACCY"])

            actual_load_disturbance = acc_longitudinal_cusum + dec_longitudinal_cusum + manuever_cusum + stable_longitudinal_cusum
            ideal_load_disturbance_score = (ideal_load_disturbance * 100) / actual_load_disturbance


            vertical_cusum = df_grouped["LINEARACCX"].cumsum().tail(1).values[0]

            stable_longitudinal_cusum_percentage=percentage(stable_longitudinal_cusum,actual_load_disturbance)
            stop_longitudinal_cusum_percentage=percentage(stop_longitudinal_cusum,actual_load_disturbance)
            acc_longitudinal_cusum_percentage=percentage(acc_longitudinal_cusum,actual_load_disturbance)
            dec_longitudinal_cusum_percentage=percentage(dec_longitudinal_cusum,actual_load_disturbance)
            vertical_cusum_percentage=percentage(vertical_cusum,actual_load_disturbance)

            end_time = df[-1:]["SAMPLETIME"].min()
            start_time = df[1:]["SAMPLETIME"].min()
            label = labels[filename]
            a = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S:%f")
            timezone = daylight(a.hour, a.minute)

            columns=["MEAN_LINX","MEAN_LINY","STD_LINX","STD_LINY","STD_LINZ",
                     "MEAN_JERK_LINX","MEAN_JERK_LINY","MEAN_JERK_LINZ",
                     "MEAN_SPEED","DURATION","DATASET","ISO2631",
                     "MANUEVER_COUNT","ACC_COUNT","DEC_COUNT","STABLE_COUNT","TOTAL_COUNT",
                     "ACTUAL_LOAD_DISTURBANCE","IDEAL_LOAD_DISTURBANCE","IDEAL_LOAD_DISTURBANCE_SCORE",
                     "ACTUAL_ROAD_DISTURBANCE","IDEAL_ROAD_DISTURBANCE","IDEAL_ROAD_DISTURBANCE_SCORE",
                     "SAMPLE_COUNT","STOP_COUNT",
                     "ACC_PERCENTAGE","DEC_PERCENTAGE","STOP_PERCENTAGE","STABLE_PERCENTAGE",
                     "STABLE_LONG_MEAN","STOP_LONG_MEAN","ACC_LONG_MEAN","DEC_LONG_MEAN",
                     "STABLE_LONG_STD","STOP_LONG_STD","ACC_LONG_STD","DEC_LONG_STD",
                     "ACC_LONG_CUSUM","DEC_LONG_CUSUM","STABLE_LONG_CUSUM","STOP_LONG_CUSUM,MANUEVER_CUSUM",
                     "STABLE_VERTICAL_MEAN","ACC_VERTICAL_MEAN","DEC_VERTICAL_MEAN","STOP_VERTICAL_MEAN",
                     "VERTICAL_CUSUM",
                     "STABLE_LONG_CUSUM_PERC",
                     "ACC_LONG_CUSUM_PERC",
                     "DEC_LONG_CUSUM_PERC",
                     "STOP_LONG_CUSUM_PERC",
                     "VERTICAL_CUSUM_PERC",
                     "START_TIME","END_TIME","TIMEZONE","LABEL"]

            constructed=constructed.append(pd.DataFrame({
                                                         "MEAN_LINX":mean_linx,
                                                         "MEAN_LINY":mean_liny,
                                                         "MEAN_LINZ":mean_linz,
                                                         "STD_LINX":std_linx,
                                                         "STD_LINY":std_liny,
                                                         "STD_LINZ":std_linz,
                                                         "MEAN_JERK_LINX":mean_jerk_linx,
                                                         "MEAN_JERK_LINY":mean_jerk_liny,
                                                         "MEAN_JERK_LINZ":mean_jerk_linz,
                                                         "MEAN_SPEED":mean_speed,
                                                         "DATASET":filename,
                                                         "DURATION":duration,
                                                         "ISO2631":iso2631,
                                                         "MANUEVER_COUNT":manuever_count,
                                                         "ACC_COUNT":acc_count,
                                                         "DEC_COUNT":dec_count,
                                                         "STABLE_COUNT":stable_count,
                                                         "TOTAL_COUNT":total_events,
                                                         "ACTUAL_LOAD_DISTURBANCE":actual_load_disturbance,
                                                         "IDEAL_LOAD_DISTURBANCE":ideal_load_disturbance,
                                                         "IDEAL_LOAD_DISTURBANCE_SCORE": ideal_load_disturbance_score,
                                                         "ACTUAL_ROAD_DISTURBANCE":actual_road_disturbance,
                                                         "IDEAL_ROAD_DISTURBANCE":ideal_road_disturbance,
                                                         "IDEAL_ROAD_DISTURBANCE_SCORE":ideal_road_disturbance_score,
                                                         "SAMPLE_COUNT":sample_count,
                                                         "STOP_COUNT":stop_count,
                                                         "ACC_PERCENTAGE":acc_percentage,
                                                         "DEC_PERCENTAGE":dec_percentage,
                                                         "STOP_PERCENTAGE":stop_percentage,
                                                         "STABLE_PERCENTAGE":stable_percentage,
                                                         "STABLE_LONG_MEAN":stable_longitudinal_mean,
                                                         "STOP_LONG_MEAN":stop_longitudinal_mean,
                                                         "ACC_LONG_MEAN":acc_longitudinal_mean,
                                                         "DEC_LONG_MEAN":dec_longitudinal_mean,
                                                         "STABLE_LONG_STD":stable_longitudinal_std,
                                                         "STOP_LONG_STD":stop_longitudinal_std,
                                                         "ACC_LONG_STD":acc_longitudinal_std,
                                                         "DEC_LONG_STD":dec_longitudinal_std,
                                                         "ACC_LONG_CUSUM":acc_longitudinal_cusum,
                                                         "DEC_LONG_CUSUM":dec_longitudinal_cusum,
                                                         "STABLE_LONG_CUSUM":stable_longitudinal_cusum,
                                                         "STOP_LONG_CUSUM":stop_longitudinal_cusum,
                                                         "MANUEVER_CUSUM":manuever_cusum,
                                                         "STABLE_VERTICAL_MEAN":stable_vertical_mean,
                                                         "ACC_VERTICAL_MEAN":acc_vertical_mean,
                                                         "DEC_VERTICAL_MEAN":dec_vertical_mean,
                                                         "STOP_VERTICAL_MEAN":stop_vertical_mean,
                                                         "VERTICAL_CUSUM":vertical_cusum,
                                                         "STABLE_LONG_CUSUM_PERC":stable_longitudinal_cusum_percentage,
                                                         "ACC_LONG_CUSUM_PERC":acc_longitudinal_cusum_percentage,
                                                         "DEC_LONG_CUSUM_PERC":dec_longitudinal_cusum_percentage,
                                                         "STOP_LONG_CUSUM_PERC":stop_longitudinal_cusum_percentage,
                                                         "VERTICAL_CUSUM_PERC":vertical_cusum_percentage,
                                                         "START_TIME":start_time,
                                                         "END_TIME":end_time,
                                                         "TIMEZONE":timezone,
                                                         "LABEL":label},index=[filename],columns=columns))
            pointer = pointer + 1
            df_grouped.to_csv(_path + "/_TEST - " + filename + "_" + st + ".csv", encoding='utf-8')
            df_grouped_total=df_grouped_total.append(df_grouped)
            dict_df[filename]=df_grouped
        except Exception,e:
            print("NAN"+str(e))
constructed.to_csv(_path+"/TEST-"+st+".csv",encoding='utf-8')

print("end")
execution_time_end = current_milli_time()
print("TOTAL TIME: "+str(millisecond_to_minute(execution_time_end,execution_time_start)))


#df_grouped.groupby(pd.cut(df_grouped["SPEEDINKMH"],np.arange(0,200,10)))["LINEARACCX"].mean()