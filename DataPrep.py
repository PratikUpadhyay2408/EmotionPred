import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

# convert a single WESAD pickle file to a pandas dataframe

wrist_columns=['ACC', 'BVP', 'EDA', 'TEMP']
chest_columns=['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']

all_columns = ['c_acc_x', 'c_acc_y', 'c_acc_z', 'w_acc_x', 'w_acc_y', 'w_acc_z', 'c_eda', 'w_eda',
               'c_temp', 'w_temp', 'ecg', 'emg', 'resp', 'bvp', 'label']

# I have decided to resample everything to 700hz for the following reasons
# a. The chest data and the labels are all at 700 hz,
# b. Up or down sampling labels is not sensible since they represent emotional states.
# c. Downsampling other data risks loss of info.
# d. This might cause the data to become huge


def upsample_array(array, upsample_to):

    upsampled_array = signal.resample(array, upsample_to)
    return upsampled_array


def get_label_col( unpkld_dict ):
    label=unpkld_dict["label"].reshape(unpkld_dict["label"].shape[0], 1)
    return len(label), label


def convert_chest_to_df(sid, unpkld_dict):

    # get max instances in this pkl
    max_dp = len(unpkld_dict['signal']['chest']['ECG'])

    # pack into df
    df = pd.DataFrame()
    df['sid'] = np.full(max_dp, sid)

    df['c_acc_x'] = [ item[0] for item in unpkld_dict["signal"]['chest']['ACC'] ]
    df['c_acc_y'] = [ item[1] for item in unpkld_dict["signal"]['chest']['ACC'] ]
    df['c_acc_z'] = [ item[2] for item in unpkld_dict["signal"]['chest']['ACC'] ]
    df['ecg'] = unpkld_dict["signal"]['chest']['ECG']
    df['emg'] = unpkld_dict["signal"]['chest']['EMG']
    df['c_eda'] = unpkld_dict["signal"]['chest']['EDA']
    df['c_temp'] = unpkld_dict["signal"]['chest']['Temp']
    df['resp'] = unpkld_dict["signal"]['chest']['Resp']

    return df


def convert_wrist_to_df( sid, unpkld_dict, upsample_instances ):

    upsample_700hz = upsample_instances
    df = pd.DataFrame()
    df['sid'] = np.full(upsample_700hz, sid)

    w_acc_new = upsample_array( unpkld_dict["signal"]['wrist']['ACC'], upsample_700hz )

    df['w_acc_x'] = [ item[0] for item in w_acc_new ]
    df['w_acc_y'] = [ item[1] for item in w_acc_new ]
    df['w_acc_z'] = [ item[2] for item in w_acc_new ]

    df['bvp'] = upsample_array( unpkld_dict["signal"]['wrist']['BVP'], upsample_700hz )
    df['w_eda'] = upsample_array( unpkld_dict["signal"]['wrist']['EDA'], upsample_700hz )
    df['w_temp'] = upsample_array( unpkld_dict["signal"]['wrist']['TEMP'], upsample_700hz )

    return df


def combine_wrist_chest_label(wrist_df, chest_df, label):
    # Assuming your first dataframe is df1 and the second dataframe is df2
    # Concatenate the dataframes along the columns axis
    result_df = pd.concat([chest_df[['sid', 'c_acc_x', 'c_acc_y', 'c_acc_z', 'ecg', 'emg', 'c_eda', 'c_temp', 'resp']],
                           wrist_df[['w_acc_x', 'w_acc_y', 'w_acc_z', 'w_eda', 'bvp', 'w_temp']]], axis=1)

    result_df['label'] = label
    # 5,6 and 7 are not valid labels in our scope, thus drop the data.
    result_df.drop(result_df[result_df['label'].isin([5, 6, 7])].index, inplace=True)

    return result_df


if not os.path.isdir("MergedData"):
    os.mkdir("MergedData")

for i in range(1, 18):
    if not os.path.isfile( f"WESAD/S{i}/S{i}.pkl" ):
        pass
    else:
        unpkld_dict = pd.read_pickle( f"WESAD/S{i}/S{i}.pkl")
        upsample_instances, label = get_label_col(unpkld_dict)
        c_df = convert_chest_to_df(i,unpkld_dict)
        w_df = convert_wrist_to_df(i,unpkld_dict,upsample_instances)
        merged_df = combine_wrist_chest_label(w_df, c_df, label)
        merged_df.to_parquet(f'MergedData/S{i}.parquet', index=False)

