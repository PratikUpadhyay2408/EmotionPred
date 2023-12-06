import pandas as pd
import keras
import csv
import pathlib as p

columns = ["c_acc_x", "c_acc_y", "c_acc_z", "ecg", "emg", "c_eda", "c_temp", "resp",
           "w_acc_x", "w_acc_y", "w_acc_z", "w_eda", "bvp", "w_temp"]

emo_states = ["Transient", "Usual", "Stressed", "Amused", "Meditative" ]


def write_test_df(data_dir, num_instances=-1):
    # Create a list of delayed objects for reading Parquet files
    full_df = pd.DataFrame()
    for parquet_file in data_dir.glob('*.parquet'):
        print(f"reading {parquet_file}")
        temp_df = pd.read_parquet(parquet_file)
        if num_instances > 0:
            # stratified sampling of n examples from each subject.
            temp_df = temp_df.groupby("label").sample(n=num_instances, random_state=4213)

        full_df = pd.concat([full_df, temp_df])

    full_df = full_df.sample(frac=1)
    test_df = full_df.drop(["sid", "label"], axis=1)  # Specify axis=1 to drop columns
    sid = full_df["sid"].tolist()
    sub_info_df = pd.read_csv("D:\AI_Portfolio\EmotionPred\MergedData\Subject_Info.csv")

    return test_df, sub_info_df, sid


if __name__ == "__main__":

    model = keras.models.load_model("D:\AI_Portfolio\EmotionPred\Results\FinalModel.h5")
    data_dir = p.Path("D:\AI_Portfolio\EmotionPred\MergedData")

    test_df, sub_info_df, sid = write_test_df(data_dir, num_instances=2500)

    current_row = {}
    subject_row = {}
    pred_df=pd.DataFrame()
    for i in range(0, len(test_df)):
        # setup predict csv file
        current_row["id"] = i
        current_row = test_df.iloc[[i]].to_dict(orient='records')[0]
        # setup subject info file.
        print(sid[i])
        subject_row = sub_info_df.loc[ sub_info_df['sid']==sid[i] ].to_dict(orient='records')[0]
        # do the prediction
        test = model.predict(test_df.iloc[[i]])
        current_row["Emotional_State"] = test.argmax()

        with open("D:\AI_Portfolio\EmotionPred\predict.csv", 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=current_row.keys())
            # Write the header
            writer.writeheader()
            # Write the data
            writer.writerow(current_row)

        with open("D:\AI_Portfolio\EmotionPred\subject.csv", 'w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            # Write key-value pairs as separate rows
            for key, value in subject_row.items():
                writer.writerow([key, value])