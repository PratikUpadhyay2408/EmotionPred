import numpy as np
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import pathlib as p


def write_emo_state_df(data_dir, label_val=0, num_instances=-1):
    # Create a list of delayed objects for reading Parquet files
    full_df = pd.DataFrame()
    for parquet_file in data_dir.glob('*.parquet'):
        print(f"reading {parquet_file}")
        temp_df = pd.read_parquet(parquet_file)
        temp_df = temp_df[temp_df['label'] == label_val]
        if num_instances > 0:
            temp_df = temp_df.sample(n=num_instances)
        full_df = pd.concat([full_df, temp_df])

    index_arr = np.linspace(start=1, stop=len(full_df), num=len(full_df))
    full_df.insert( 0, 'sample_id', index_arr )
    full_df.drop( ['sid'], axis=1 )


    return full_df

if __name__ == "__main__":

        # Replace 'wise' and '100' with your actual username and password
        username = 'postgres'
        password = 'panda'
        host = 'localhost'
        port = '5432'
        database = 'Sensor_Input_WESAD'
        # Create the SQLAlchemy engine with the modified connection string
        engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')
        if not database_exists(engine.url):
            create_database(engine.url)

        print(engine.url)
        data_dir = p.Path("MergedData")
        stress_df = write_emo_state_df(data_dir, 2, num_instances=2500)
        amused_df = write_emo_state_df(data_dir, 3, num_instances=2500)
        meditative_df = write_emo_state_df(data_dir, 4, num_instances=2500)
        # Write the DataFrame to the PostgreSQL database, creating a new table 'your_table' if it doesn't exist
        stress_df.to_sql('Sensor_Input', engine, if_exists='replace', index=False)
        amused_df.to_sql('Sensor_Input_amused', engine, if_exists='replace', index=False)
        meditative_df.to_sql('Sensor_Input_meditative', engine, if_exists='replace', index=False)

        print("DataFrames successfully written to PostgreSQL.")
