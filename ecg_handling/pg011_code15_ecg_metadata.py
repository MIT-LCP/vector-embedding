import pandas as pd
import joblib
import glob



code15_df = pd.read_csv('/mnt/s/Workfolder/CODE15%/exams.csv')

code15_df['tfrecord_path'] = "/mnt/s/Workfolder/vectorembedding/tfrecord/code_15_ecg/id_" + code15_df["exam_id"].astype(str).str.zfill(10) 


joblib.dump(code15_df,'/mnt/s/Workfolder/vectorembedding/df/code15_df')


