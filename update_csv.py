import pandas as pd

# 打印CSV文件的实际列名
df = pd.read_csv('./data/wflw/face_landmarks_wflw_train.csv')
print("实际列名:", df.columns.tolist())