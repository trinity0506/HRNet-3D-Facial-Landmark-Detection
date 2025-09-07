import pandas as pd
import os

df = pd.read_csv(config['DATASET.TRAINSET'])
missing = []

for path in df['image_path']:
    if not os.path.exists(os.path.join(config['DATASET.ROOT'], path)):
        missing.append(path)

print(f"缺失文件数: {len(missing)}/{len(df)}")
if missing:
    with open('missing_files.csv', 'w') as f:
        f.write("\n".join(missing))