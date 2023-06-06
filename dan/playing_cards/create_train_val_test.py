import pandas as pd 
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
from os.path import join

data = pd.read_csv('raw/labels.csv')

X = list(data['img'].values)
y = list(data['label'].values)

X_tv, X_test, y_tv, y_test = train_test_split(
	X, y, test_size=0.2, random_state=0, stratify=y)


test = pd.DataFrame({'img': X_test, 'label': y_test})

X_train, X_val, y_train, y_val = train_test_split(
	X_tv, y_tv, test_size=0.2, random_state=0, stratify=y_tv)

train = pd.DataFrame({'img': X_train, 'label': y_train})
val = pd.DataFrame({'img': X_val, 'label': y_val})



def process_df(df, name):
	Path(join('images', name)).mkdir(exist_ok=True)
	for d in df.values:
		im = d[0]
		shutil.copyfile(join('raw', im), join('images', name, im))
	df.to_csv(join('images', name, 'labels.csv'), index=None)

process_df(train, 'train')
process_df(val, 'val')
process_df(test, 'test')
