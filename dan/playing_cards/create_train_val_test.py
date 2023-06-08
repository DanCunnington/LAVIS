import pandas as pd 
from sklearn.model_selection import train_test_split
from pathlib import Path
from os.path import join
from PIL import Image

prefix = 'aca_'
data = pd.read_csv(f'{prefix}raw/labels.csv')

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
	Path(join(f'{prefix}images', name)).mkdir(exist_ok=True, parents=True)
	for d in df.values:
		im = d[0]
		# Resize image
		img = Image.open(join(f'{prefix}raw', im))
		new_height = 480
		new_width  = int(new_height * float(img.size[0]) / float(img.size[1]))
		img = img.resize((new_width, new_height), Image.LANCZOS)
		img.save(join(f'{prefix}images', name, im))
	df.to_csv(join(f'{prefix}images', name, 'labels.csv'), index=None)

process_df(train, 'train')
process_df(val, 'val')
process_df(test, 'test')
