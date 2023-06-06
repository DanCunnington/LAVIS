import os, sys
from PIL import Image
from os.path import join


for im in os.listdir('raw'):
	if '.jpg' in im:
		img = Image.open(join('raw', im))
		new_height = 480
		new_width  = int(new_height * float(img.size[0]) / float(img.size[1]))
		img = img.resize((new_width, new_height), Image.LANCZOS)
		img.save(join('resized', im))