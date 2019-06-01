#%%
import os
import sys
from PIL import Image

files = [x for x in os.listdir('plots')]
files = sorted(files)
files = files[2:]
images = [Image.open("plots/{}".format(x)) for x in files]
widths, heights = zip(*(i.size for i in images))

final_width = max(widths)
final_height = sum(heights)

img = Image.new('RGB', (final_width, final_height))

y_offset = 0
for image in images:
    img.paste(image, (0, y_offset))
    y_offset += image.size[1]

img.save('final.jpg')



#%%
google home show images