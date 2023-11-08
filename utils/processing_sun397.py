from PIL import Image
from pathlib import Path
import os
import tqdm
import numpy as np

path = Path("/home/junyi/data/SUN397")
out_path = Path("/home/junyi/data/SUN397_mod")
subdir = list(path.rglob("sun_*.jpg"))
for p in tqdm.tqdm(subdir):
    image = Image.open(p)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    new_width = min(image.size)
    width, height = image.size
    left = (width - new_width) // 2
    top = (height - new_width) // 2
    right = (width + new_width) // 2
    bottom = (height + new_width) // 2
    image = image.crop((left, top, right, bottom))
    image = image.resize((64, 64))
    path_parts = np.asarray(p.parts)
    path_parts[np.where(path_parts == "SUN397")] = "SUN397_mode"
    os.makedirs(Path(*path_parts[:-1]), exist_ok=True)
    image.save(Path(*path_parts))

