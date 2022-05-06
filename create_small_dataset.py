import shutil
import os
import random

images = os.listdir("../ImageNet/val")
images = [i for i in images if i.endswith(".JPEG")]
images = random.sample(images, 4000)
for i in images:
    shutil.move("../ImageNet/val/"+i, "../ImageNet/val_4000/"+i)