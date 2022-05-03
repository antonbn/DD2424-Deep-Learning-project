import shutil
import os
import random

images = os.listdir("../ImageNet/train")
images = [i for i in images if i.endswith(".JPEG")]
images = random.sample(images, 40000)
for i in images:
    shutil.move("../ImageNet/train/"+i, "../ImageNet/train_40000/"+i)