import shutil
import os
import random

images = os.listdir("../ImageNet")
images = [i for i in images if i.endswith(".JPEG")]
images = random.choices(images, k=40000)
for i in images:
    print("../ImageNet/train/"+i)
    print("../ImageNet/train_40000/"+i)
    print()
    #shutil.move("../ImageNet/train/"+i, "../ImageNet/train_40000/"+i)