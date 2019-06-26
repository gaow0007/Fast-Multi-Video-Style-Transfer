import numpy
import sys, os
import shutil
cnt = 0
for idx, name in enumerate(sorted(os.listdir('./tiger/'))):
    # if idx % 8 != 0:
    #     continue

    orifile = os.path.join('./tiger/', name)
    nxtfile = os.path.join('./testimages/', str(cnt//2) + '_' + str(cnt%2) + '.png')
    shutil.copy(orifile, nxtfile)
    cnt = cnt + 1

