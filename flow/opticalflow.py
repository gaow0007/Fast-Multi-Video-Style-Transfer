import cv2
import numpy as np


# input is numpy image array
def opticalflow(img1, img2):
    prvs = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    next = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros((prvs.shape[0], prvs.shape[1], 3))
    hsv[..., 1] = 255

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    print(hsv.shape, ang.shape)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    mask = np.where(x > 0.5, 0, 1)
    gray[mask]

    return rgb, gray[mask]

if __name__ == '__main__':
    flow = opticalflow(cv2.imread('testimages/0_0.jpg'), cv2.imread('testimages/0_1.jpg'))
    print(flow.shape)

    # cv2.save(rgb, 'rgb.jpg')
