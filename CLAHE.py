#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/jWShMEhMZI4

"""
Histogram equalization: 
Stretches histogram to include all ranges if the original histogram is confined
only to a small region - low contrast images. 
But, this type of stretching may not result in ideal results and gives 
too bright and too dark regions in the image. This can be very bad for images
with large intensity variations. 
CLAHE: COntrast limited adaptive histogram equalization
Regular histogram equalization uses global contrast of the image. This results in
too bright and too dark regions as the histogram stretches and is not confined
to specific region.
Adaptive histogram equalization divides the image into small tiles and within 
each tile the histogram is equalized. Tile size is typically 8x8. 
If theimage contains noise, it gets amplified during this process. Therefore, 
contrast limiting is applied to limit the contrast below a specific limit.
Bilinear interpolation is performed between tile borders. 
Below, let us perform both histogram equalization and CLAHE and compare the results. 
The best way to work with color images is by converting them to luminance space,
e.g. LAB, and enhancing lumincnace channel only and eventually combining all channels. 
    
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from matplotlib import pyplot as plt
images_list = []
SIZE = 150

path = "test/*.*"

img_number = 1
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    img= cv2.imread(file, 0)  #now, we can read each file since we have the full path
    img = cv2.resize(img, (SIZE, SIZE))
    # images_list.append(img)
    lab_img= cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #Converting image to LAB Color so CLAHE can be applied to the luminance channel
    lab_img= cv2.cvtColor(lab_img, cv2.COLOR_BGR2LAB)
    # #Splitting the LAB image to L, A and B channels, respectively
    l, a, b = cv2.split(lab_img)
    # plt.hist(l.flat, bins=100, range=(0,255))
    #plt.plot(cdf_normalized, color = 'b')
    equ = cv2.equalizeHist(l)
   
    hist,bins = np.histogram(equ.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    #plt.hist(equ.flatten(),256,[0,256], color = 'r')
    
    updated_lab_img1 = cv2.merge((equ,a,b))
    #Convert LAB image back to color (RGB)
    hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)
    # #Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(equ)
    # plt.hist(img.flat, bins=100, range=(0,256))
    updated_lab_img2 = cv2.merge((clahe_img,a,b))
    #Combine the CLAHE enhanced L-channel back with A and B channels
    CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
    plt.hist(CLAHE_img.flat, bins, range=(0,256))
    filename = 'test_result'+'1'+str(img_number)+'.jpg'
    cv2.imwrite('test_result/'+filename, CLAHE_img)
    cv2.waitKey(0)
    img_number += 1 
    
 
    
 
    
 
# =============================================================================
# images_list = np.array(images_list)
# 
# img_number = 1
# for input_img in range(images_list.shape[0]):
# # =============================================================================
# #     input_img = images_list[image,:,:]
# #     print (input_img)
# # =============================================================================
#    
#     input_img = cv2.imread(input_img, 1)
#     lab_img= cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab_img)
#     equ = cv2.equalizeHist(l)
#     updated_lab_img1 = cv2.merge((equ,a,b))
#     hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     clahe_img = clahe.apply(l)
#     plt.hist(img.flat, bins=100, range=(0,255))
#     updated_lab_img2 = cv2.merge((clahe_img,a,b))
#     CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
#     filename = 'savedImage'+str(img_number)+'.jpg'
#     #Grey images. For color add another dim.
#     # smoothed_image = img_as_ubyte(gaussian(input_img, sigma=5, mode='constant', cval=0.0))
#     cv2.imwrite(filename, CLAHE_img)
#     cv2.waitKey(0)
#     img_number +=1 
# =============================================================================

# =============================================================================
# 
# img = cv2.imread("clahe_test/monkeypox43.jpg", 1)
# 
# #img = cv2.imread('images/retina.jpg', 1)
# 
# #Converting image to LAB Color so CLAHE can be applied to the luminance channel
# lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# 
# #Splitting the LAB image to L, A and B channels, respectively
# l, a, b = cv2.split(lab_img)
# 
# #plt.hist(l.flat, bins=100, range=(0,255))
# ###########Histogram Equlization#############
# #Apply histogram equalization to the L channel
# # plt.hist(l.flat,bins=200,range=(0,255))
# equ = cv2.equalizeHist(l)
# 
# # =============================================================================
# # plt.imshow(equ, cmap='gray')
# # #Combine the Hist. equalized L-channel back with A and B channels
# updated_lab_img1 = cv2.merge((equ,a,b))
# # 
# # #Convert LAB image back to color (RGB)
# hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)
# # 
# # ###########CLAHE#########################
# # #Apply CLAHE to L channel
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# clahe_img = clahe.apply(l)
# plt.hist(img.flat, bins=100, range=(0,255))
# # 
# # #Combine the CLAHE enhanced L-channel back with A and B channels
# updated_lab_img2 = cv2.merge((clahe_img,a,b))
# # 
# # #Convert LAB image back to color (RGB)
# CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
# # 
# # 
# filename = 'savedImage.jpg'
# cv2.imshow("Original image", img)
# cv2.imshow("Equalized image", hist_eq_img)
# cv2.imshow('CLAHE Image', CLAHE_img)
# cv2.imwrite(filename, CLAHE_img)
# cv2.waitKey(0)
# =============================================================================
# cv2.destroyAllWindows() 
# 
# =============================================================================
