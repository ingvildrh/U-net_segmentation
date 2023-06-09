import os
import cv2
import numpy as np
import mediapipe as mp


mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp.selfie_segmentation.SelfieSegmentation(model_selection=1)


img = cv2.imread('C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/U-NET/UNET_trying/augmented_data/new_data_111_1111/test/image/IMG_0882_0.png')

RGB = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)

results = selfie_segmentation.process(RGB)
mask = results.segmentation_mask
cv2.imshow("mask", mask)
cv2.imshow("RGB", RGB)
cv2.waitKey(0)











# import cv2
# import numpy as np

# ''' Parameters '''
# BLUR = 21
# CANNY_THRESH_1 = 10
# CANNY_THRESH_2 = 160
# MASK_DILATE_ITER = 12
# MASK_ERODE_ITER = 10
# MASK_COLOR =  (0.0,1.0,0.0) # In BGR format

# ''' Process '''

# ''' Read image '''
# img = cv2.imread('C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/U-NET/UNET_trying/augmented_data/new_data_111_1111/test/image/IMG_0882_0.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ''' Edge detection '''
# edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
# cv2.imshow("edges", edges)
# cv2.waitKey(0)
# edges = cv2.dilate(edges, None)
# edges = cv2.erode(edges, None)

# ''' Find contours in edges, sort by area '''
# contour_info = []
# contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# for c in contours:
#     contour_info.append((
#         c,
#         cv2.isContourConvex(c),
#         cv2.contourArea(c),
#     ))

# contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
# max_contour = contour_info[0]

# ''' Create empty mask, draw filled polygon on it corresponding to largest contour '''
# # Mask is black, polygon is white
# mask = np.zeros(edges.shape)
# cv2.fillConvexPoly(mask, max_contour[0], (255))

# ''' Smooth mask, then blur it '''
# mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
# mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
# mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
# mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

# ''' Blend masked img into MASK_COLOR background '''
# mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices,
# img         = img.astype('float32') / 255.0                 #  for easy blending

# masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
# masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit

# cv2.imshow('img', masked)                                   # Display
# cv2.waitKey()