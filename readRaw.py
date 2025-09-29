#%% Loading packages
## rawpy documentation : https://letmaik.github.io/rawpy/api/

import numpy as np
import rawpy
import os
import matplotlib.pyplot as plt
import cv2
import sys
import PIL.Image as Image

#%% Reading input
path = sys.argv[1]
print("Reading image...")
imgPath = path

#%% Class
class rawImageClass():
    def __init__(self, rawDict):
        # Image
        self.image = np.float32(rawDict.raw_image_visible.copy())

        # Blacks and whites
        self.whiteLevels = rawDict.camera_white_level_per_channel
        self.blackLevels = rawDict.black_level_per_channel
        self.rawWhiteLevel = rawDict.white_level

        colorDescription = rawDict.color_desc
        colors = [chr(colorDescription[i]) for i in range(len(colorDescription))]
        bayer = rawDict.raw_pattern.flatten()
        bayer = [colors[bayer[i]] for i in range(len(bayer))]

        # Color
        self.bayerPattern = bayer
        self.ccMatrix = rawDict.rgb_xyz_matrix[0:-1,:] #rawDict.color_matrix[0:3,0:3]
        self.whiteBalance = rawDict.camera_whitebalance[:-1]

        # Tone
        self.toneCurve = rawDict.tone_curve

#%% Reading image
with rawpy.imread(imgPath) as raw:
    rawImage = rawImageClass(raw)

#%% Demosaicing opencv
# Normalize and convert to uint8 (assuming black level approx constant across channels)
rawimg = (rawImage.image - rawImage.blackLevels[0]) / (rawImage.whiteLevels[0] - rawImage.blackLevels[0])
rawimg = np.clip(rawimg, 0, 1)

# Convert to 8-bit
image8bit = (rawimg * 255).astype(np.uint8)

# Apply OpenCV demosaicing
rawRGBdb = cv2.cvtColor(image8bit, cv2.COLOR_BAYER_RGGB2RGB)
rawRGBdb = np.float32(rawRGBdb)/255

# Apply White Balance
wb = rawImage.whiteBalance  # Usually 3 values [R_gain, G_gain, B_gain]
for c in range(3):
    rawRGBdb[..., c] *= wb[c]
rawRGBdb = np.clip(rawRGBdb, 0, 1)

#%%
XYZ_to_cam = rawImage.ccMatrix
sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]], dtype=np.double)
sRGB_to_cam = np.dot(XYZ_to_cam, sRGB_to_XYZ)
norm = np.tile(np.sum(sRGB_to_cam, 1), (3, 1)).transpose()
sRGB_to_cam = sRGB_to_cam / norm
cam_to_sRGB = np.linalg.inv(sRGB_to_cam)
linearRGB = np.einsum('ij,...j', cam_to_sRGB, rawRGBdb)  # performs the matrix-vector product for each pixel


image_sRGB = linearRGB.copy()
i = linearRGB < 0.0031308
j = np.logical_not(i)
image_sRGB[i] = 323 / 25 * image_sRGB[i]
image_sRGB[j] = 211 / 200 * image_sRGB[j] ** (5 / 12) - 11 / 200
image_sRGB = np.clip(image_sRGB, 0, 1)

#%% Converting to 8 bits
imgQuantized = np.uint8(image_sRGB*255)

#%% Converting array to image and displaying
imgPillow = Image.fromarray(imgQuantized)
imgPillow.show()

#%% Saving
imgPillow.save(imgPath[:-4]+".jpg")
print("Save as 8-bit JPEG.")




