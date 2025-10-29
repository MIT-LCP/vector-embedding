import torchvision
from pathlib import Path
import numpy as np
import cv2
import pydicom as dicom
import torch

COARSE_VIEWS=['A2C',
 'A3C',
 'A4C',
 'A5C',
 'Apical_Doppler',
 'Doppler_Parasternal_Long',
 'Doppler_Parasternal_Short',
 'Parasternal_Long',
 'Parasternal_Short',
 'SSN',
 'Subcostal']

def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)
    return img


def mask_outside_ultrasound(original_pixels: np.array) -> np.array:
    """
    Masks all pixels outside the ultrasound region in a video.

    Args:
    vid (np.ndarray): A numpy array representing the video frames. FxHxWxC

    Returns:
    np.ndarray: A numpy array with pixels outside the ultrasound region masked.
    """
    try:
        testarray=np.copy(original_pixels)
        vid=np.copy(original_pixels)
        ##################### CREATE MASK #####################
        # Sum all the frames
        frame_sum = testarray[0].astype(np.float32)  # Start off the frameSum with the first frame
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_YUV2RGB)
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_RGB2GRAY)
        frame_sum = np.where(frame_sum > 0, 1, 0) # make all non-zero values 1
        frames = testarray.shape[0]
        for i in range(frames): # Go through every frame
            frame = testarray[i, :, :, :].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.where(frame>0,1,0) # make all non-zero values 1
            frame_sum = np.add(frame_sum,frame)

        # Erode to get rid of the EKG tracing
        kernel = np.ones((3,3), np.uint8)
        frame_sum = cv2.erode(np.uint8(frame_sum), kernel, iterations=10)

        # Make binary
        frame_sum = np.where(frame_sum > 0, 1, 0)

        # Make the difference frame fr difference between 1st and last frame
        # This gets rid of static elements
        frame0 = testarray[0].astype(np.uint8)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_YUV2RGB)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        frame_last = testarray[testarray.shape[0] - 1].astype(np.uint8)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_YUV2RGB)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_RGB2GRAY)
        frame_diff = abs(np.subtract(frame0, frame_last))
        frame_diff = np.where(frame_diff > 0, 1, 0)

        # Ensure the upper left hand corner 20x20 box all 0s.
        # There is a weird dot that appears here some frames on Stanford echoes
        frame_diff[0:20, 0:20] = np.zeros([20, 20])

        # Take the overlap of the sum frame and the difference frame
        frame_overlap = np.add(frame_sum,frame_diff)
        frame_overlap = np.where(frame_overlap > 1, 1, 0)

        # Dilate
        kernel = np.ones((3,3), np.uint8)
        frame_overlap = cv2.dilate(np.uint8(frame_overlap), kernel, iterations=10).astype(np.uint8)

        # Fill everything that's outside the mask sector with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.where(frame_overlap!=100,255,0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(frame_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours[0] has shape (445, 1, 2). 445 coordinates. each coord is 1 row, 2 numbers
        # Find the convex hull
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(frame_overlap, [hull], -1, (255, 0, 0), 3)
        frame_overlap = np.where(frame_overlap > 0, 1, 0).astype(np.uint8) #make all non-0 values 1
        # Fill everything that's outside hull with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.array(np.where(frame_overlap != 100, 255, 0),dtype=bool)
        ################## Create your .avi file and apply mask ##################
        # Store the dimension values

        # Apply the mask to every frame and channel (changing in place)
        for i in range(len(vid)):
            frame = vid[i, :, :, :].astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            frame = cv2.bitwise_and(frame, frame, mask = frame_overlap.astype(np.uint8))
            vid[i,:,:,:]=frame
        return vid
    except Exception as e:
        print("Error masking returned as is.")
        return vid


def process_dicom_echoprime(dicom_path, n_frames=32, frame_stride=2, video_size=224):
    try:
        dcm = dicom.dcmread(dicom_path, defer_size="1KB")
        pixels = dcm.pixel_array

        
        # 3 channel
        if pixels.ndim == 3:
            pixels = np.repeat(pixels[..., None], 3, axis=3)
        
        # masking
        pixels = mask_outside_ultrasound(pixels)
        
        # EchoPrime statistics
        mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)
        
        # resize
        x = np.zeros((len(pixels), video_size, video_size, 3))
        for i in range(len(x)):
            x[i] = crop_and_scale(pixels[i], res=(video_size, video_size))
        
        # [T, H, W, C] -> [C, T, H, W]
        x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
        
        # Z-score standardization
        x.sub_(mean).div_(std)

        # frame cnt adjustment
        if x.shape[1] < n_frames:
            padding = torch.zeros((3, n_frames - x.shape[1], video_size, video_size), dtype=torch.float)
            x = torch.cat((x, padding), dim=1)
            
        start = 0
        stack_of_video = x[:, start:(start + n_frames):frame_stride, :, :]
        
        return stack_of_video.unsqueeze(0)
        
    except Exception as e:
        print(f"DICOM handling error {dicom_path}: {e}")
        raise  