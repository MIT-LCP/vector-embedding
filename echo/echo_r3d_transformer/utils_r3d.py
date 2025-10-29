import argparse
import torch
import numpy as np
import os
import cv2
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
import gc
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

import pydicom



class r3dmodel(nn.Module):
  def __init__(self, model1, regression = False):
    super(r3dmodel, self).__init__()
    self.regression = regression
    self.preloaded_model = model1
    self.new_layer1 = nn.Linear(400,1)
    if self.regression == False:
        self.new_layer2 = nn.Sigmoid()
  def forward(self, x):
    x = self.preloaded_model(x)
    x = self.new_layer1(x)
    if self.regression == False:
        x = self.new_layer2(x)
    return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class DeepFeatures(torch.nn.Module):
    '''
    This class extracts, reads, and writes data embeddings using a pretrained deep neural network. Meant to work with
    When using with a 3 channel image input and a pretrained model from torchvision.models please use the
    following pre-processing pipeline:

    transforms.Compose([transforms.Resize(imsize),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) ## As per torchvision docs

    Args:
        model (nn.Module): A Pytorch model that returns an (B,1) embedding for a length B batched input
        imgs_folder (str): The folder path where the input data elements should be written to
    '''

    def __init__(self, model):
        super(DeepFeatures, self).__init__()
        self.model = model
        self.model.eval()

    def generate_embeddings(self, x):
        return(self.model(x))



def avi_to_tensor(video_file, max_frames=100):
    """Convert an avi file to a tensor object.
    Args:
        video_file (str): path to the avi file to convert
        max_frames (int): cap for number of frames to convert
    Returns:
        PyTorch tensor representation of the avi video
    """
    # initialize a list to store video frames
    frames = []
    # open the video file
    cap = cv2.VideoCapture(video_file)
    # read frames from the video file
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # transpose the frame shape from (height, width, channel) to 
        # (channel, height, width)
        frame_t = np.transpose(frame_rgb, (2, 0, 1))
        frames.append(frame_t)
        # stop reading frames if maximum number of frames is reached
        if max_frames is not None and len(frames) >= max_frames:
            break
    # release the video file
    cap.release()
    # stack the frames to create a 4D numpy array
    video_array = np.stack(frames, axis=0)
    # convert the numpy array to a PyTorch tensor
    video_tensor = torch.from_numpy(video_array).float()
    return video_tensor



def dicom_to_tensor(dicom_file, max_frames=100):
    """Convert a DICOM file (multi-frame) to a PyTorch tensor.
    Args:
        dicom_file (str): Path to the DICOM file to convert.
        max_frames (int): Maximum number of frames to process.
    Returns:
        PyTorch tensor representation of the DICOM video.
    """
    # Load the DICOM file
    dicom_data = pydicom.dcmread(dicom_file)
    
    # Ensure the file contains pixel data
    if not hasattr(dicom_data, 'PixelData'):
        raise ValueError("DICOM file does not contain pixel data")
    
    # Extract pixel array
    pixel_array = dicom_data.pixel_array  # Shape: (frames, height, width) or (height, width)
    
    # If single-frame, add a frame dimension
    if pixel_array.ndim == 2:
        pixel_array = np.expand_dims(pixel_array, axis=0)
    
    # Convert grayscale to 3-channel RGB if necessary
    frames = []
    for i in range(min(len(pixel_array), max_frames)):
        frame = pixel_array[i]
        if len(frame.shape) == 2:  # Grayscale
            frame_rgb = np.stack([frame] * 3, axis=-1)  # Convert to (H, W, 3)
        else:
            frame_rgb = frame  # Already RGB
        
        # Transpose to (C, H, W)
        frame_t = np.transpose(frame_rgb, (2, 0, 1))
        frames.append(frame_t)
    
    # Stack frames into a 4D numpy array
    video_array = np.stack(frames, axis=0)  # Shape: (T, C, H, W)
    
    # Convert to PyTorch tensor
    video_tensor = torch.from_numpy(video_array).float()
    
    return video_tensor