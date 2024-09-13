# data_preparation.py
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_videos(video_paths):
    frames = []
    labels = []
    
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            # Assume labels are provided separately
            labels.append(get_label_from_path(path))
        cap.release()
    
    return np.array(frames), np.array(labels)

def get_label_from_path(path):
    # Dummy label function; replace with actual label extraction logic
    return 0

# Example usage
video_paths = ['path_to_video1.mp4', 'path_to_video2.mp4']
frames, labels = load_videos(video_paths)
