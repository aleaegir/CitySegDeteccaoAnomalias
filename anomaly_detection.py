# anomaly_detection.py
import cv2
import tensorflow as tf

def load_model():
    return tf.keras.models.load_model('behavior_detection_model.h5')

def detect_anomalies(video_path, model):
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(np.expand_dims(processed_frame, axis=0))
        if prediction > 0.5:
            print("Anomaly detected")
        # Additional logic to handle anomalies
    cap.release()

def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    return frame

# Example usage
model = load_model()
detect_anomalies('path_to_video.mp4', model)
