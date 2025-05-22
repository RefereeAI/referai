import os
import numpy as np
import onnxruntime as ort
import cv2
from typing import Any
from fastapi import Request

# Device configuration (CPU, since torch.cuda is not used)
device = "cpu"

# Manual normalization (replaces torchvision transforms)
def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame / 255.0  # Scale to [0, 1]
    mean = np.array([0.45, 0.45, 0.45]).reshape((3, 1, 1))
    std = np.array([0.225, 0.225, 0.225]).reshape((3, 1, 1))
    return (frame - mean) / std

def load_models(model_dir: str):
    """Load all .onnx models from a folder into a dictionary."""
    models = {}
    for file in os.listdir(model_dir):
        if file.endswith(".onnx"):
            path = os.path.join(model_dir, file)
            # Extract model type from filename (e.g., 'slowfast' from 'modelo_xgb_slowfast.onnx')
            model_type = file.replace("modelo_xgb_", "").replace(".onnx", "")
            if model_type in ["mvit", "x3d", "slowfast"]:
                models[model_type] = ort.InferenceSession(path)
            else:
                print(f"Warning: Unknown model type in file {file}, skipping.")

    return models

def load_severity_models(model_dir: str):
    """Load severity ONNX models into a dictionary by type."""
    models = {}
    for file in os.listdir(model_dir):
        if file.endswith(".onnx") and file.startswith("modelo_xgboost_severity_"):
            path = os.path.join(model_dir, file)
            model_type = file.replace("modelo_xgboost_severity_", "").replace(".onnx", "")
            if model_type in ["mvit", "x3d", "slowfast"]:
                models[model_type] = ort.InferenceSession(path)
            else:
                print(f"Warning: Unknown severity model type in file {file}, skipping.")

    return models

def load_x3d_model():
    return ort.InferenceSession(os.path.join(os.path.dirname(__file__), "x3d.onnx"))

def load_slowfast_model():
    return ort.InferenceSession(os.path.join(os.path.dirname(__file__), "slowfast.onnx"))

def load_mvit_model():
    return ort.InferenceSession(os.path.join(os.path.dirname(__file__), "mvit.onnx"))

def preprocess_video_for_mvit(video_path, start_frame=60, end_frame=80, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise RuntimeError("The video contains no frames")
    
    # Adjust the range limits
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.transpose(2, 0, 1)  # (C, H, W)
        frame = normalize_frame(frame)
        frames.append(frame)
    
    cap.release()
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))  # Pad if there are less than 16 frames
    
    frames_array = np.stack(frames).transpose(1, 0, 2, 3)  # (C, T, H, W)
    return np.expand_dims(frames_array, axis=0).astype(np.float32)  # (1, C, T, H, W)

def preprocess_video_for_x3d(video_path, frame_size=(224, 224), num_frames=32):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {video_path}")
        
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose(2, 0, 1)  # (C, H, W)
        frame = normalize_frame(frame)
        frames.append(frame)
        
        if len(frames) >= num_frames:
            break

    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not extract frames from video: {video_path}")
    
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    frames_array = np.stack(frames).transpose(1, 0, 2, 3)  # (C, T, H, W)
    return np.expand_dims(frames_array, axis=0).astype(np.float32)  # (1, C, T, H, W)

def preprocess_video_for_slowfast(video_path, frame_size=(224, 224), slow_frames=8, fast_frames=32):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {video_path}")
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose(2, 0, 1)  # (C, H, W)
        frame = normalize_frame(frame)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not extract frames from video: {video_path}")
    
    # Ensure enough frames for both streams
    while len(frames) < fast_frames:
        frames.append(frames[-1])
    
    frames = np.stack(frames).transpose(1, 0, 2, 3)  # (C, T, H, W)
    
    # Prepare inputs for SlowFast
    slow_idx = np.linspace(0, len(frames[0]) - 1, slow_frames).astype(int)  # Subsample for the slow stream
    fast_idx = np.arange(fast_frames)  # All frames for the fast stream
    
    slow_frames_data = frames[:, slow_idx, :, :]  # (C, T_slow, H, W)
    fast_frames_data = frames[:, fast_idx, :, :]  # (C, T_fast, H, W)
    
    return (np.expand_dims(slow_frames_data, axis=0).astype(np.float32), 
            np.expand_dims(fast_frames_data, axis=0).astype(np.float32))  # [(1, C, T_slow, H, W), (1, C, T_fast, H, W)]

def extract_features_slowfast(video_path, model):
    slow_frames, fast_frames = preprocess_video_for_slowfast(video_path)  # Extract frames for both streams
    
    # Run ONNX model
    input_names = [inp.name for inp in model.get_inputs()]
    outputs = model.run(None, {input_names[0]: slow_frames, input_names[1]: fast_frames})
    
    return np.array(outputs[0]).squeeze()

def extract_features_mvit(video_path, start_frame, end_frame, model):
    frames = preprocess_video_for_mvit(video_path, start_frame, end_frame)
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: frames})
    return np.array(outputs[0]).squeeze()

def extract_features_x3d(video_path, model):
    frames = preprocess_video_for_x3d(video_path)
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: frames})
    return np.array(outputs[0]).squeeze()

def predict(video_paths: list,request: Request) -> dict:
    
    foul_models = request.app.state.foul_models
    severity_models = request.app.state.severity_models
    
    # Load feature extraction models
    mvit = load_mvit_model()
    x3d = load_x3d_model()
    slowfast = load_slowfast_model()

    # Initialize lists to store action clips
    action_clips_mvit = []
    action_clips_x3d = []
    action_clips_slowfast = []

    # Process each video and extract features
    for video_path in video_paths:   
        try:
            features_mvit = extract_features_mvit(video_path, 60, 80, mvit)
            features_x3d = extract_features_x3d(video_path, x3d)
            features_slowfast = extract_features_slowfast(video_path, slowfast)

            action_clips_mvit.append(features_mvit)
            action_clips_x3d.append(features_x3d)
            action_clips_slowfast.append(features_slowfast)
        except Exception as e:
            print(f"Error while processing video {video_path}: {e}")

    
    action_features = {
    "mvit": np.mean(action_clips_mvit, axis=0),
    "x3d": np.mean(action_clips_x3d, axis=0),
    "slowfast": np.mean(action_clips_slowfast, axis=0)
    }

    print("Action features shape: ", len(action_features))

    # Verify that all features have been calculated for each 3 models
    if len(action_features) != 3:
        raise RuntimeError("All 3 models should have produced features.")
    
    foul_preds = []
    foul_model_results = []
    for model_type in ["mvit", "x3d", "slowfast"]:
        model = foul_models.get(model_type)
        if model is None:
            print(f"No foul model found for {model_type}")
            continue
        feature = action_features[model_type]
        input_name = model.get_inputs()[0].name
        output = model.run(None, {input_name: feature[np.newaxis, :].astype(np.float32)})
        probabilities = output[1]  # We assume the ONNX model returns softmax
        prediction = int(np.argmax(probabilities))
        print(prediction)
        print(f"{model_type} foul probabilities: {probabilities}")
        foul_preds.append(prediction)
        foul_model_results.append({
            "model": f"Foul Model {model_type}",
            "prediction": prediction
        })

    # Calculate the average of the predictions
    total_foul_preds = len(foul_preds)
    foul_pct = (foul_preds.count(1)/total_foul_preds) * 100
    no_foul_pct = (foul_preds.count(0)/total_foul_preds) * 100

    # Calculate the severity predictions only if a foul is detected
    if foul_pct > no_foul_pct:
        severity_preds = []
        severity_model_results = []

        for model_type in ["mvit", "x3d", "slowfast"]:
            model = severity_models.get(model_type)
            if model is None:
                print(f"No severity model found for {model_type}")
                continue
            feature = action_features[model_type]
            input_name = model.get_inputs()[0].name
            output = model.run(None, {input_name: feature[np.newaxis, :].astype(np.float32)})
            probabilities = output[1]  # We assume the ONNX model returns softmax
            prediction = int(np.argmax(probabilities))
            print(f"{model_type} severity probabilities: {probabilities}")
            severity_preds.append(prediction)
            severity_model_results.append({
                "model": f"Severity Model {model_type}",
                "prediction": prediction
            })
        # Calculate the average of the severity predictions
        total_severity_preds = len(severity_preds)
        red_card_pct = (severity_preds.count(1)/total_severity_preds) * 100
        yellow_card_pct = (severity_preds.count(2)/total_severity_preds) * 100
        no_card_pct = (severity_preds.count(0)/total_severity_preds) * 100
    else:
        red_card_pct = 0
        yellow_card_pct = 0
        no_card_pct = 100
        severity_model_results = []
    # Change the prediction to int
    foul_model_results = [
        {
            "model": result["model"],
            "prediction": int(result["prediction"])
        }
        for result in foul_model_results
    ]

    # Change the severity predictions to int
    severity_model_results = [
        {
            "model": result["model"],
            "prediction": int(result["prediction"])
        }
        for result in severity_model_results
    ]

    return {
        "is_foul": bool(foul_pct > no_foul_pct),
        "foul_confidence": float(foul_pct),
        "no_foul_confidence": float(no_foul_pct),
        "foul_model_results": foul_model_results,
        "severity": {
            "no_card": float(no_card_pct),
            "red_card": float(red_card_pct),
            "yellow_card": float(yellow_card_pct),
        },
        "severity_model_results": severity_model_results,
    }