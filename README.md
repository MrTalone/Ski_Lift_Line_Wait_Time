# Ski Lift Line Wait Time Tracker

**By:** Talon Finehout

## Overview

This project uses a custom-trained YOLO11n model to detect and track people in ski lift lines across provided videos. The model runs with OpenCV, allowing you to select a region of interest (ROI) for focused tracking. You can visualize tracking activity, unique person counts, and average time spent in the ROI.

## Features

- **Person Detection & Tracking:** Detects and tracks only people (class 0) inside the selected ROI
- **Unique ID Assignment:** Each tracked person receives a unique identifier
- **Real-time Metrics:**
  - Bounding boxes with unique IDs and time spent in ROI
  - Count of unique tracked people over the past second
  - Average time spent per person in the ROI
- **Interactive ROI Selection:** Pause video and select custom regions of interest with mouse clicks
- **Real-time Updates:** OpenCV-based visualization with live tracking data

## Included Files

- `blackcomb_short.mp4` – Short test video from Blackcomb mountain
- `blackcomb_long.mp4` – Long test video from Blackcomb mountain
- `parkcity.mp4` – Test video from Park City mountain
- `final_train_model.py` – Script to train a new YOLO model
- `final_opencv.py` – Script to run video tracking with OpenCV
- `Final_Project/` – Folder containing trained model weights

## Requirements

To use this project, install the following dependencies:

- **Ultralytics YOLO11:** [Official Installation Guide](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)
- **numpy** – Numerical computing library
- **cv2 (OpenCV)** – Computer vision library
- **collections** – For defaultdict functionality

## Getting Started

### Option 1: Use the Pre-trained Model (Recommended)

1. Clone or download this repository
2. Install the required dependencies (see Requirements section)
3. Run `final_opencv.py` to start tracking

For video datasets and a pre-trained model, download from: [Google Drive Link](https://drive.google.com/drive/folders/13tiatBjvUhCxI8e5SOuBkVOcw6-mcMLD?usp=drive_link)

### Option 2: Train a New Model (Optional)

1. Open `final_train_model.py`
2. Run the script to train a new model using your own dataset
3. The trained model will be saved to the `Final_Project/` folder
4. Only retrain if you want to update or modify the current model

## How to Run the Tracker

1. Open and run `final_opencv.py`
2. When prompted, select a video:
   - `1` = blackcomb_short.mp4
   - `2` = parkcity.mp4
   - `3` = blackcomb_long.mp4
   - (Blank/Enter = defaults to blackcomb_short.mp4)

3. The video will begin playing
4. **Press `p`** to pause the video and select a Region of Interest (ROI)
5. **Click 4 points** to define your ROI (recommended order: top-left → top-right → bottom-right → bottom-left)
6. **Press `r`** to reset the ROI while paused
7. **Press `q`** to quit the application

## Technical Details

- **Model:** YOLO11n (nano version for efficient inference)
- **Language:** Python
- **Detection Classes:** People (class 0)
- **Processing:** Real-time video frame processing with OpenCV
