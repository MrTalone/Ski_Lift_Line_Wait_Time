# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:36:48 2025

@author: talon

HELLO! this file if for using the yolo 11 model trained in finla_train_model that uses ski lift line videos
There will be an input where you can select a video from the set I have or use your own video as well!
You must pause the video and set an ROI or region of intereset for the model to work.
once you set an ROI and press play you can see the model at work track individuals and tracking an average time people have been in the roi
this will help evaulate how long it takes to get to the lift

"""
# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# %%

from ultralytics import YOLO
import numpy as np
import cv2
from collections import defaultdict

model = YOLO('Final_Project/my_model/weights/best.pt')

#make ROI
photo_coords = []
tracking_roi_set = False
#make mouse click listener to make roi
def mouse_click(event, x, y, flags, param):
    global paused, photo_coords, tracking_roi_set
    if paused and event == cv2.EVENT_LBUTTONDOWN and len(photo_coords) < 4:
        print(f"coords: ({x}, {y})")
        photo_coords.append((x, y))
        if len(photo_coords) == 4:
            tracking_roi_set = True
            print("ROI set. Resuming tracking within selected area.")
            model.trackers.clear()  # Clear tracker state

        
#load video to track on
paused = False
video = input("Select Video - 1: blackcomb_short, 2: parkcity, 3: blackcomb_long: ")

if video == "" or video == "1":
    video = "blackcomb_short.mp4"
elif video == "2":
    video = "parkcity.mp4"
elif video == "3":
    video = "blackcomb_long.mp4"
else:
    print("Invalid selection. Using default video.")
    video = "blackcomb_short.mp4"

cap = cv2.VideoCapture(video)
cv2.namedWindow("TrackingVideo")
cv2.setMouseCallback("TrackingVideo", mouse_click)

ret, old_frame = cap.read()
if not ret:
    print("ERROR")
    

# Export to ONNX format
model.export(format="onnx")

#from https://docs.ultralytics.com/modes/track/#tracker-arguments to use collections defauct dict
track_history = defaultdict(list)
track_time = defaultdict(int)  # how long each person has been present
#me tracking the active ids in roi
active_track_ids = set()

frame_counter = 0
avg_time = 0

#while video is running show video and prompt user over image
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        display_frame = frame.copy()

        #when roi set, track within the roi
        if tracking_roi_set:
            xs = [pt[0] for pt in photo_coords]
            ys = [pt[1] for pt in photo_coords]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            
            
            roi = frame[y_min:y_max, x_min:x_max].copy()

            #make a prediction and track on people in roi
            result = model.track(roi, persist=True, classes=[0])[0] 

            frame_counter += 1

            #if there is a box an id conitnue
            if result.boxes and result.boxes.id is not None:
                #get results boxes and ids
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.cpu().int().tolist()
                confs = result.boxes.conf.cpu()

                #for each box find center within roi
                for box, track_id,confs in zip(boxes, track_ids,confs):
                    x_center, y_center, w, h = box
                    x, y = float(x_center), float(y_center)
                    
                    # Offset coordinates to match original frame
                    x_center += x_min
                    y_center += y_min
                    x += x_min
                    y += y_min
                    
                    #if in roi add to active tracked ids and start timer
                    if x_min <= x <= x_max and y_min <= y <= y_max:

                        #add active track id to list and add 1 for every frame they are detectected in
                        active_track_ids.add(track_id)
                        track_time[track_id] += 1 #"timer"
                        
                        # Calculate top-left corner of the bounding box
                        x1 = int(x_center - w / 2)
                        y1 = int(y_center - h / 2)
                        
                        # Calculate bottom-right corner of the bounding box
                        x2 = int(x_center + w / 2)
                        y2 = int(y_center + h / 2)
    
                        #make box around classified person
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"ID: {track_id}, t: {track_time[track_id]}"
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
                        
                        
                        # Retrieve the movement history (list of past positions) for a given track ID
                        track = track_history[track_id]
                        
                        # Append the current (x, y) position to the track history
                        track.append((x, y))
                        
                        # If the history exceeds 30 positions, remove the oldest one to keep it at a fixed length
                        if len(track) > 30:
                            track.pop(0)
                        
                        #make a line to track id (not needed)
                        #points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        #cv2.polylines(display_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            
            #compile average time every 15 frames or half a second (videos are 30fps)
            if frame_counter >= 15:
                #print(f"Unique people in ROI over last 15 frames: {len(active_track_ids)}")
                    
                
                #if there is a track time then calculate average
                if track_time:
                    total_time = sum(track_time.values())
                    avg_time = total_time / len(track_time)
                    #print(f"Average time per person in ROI: {avg_time:.2f} frames")

                #every second (30 frames) clear out active track ids and reset frame counter
                if frame_counter >= 30:
                    active_track_ids.clear()
                    frame_counter = 0

            # Draw ROI
            pts = np.array(photo_coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # Display unique person count
            cv2.putText(display_frame, f"Unique IDs (30f): {len(active_track_ids)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display average time per person
            cv2.putText(display_frame, f"Avg Time: {avg_time:.1f} seconds", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
                       

        else:
            display_frame = frame.copy()

    else:
        display_frame = frame.copy()
        for point in photo_coords:
            #show ROI point
            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
        if len(photo_coords) == 4:
            #show ROI area
            pts = np.array(photo_coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    height = display_frame.shape[0]
    if paused:
        #show roi suggestion for user when pauses
        cv2.putText(display_frame,
                    "Select ROI: TL, TR, BR, BL (best results)",
                    (20, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        #show controls to user
        cv2.putText(display_frame,
                    "Press 'P' to pause | 'R' to reset | 'Q' to quit",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("TrackingVideo", display_frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("p"):
        paused = not paused
        
    elif key == ord("r") and paused:
        #reset everything!
        photo_coords = []
        tracking_roi_set = False
        active_track_ids.clear()
        track_time.clear()
        track_history.clear()
        frame_counter = 0
        avg_time = 0
        #re load model to reset its tracked objects
        model = YOLO('Final_Project/my_model/weights/best.pt')
        # Export to ONNX format
        model.export(format="onnx")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
####################################################################################################################################
'''
Bellow are some sources i used

Loading model:
https://github.com/ultralytics/ultralytics/issues/10297

tracking objects in a frame:
https://docs.ultralytics.com/modes/track/
    
changing classes to detect:
https://github.com/ultralytics/ultralytics/issues/18525

using results from model.track to build custom views:
https://blog.roboflow.com/monitor-retail-queues/

'''
