
# import spiga
import numpy as np
import os
import sys
import time
import json
import pickle 
import cv2
import subprocess
import mediapipe as mp
from collections import deque
import argparse
from scipy.interpolate import interp1d
############################### Related to IOU tracker: ###############################
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def filter_boxes(all_frames_boxes, K):
    filtered_boxes = []

    # we first remove all the empty frames
    non_empty_boxes = [frame_boxes for frame_boxes in all_frames_boxes if frame_boxes != []]
    frames_with_empty_boxes = [i for i, frame_boxes in enumerate(all_frames_boxes) if frame_boxes == []] # we will add these back in later
    flag = {"has_missing": False, 
            "has_multiple": False, 
            "no_first_frame": False, 
            "no_last_frame": False, 
            "multiple_boxes_first_frame": False}

    for i, frame_boxes in enumerate(non_empty_boxes):
        if frame_boxes == []:
            flag["has_missing"] = True
            filtered_boxes.append([])
            continue
        # if we have multiple boxes in the first frame, we will find the first frame with only a single box, and take the box with the highest IOU with that
        if i == 0 and len(frame_boxes) > 1:
            flag["multiple_boxes_first_frame"] = True
            flag["has_multiple"] = True
            # look ahead and find the first three frame with only a single box
            non_empty_count = 0
            non_empty_index = []
            for j in range(i+1, min(i+K+1, len(non_empty_boxes))):
                if len(non_empty_boxes[j]) == 1:
                    non_empty_count += 1
                    non_empty_index.append(j)
                if non_empty_count == 3:
                    break
            # we compute  
            IOUs = np.zeros((len(frame_boxes)))
            for j in range(len(non_empty_index)):
                IOUs += np.array([calculate_iou(frame_box[1], non_empty_boxes[non_empty_index[j]][0][1]) for frame_box in frame_boxes])
            best_box = frame_boxes[np.argmax(IOUs)]
            filtered_boxes.append(best_box[1])
            non_empty_count += 1 
        elif len(frame_boxes) == 1:
            filtered_boxes.append(frame_boxes[0][1])
        else:
            flag["has_multiple"] = True
            IOUs = np.zeros((len(frame_boxes)))
            # print("here")
            for j in range(max(0, i-K), i):
                # print(frame_boxes, filtered_boxes[j][1])
                # compute the IOU between all boxes on the current frame and the previous K frames
                IOUs += np.array([calculate_iou(frame_box[1], filtered_boxes[j]) for frame_box in frame_boxes])
            IOUs /= K
            # if the best IOU is greater than 0.4, we take that box
            if np.max(IOUs) > 0.4:
                best_box = frame_boxes[np.argmax(IOUs)]
                filtered_boxes.append(best_box[1])
            else:
                # otherwise we just take the bb from previous frame
                filtered_boxes.append(filtered_boxes[-1])
    # add back the empty frames
    for i in frames_with_empty_boxes:
        flag["has_missing"] = True
        filtered_boxes.insert(i, [])
    # see if the first and last frame 
    if filtered_boxes[0] == []:
        flag["no_first_frame"] = True
        # set this frame to be the first frame with a box
        for i in range(1, len(filtered_boxes)):
            if filtered_boxes[i] != []:
                filtered_boxes[0] = filtered_boxes[i]
                break
    if filtered_boxes[-1] == []:
        flag["no_last_frame"] = True
        # set this frame to be the last frame with a box
        for i in range(len(filtered_boxes)-2, -1, -1):
            if filtered_boxes[i] != []:
                filtered_boxes[-1] = filtered_boxes[i]
                break
    # use linear interpolation to fill in the empty frames
    for i, frame_boxes in enumerate(filtered_boxes):
        if frame_boxes == []:
            left_idx = i
            while left_idx > 0 and filtered_boxes[left_idx] == []:
                left_idx -= 1
            right_idx = i
            while right_idx < len(filtered_boxes) - 1 and filtered_boxes[right_idx] == []:
                right_idx += 1
            left_box = filtered_boxes[left_idx]
            right_box = filtered_boxes[right_idx]
            if left_box == [] or right_box == []:
                continue
            f = interp1d([left_idx, right_idx], [left_box, right_box], axis=0)
            interpolated_box = f(i)
            filtered_boxes[i] = interpolated_box
    return filtered_boxes, flag

def interpolate_boxes(filtered_boxes):
    interpolated_boxes = []
    last_valid_frame = -1
    for i, boxes in enumerate(filtered_boxes):
        if boxes:
            if last_valid_frame != -1 and last_valid_frame != i - 1:
                # Interpolate between last valid frame and current frame
                start_box = interpolated_boxes[last_valid_frame][0]
                end_box = boxes[0]
                for j in range(last_valid_frame + 1, i):
                    t = (j - last_valid_frame) / (i - last_valid_frame)
                    interpolated_box = [
                        start_box[k] + t * (end_box[k] - start_box[k])
                        for k in range(4)
                    ]
                    interpolated_boxes.append([interpolated_box])
            interpolated_boxes.append(boxes)
            last_valid_frame = i
        else:
            interpolated_boxes.append([])
    
    return interpolated_boxes

def process_bounding_boxes(metadata, K):
    bound_box_list = metadata['raw_bbox_frames']
    
    # Filter boxes
    filtered_boxes, flag = filter_boxes(bound_box_list, K)
    
    # Interpolate empty frames
    # interpolated_boxes = interpolate_boxes(filtered_boxes)
    interpolated_boxes = filtered_boxes
    return interpolated_boxes, flag

############################### Related to IOU tracker: ###############################

# Function to generate random colors
def generate_random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def process_frame(image, face_detection):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    faces = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            faces.append((x, y, width, height))
    return faces

def track_faces(all_frames_faces, K):
    tracked_faces = []
    for i, current_faces in enumerate(all_frames_faces):
        frame_tracked_faces = []
        for face in current_faces:
            avg_iou = 0
            count = 0
            for j in range(max(0, i-K), min(len(all_frames_faces), i+K+1)):
                if i == j:
                    continue
                best_iou = max([calculate_iou(face, other_face) for other_face in all_frames_faces[j]], default=0)
                avg_iou += best_iou
                count += 1
            if count > 0:
                avg_iou /= count
                if avg_iou > 0.5:  # You can adjust this threshold
                    frame_tracked_faces.append(face)
        tracked_faces.append(frame_tracked_faces)
    return tracked_faces


############### generate video_split.pkl file ###############

if __name__ == '__main__':

    DATASET_ROOT = "/data/celebv-text/"
    VIDEO_ROOT = os.path.join(DATASET_ROOT, "videos")
    OUTPUT_ROOT = os.path.join(DATASET_ROOT, "boundbox_mediapipe")

    file_list_path_template = '/data/celebv-text/splitting/video_split_{}.pkl'
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Process some file paths.")
    parser.add_argument('--shard_id', type=str, required=True)
    parser.add_argument('--output_raw_video', type=bool, default=False)
    args = parser.parse_args()

    shard_id = args.shard_id
    
    file_list_path = file_list_path_template.format(shard_id)
    do_output_raw_video = args.output_raw_video
    
    # Read filenames from the pickle file
    with open(file_list_path, 'rb') as file:
        filenames = pickle.load(file)
    test_videos = [filename[0] for filename in filenames]
    # prep a run log
    run_log = []
    log_path = os.path.join(OUTPUT_ROOT, "run_log", "log_{}.txt".format(shard_id))
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    for i in range(0, len(test_videos)):
    # for i in range(95, len(test_videos)):
        video_name = test_videos[i]
        start_time = time.time()
        input_path = os.path.join(VIDEO_ROOT, video_name + ".mp4")
        output_path = os.path.join(OUTPUT_ROOT, video_name + "_raw.mp4")
        out_pickle_path = os.path.join(OUTPUT_ROOT, video_name + ".pickle")
        
        # if the output file already exists, skip this video
        if os.path.exists(out_pickle_path):
            # load the pickle file and get the metadata
            with open(out_pickle_path, "rb") as f:
                metadata = pickle.load(f)
            frame_count = metadata["frame_count"]
            detected_frame_count = metadata["detected_frame_count"]
            maximum_detected_face = metadata["maximum_detected_face"]
            fps = metadata["fps"]
            frame_width = metadata["frame_width"]
            frame_height = metadata["frame_height"]
            processing_time = metadata["processing_time"]
            flag = metadata["IOU flag"]
            current_log = {"video_name": video_name,
                            "frame_count": frame_count,
                            "detected_frame_count": detected_frame_count,
                            "maximum_detected_face": maximum_detected_face,
                            "processing_time": processing_time,
                            "IOU flag": flag,
                            "fps": fps,
                            "frame_width": frame_width,
                            "frame_height": frame_height}
            run_log.append(current_log)
            with open(log_path, "w") as f:
                json.dump(run_log, f)
            continue
        # load video
        input_video = cv2.VideoCapture(input_path)
        # Create VideoWriter object
        frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if do_output_raw_video:
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # start tracking faces
        face_colors = {}
        last_frame_faces = []
        next_face_id = 0
        bbox_frames = []
        frame_id = 0
        # relevant metadata
        detected_frame_count = 0
        maximum_detected_face = 0
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            while input_video.isOpened():
                success, image = input_video.read()
                if not success:
                    break

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)

                current_frame_faces = []
                if results.detections:
                    detected_frame_count += 1
                    maximum_detected_face = max(maximum_detected_face, len(results.detections))
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = image.shape
                        x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                        
                        current_box = (x, y, width, height)
                        matched = False

                        for last_face in last_frame_faces:
                            if calculate_iou(current_box, last_face[1]) > 0.5:  # IOU threshold
                                face_id = last_face[0]
                                matched = True
                                break

                        if not matched:
                            face_id = next_face_id
                            next_face_id += 1

                        if face_id not in face_colors:
                            face_colors[face_id] = tuple(np.random.randint(0, 255, 3).tolist())

                        color = face_colors[face_id]
                        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
                        mp_drawing.draw_detection(image, detection)

                        current_frame_faces.append((face_id, current_box))
                    bbox_frames.append(current_frame_faces)
                else:
                    bbox_frames.append([])
                last_frame_faces = current_frame_faces
                if do_output_raw_video:
                    out.write(image)
            frame_id += 1
        end_time = time.time()
        input_video.release()
        if do_output_raw_video:
            out.release()

        output_dict = {"raw_bbox_frames": bbox_frames,  # the bounding boxes for each frame, it's a list of lists of lists 
                    "frame_count": frame_count,  # frame count as per opencv
                    "fps": fps,                  # fps as per opencv
                    "frame_width": frame_width,  # frame width
                    "frame_height": frame_height, # frame height
                    "detected_frame_count": detected_frame_count, # number of frames where faces are detected
                    "maximum_detected_face": maximum_detected_face}
        # output the output_dict to a pickle file
        with open(os.path.join(out_pickle_path), "wb") as f:
            pickle.dump(output_dict, f)


        # run the IOU tracker
        processed_boxes, flag = process_bounding_boxes(output_dict, 5)
        # output a video that shows the bounding boxes
        video_path = os.path.join(OUTPUT_ROOT, video_name + ".mp4")
        out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        # load the input video again
        input_video = cv2.VideoCapture(input_path)
        frame_id = 0
        while input_video.isOpened():
            success, image = input_video.read()
            if not success:
                break
            # if processed_boxes[frame_id]:
            try:
                x, y, w, h = processed_boxes[frame_id]
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                out.write(image)
                frame_id += 1
            except:
                break
        out.release()
        input_video.release()
        end_time = time.time()

        output_dict["processed_bbox_frames"] = processed_boxes
        output_dict["processing_time"] = end_time - start_time
        output_dict["IOU flag"] = flag
        
        # output the output_dict to a pickle file
        with open(os.path.join(out_pickle_path), "wb") as f:
            pickle.dump(output_dict, f)

        current_log = {"video_name": video_name, 
                       "frame_count": frame_count, 
                       "detected_frame_count": detected_frame_count, 
                       "maximum_detected_face": maximum_detected_face, 
                       "processing_time": end_time - start_time,
                       "IOU flag": flag, 
                       "fps": fps, 
                       "frame_width": frame_width,
                       "frame_height": frame_height}
        run_log.append(current_log)
        
        with open(log_path, "w") as f:
            json.dump(run_log, f)


        print(f"Processing video {test_videos[i]} with framecount of {frame_count} took {end_time - start_time} seconds")
        