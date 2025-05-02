import json
import pickle as pkl
import os, sys
import numpy as np
import torch
import tqdm
def levenshtein_distance(str1, str2):
    # Create a matrix to store the distances
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    # Initialize the matrix with zeros
    matrix = [[0 for _ in range(len_str2)] for _ in range(len_str1)]

    # Fill the first row and column with the index values
    for i in range(len_str1):
        matrix[i][0] = i
    for j in range(len_str2):
        matrix[0][j] = j

    # Compute the Levenshtein distance
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1

            # Take the minimum of insertion, deletion, or substitution
            matrix[i][j] = min(matrix[i - 1][j] + 1,       # Deletion
                               matrix[i][j - 1] + 1,       # Insertion
                               matrix[i - 1][j - 1] + cost)  # Substitution

    # The final value is the Levenshtein distance
    return matrix[len_str1 - 1][len_str2 - 1]
if __name__ == "__main__":
    DATASET_ROOT = "/data/celebv-text/"
    HEAD_ORIENTATION_ROOT = os.path.join(DATASET_ROOT, "head_orientations")    
    EXPRESSION_CODE_ROOT =  os.path.join(DATASET_ROOT, "expression_code")
    AUDIO_DATA_ROOT = os.path.join(DATASET_ROOT, "audios")
    BOUND_BOX_ROOT = os.path.join(DATASET_ROOT, "boundbox_mediapipe")
    VIDEO_ROOT = os.path.join(DATASET_ROOT, "videos")
    
    output_key_path = os.path.join(DATASET_ROOT, "keys.txt")
    annotation_path = os.path.join(DATASET_ROOT, "annotations.pkl")

    ############################################################################################################
    # Step 1: get all the video ids
    ############################################################################################################

    video_root_files = os.listdir(VIDEO_ROOT)
    all_video_ids = []
    # list all the video ids from the video root
    for i, video_id in tqdm.tqdm(enumerate(video_root_files)):
        if video_id.endswith(".mp4"):
            all_video_ids.append(video_id.split(".")[0])
    
    ############################################################################################################
    # Step 2: remove all the videos that do not have audio
    ############################################################################################################


    audio_root_files = os.listdir(AUDIO_DATA_ROOT)
    all_audio_ids = []
    # list all the audio ids from the audio root
    for i, audio_id in tqdm.tqdm(enumerate(audio_root_files)):
        if audio_id.endswith(".m4a"):
            all_audio_ids.append(audio_id.split(".")[0])

    video_ids_with_audio = []
    # list all the video ids that have audio
    for video_id in all_video_ids:
        if video_id in all_audio_ids:
            video_ids_with_audio.append(video_id)
    
    usable_videos = video_ids_with_audio # update the list of usable videos
    print(len(all_video_ids) - len(usable_videos))

    ############################################################################################################
    # Step 3: remove all the non-speech videos
    ############################################################################################################


    # load the annotation file
    with open(annotation_path, "rb") as f:
        annotations = pkl.load(f)
    appearance_annotations = annotations["app"]
    action_annotations = annotations["act"]


    # find all unique action annotations
    all_possible_actions = set()
    for key in action_annotations.keys():
        for action in action_annotations[key]:
            all_possible_actions.add(action[0])
    print(all_possible_actions)

    talking_annotations_labels = ["sing", "shout", "whisper", "talk", "read"]

    # sample a single video_id
    video_no_annotation = []
    video_with_speech = []
    video_with_reading = []
    video_with_instruments = []
    video_with_sleeping = []
    excluded_video = []
    for i in range(0, len(usable_videos)):
        try:
            video_id = usable_videos[i]
            app = appearance_annotations[video_id]
            act = action_annotations[video_id]
            # try and find videos where there is any speech
            all_actions = [x[0] for x in act]
            if any(x in talking_annotations_labels for x in all_actions):
                video_with_speech.append(video_id)
            else:
                excluded_video.append(video_id)            
            if "read" in all_actions:
                # print(f"Video ID: {video_id} has reading annotations")
                video_with_reading.append(video_id)
            if "play_instrument music" in all_actions:
                # print(f"Video ID: {video_id} has instrument annotations")
                video_with_instruments.append(video_id)
            if "sleep" in all_actions:
                # print(f"Video ID: {video_id} has sleeping annotations")
                video_with_sleeping.append(video_id)

        except:
            video_no_annotation.append(video_id)
            continue
            # try and find the key in action annotations with the closest match based on levenshtein distance
            all_keys = set(list(action_annotations.keys()))
            all_video_names = set(all_video_ids)
            # find the union - intersection of all keys and all video names
            key_intersection = all_keys.intersection(all_video_names)
            key_difference = all_keys.difference(key_intersection)
            # find the difference between the union and intersection
            min_distance = 1000
            closest_key = None
            for key in key_difference:
                distance = levenshtein_distance(key, video_id)
                if distance < min_distance:
                    min_distance = distance
                    closest_key = key
            print(closest_key, video_id)
    
    print(f"Videos with speech: {len(video_with_speech)}")
    print(f"Videos with no annotation: {len(video_no_annotation)}")
    print(f"Videos with reading: {len(video_with_reading)}")
    print(len(usable_videos) - len(video_with_speech))

    usable_videos = video_with_speech # update the list of usable videos
    potentially_useful_videos = video_no_annotation
    print(len(usable_videos))

    ############################################################################################################
    # Step 4: remove all videos with poor head tracking
    ############################################################################################################

    
    # remove the videos with too many invalid frames:
    log_dir_head_orinetation = os.path.join(HEAD_ORIENTATION_ROOT, "runlog")    

    # track the run logs
    movie_id_no_issue_with_head_orientation = {}  
    movie_id_some_issue_with_head_orientation = {}

    for i in range(0, 9):
        log_file = os.path.join(log_dir_head_orinetation, f"runlog_{i}.json")
        with open(log_file, "r") as f:
            log = json.load(f)
            for log_i in log:
                movie_id_no_issue_with_head_orientation[log_i["video_name"]] = log_i 
    
    for i in range(0, 3):
        log_file = os.path.join(log_dir_head_orinetation, f"runlog_head_orientation_error_{i}.json")
        with open(log_file, "r") as f:
            log = json.load(f)
            for log_i in log:
                movie_id_some_issue_with_head_orientation[log_i["video_name"]] = log_i

    keys_no_issue = list(movie_id_no_issue_with_head_orientation.keys())
    keys_some_issue = list(movie_id_some_issue_with_head_orientation.keys())

    # do a tally
    potential_issues_tracker_pass2 = {"error_too_many_missing_frames": 0, "error_missing_landmark_detection": 0, "error_cant_open_video": 0, "error_cant_open_boundbox": 0, "error_unknown": 0, "unknown and missing": 0} 
    too_many_missing_frames_list = []
    for i in range(0, len(keys_some_issue)):
        log_i = movie_id_some_issue_with_head_orientation[keys_some_issue[i]]
        error_too_many_missing_frames = log_i["error_too_many_missing_frames"]
        potential_issues_tracker_pass2["error_cant_open_boundbox"] += log_i["error_cant_open_boundbox"]
        potential_issues_tracker_pass2["error_unknown"] += log_i["error_unknown"]
        potential_issues_tracker_pass2["error_missing_landmark_detection"] += log_i["error_missing_landmark_detection"]
        potential_issues_tracker_pass2["error_cant_open_video"] += log_i["error_cant_open_video"]
        potential_issues_tracker_pass2["error_too_many_missing_frames"] += log_i["error_too_many_missing_frames"]

    files_in_head_orientations = os.listdir(HEAD_ORIENTATION_ROOT)
    valid_keys = []
    for f_name in files_in_head_orientations:
        if f_name.endswith(".pkl"):
            # see if the video also exists
            video_exists = f_name.split(".")[0] + ".mp4" in files_in_head_orientations
            if video_exists:
                valid_keys.append(f_name.split(".")[0])
    
    
    set_valid = set(valid_keys)
    set_missing_frames = set(too_many_missing_frames_list)
    set_valid = set_valid.difference(set_missing_frames)
    valid_keys = list(set_valid)    
    
    videos_with_valid_tracking = set(usable_videos).intersection(set(valid_keys))
    usable_videos = list(videos_with_valid_tracking)

    print("Number of videos with valid tracking: ", len(usable_videos))
    print("number of videos with invalid tracking: ", len(video_with_speech) - len(usable_videos))
    ############################################################################################################
    # Step 5: remove videos with too much side profiles
    ############################################################################################################

    mostly_sideway_ids = []
    mostly_forward_ids = []
    for i in range(0, len(usable_videos)):
        video_id = usable_videos[i]
        head_orientation_pkl_patch = os.path.join(HEAD_ORIENTATION_ROOT, f"{video_id}.pkl")
        with open(head_orientation_pkl_patch, "rb") as f:
            head_orientation = pkl.load(f)
        # count the number of frames that the head is facing sideway:
        head_orientation = np.array(head_orientation)
        head_orientation_magnitude = np.abs(head_orientation)
        yaw_angle = head_orientation_magnitude[:, 0]
        side_view_frames = np.where(yaw_angle > 50, 1, 0)
        if np.sum(side_view_frames) > 0.5 * len(side_view_frames):
            mostly_sideway_ids.append(video_id)
            print(f"Video ID: {video_id} has too many side views")
        else:
            mostly_forward_ids.append(video_id)
            print(f"Video ID: {video_id} has mostly forward views")
    
    usable_videos = mostly_forward_ids
    print("Number of videos with mostly forward views: ", len(usable_videos))
    print("Number of videos with mostly sideway views: ", len(mostly_sideway_ids))
    
    
    ########################################### Save the keys ################################################
    with open(output_key_path, "w") as f:
        for video_id in usable_videos:
            f.write(f"{video_id}\n")

