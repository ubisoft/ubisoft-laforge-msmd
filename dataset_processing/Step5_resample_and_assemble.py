import lmdb
import pickle as pkl
import cv2
import torch
import numpy as np
import os
import sys
import audioread
import librosa
from scipy import signal
import argparse
from tqdm import tqdm
def save_dict_in_chunks(data, file_path, chunk_size=1000):
    """
    Save a dictionary in chunks to a pickle file.

    Parameters:
    - data: The large dictionary to save.
    - file_path: Path where to save the pickle file.
    - chunk_size: Number of key-value pairs to include in each chunk.
    """
    with open(file_path, 'wb') as f:
        keys = list(data.keys())
        for i in tqdm(range(0, len(keys), chunk_size)):
            chunk = {k: data[k] for k in keys[i:i + chunk_size]}
            pkl.dump(chunk, f)

def load_dataset_pkl(file_path):
    def load_dict_in_chunks(file_path):
        """
        Load a dictionary in chunks from a pickle file.
        """
        with open(file_path, 'rb') as f:
            while True:
                try:
                    chunk = pkl.load(f)
                    yield chunk
                except EOFError:
                    break  # End of file reached

    # Example usage:
    loaded_dict = {}
    for chunk in load_dict_in_chunks(file_path):
        loaded_dict.update(chunk)  # Combine chunks into a single dictionary
    return loaded_dict

if __name__ == "__main__":
    DATASET_ROOT = "/data/celebv-text/"
    HEAD_ORIENTATION_ROOT = os.path.join(DATASET_ROOT, "head_orientations")    
    EXPRESSION_CODE_ROOT =  os.path.join(DATASET_ROOT, "expression_code")
    AUDIO_DATA_ROOT = os.path.join(DATASET_ROOT, "audios")
    BOUND_BOX_ROOT = os.path.join(DATASET_ROOT, "boundbox_mediapipe")
    VIDEO_ROOT = os.path.join(DATASET_ROOT, "videos")
    # PROCESSED_DATA_ROOT = os.path.join(DATASET_ROOT, "processed_data")
    PROCESSED_DATA_ROOT = os.path.join(DATASET_ROOT, "processed_data")
    os.makedirs(PROCESSED_DATA_ROOT, exist_ok=True)
    
    # output pickle file 30fps

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--goal_fps", type=int, default=30)
    arg_parser.add_argument("--goal_sr", type=int, default=16000)
    args = arg_parser.parse_args()
    goal_fps = args.goal_fps
    goal_sr = args.goal_sr
    goal_fps = 30
    output_pickle_path = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_v3.pkl".format(goal_fps))
    output_lmdb_path = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_v3.lmdb".format(goal_fps))
    error_files_path = os.path.join(PROCESSED_DATA_ROOT, "error_files_v3.pkl")

    compute_histogram_of_fps = False
    
    
    # get all the name to the videos which has been filtered
    valid_key_file_path = os.path.join(DATASET_ROOT, "keys.txt")
    with open(valid_key_file_path, "r") as f:
        valid_keys = f.read().splitlines()
    # get all the keys in the existing lmdb and subtract them from valid keys
    # make the map_size 100gb
    pickle_30fps = {}
    error_files = []
    if os.path.exists(output_lmdb_path):
        env = lmdb.open(output_lmdb_path, map_size=10995116277, )
        with env.begin() as txn:
            # use a tqdm to do the following
            # keys = [key.decode() for key, _ in txn.cursor()]
            keys = []
            for key, _ in tqdm(txn.cursor()):
                keys.append(key.decode())
                tqdm.write(f"Found key {key.decode()}")
                pickle_30fps[key.decode()] = pkl.loads(txn.get(key))
        env.close()
        valid_keys = list(set(valid_keys) - set(keys))
    
    if os.path.exists(error_files_path):
        error_files = pkl.load(open(error_files_path, "rb"))
    
    if compute_histogram_of_fps:
        fps = []
        # I sample 1000
        for i in range(0, 1000):
            print(f"Processing video {i} of {len(valid_keys)}")
            video_id = valid_keys[i]
            video_file = os.path.join(VIDEO_ROOT, f"{video_id}.mp4")
            video_cap = cv2.VideoCapture(video_file)
            fps.append(video_cap.get(cv2.CAP_PROP_FPS))
            video_cap.release()
        
        # plot a histogram and save it 
        import matplotlib.pyplot as plt
        plt.hist(fps, bins=100)
        plt.savefig("/experiments/fps_hist.png")
        plt.close()

    # initialize the lmdb file
    env = lmdb.open(output_lmdb_path, map_size=1099511627776, )
    with env.begin(write=True) as txn:
        # for i in range(0, 20):
        for i in range(0, len(valid_keys)):
            try:
                video_id = valid_keys[i]
                print(f"Processing video {video_id}, {i} of {len(valid_keys)}")
                head_orientation_file = os.path.join(HEAD_ORIENTATION_ROOT, f"{video_id}.pkl")
                expression_code_file = os.path.join(EXPRESSION_CODE_ROOT, f"{video_id}_code_savgol_boundbox+smooth_expression.pkl")
                audio_file = os.path.join(AUDIO_DATA_ROOT, f"{video_id}.m4a")
                bound_box_file = os.path.join(BOUND_BOX_ROOT, f"{video_id}.pickle")
                video_file = os.path.join(VIDEO_ROOT, f"{video_id}.mp4")
                
                # load everything for file i
                head_orientation = pkl.load(open(head_orientation_file, "rb")) # numpy file
                expression_code = pkl.load(open(expression_code_file, "rb")) # torch array
                expression_code = expression_code.detach().cpu().numpy()
                audio = librosa.load(audio_file, sr=None)
                video_cap = cv2.VideoCapture(video_file)
                # check if the video is valid
                fps = video_cap.get(cv2.CAP_PROP_FPS)
                print("fps is ", fps)
                sr = audio[1]
                audio_arr = audio[0]
                # if the audio has more than 1 channel, we take the first channel
                if len(audio_arr.shape) > 1:
                    audio_arr = audio_arr[:, 0]
                # resample head_orientation and expression_code to goal fps
                head_orientation_resampled = signal.resample(head_orientation, int(len(head_orientation) * goal_fps / video_cap.get(cv2.CAP_PROP_FPS)))
                expression_code_resampled = signal.resample(expression_code, int(len(expression_code) * goal_fps / video_cap.get(cv2.CAP_PROP_FPS)))
                # resample audio to goal sr
                audio_resampled = signal.resample(audio_arr, int(len(audio_arr) * goal_sr / sr))

                # save to a lmdb file

                txn.put(video_id.encode(), pkl.dumps({"head_orientation": head_orientation_resampled, 
                                                        "expression_code": expression_code_resampled, 
                                                        "audio": audio_resampled}))
                if i % 100 == 0 or i == 10:
                    txn.commit()
                    txn = env.begin(write=True)
                # save to a pickle file
                pickle_30fps[video_id] = {"head_orientation": head_orientation_resampled, 
                                        "expression_code": expression_code_resampled, 
                                        "audio": audio_resampled}
                video_cap.release()
            except:
                error_files.append(video_id)
                pkl.dump(error_files, open(error_files_path, "wb"))
                print(f"Error processing video {video_id}")
                continue            
    env.close()
