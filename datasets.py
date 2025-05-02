import io
import pickle
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
import os
import numpy as np
import torch
import torchaudio
from torch.utils import data
from scipy.interpolate import interp1d
from tqdm import tqdm
import librosa
import cv2

# https://github.com/pytorch/audio/issues/2950 , https://github.com/pytorch/audio/issues/2356
torchaudio.set_audio_backend('soundfile')
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

class ConcatDataset_with_stats(data.ConcatDataset):
    def __init__(self, datasets, coef_stats):
        super(ConcatDataset_with_stats, self).__init__(datasets)
        self.coef_stats = coef_stats

def get_dataset(args, device, only_eval=False, batch_overfit_size=-1):
    print("loading dataset from ", args.data_root)
    try:
        args.data_root = Path(args.data_root)
    except:
        print("dataroot already a path")

    if batch_overfit_size > 0:
        # if we are batch overfitting, we are also not gonna random pad the audio
        do_random_pad = False
    else:
        do_random_pad = True

    if args.dataset_type == "ravdess+celebv-text-medium":
        data_root = args.data_root

        ravedess_root = Path("/data/ravdess/processed_data")

        train_dataset_ravdess = DatasetPickle(ravedess_root / 'processed_ravdess_30fps_v3.pkl', 
                                           ravedess_root / 'processed_ravdess_30fps_v3_keys_train.txt', 
                                           None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, celebv_text=False, full_dataset=True)
        val_dataset_ravdess = DatasetPickle(ravedess_root / 'processed_ravdess_30fps_v3.pkl',
                                            ravedess_root / 'processed_ravdess_30fps_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, celebv_text=False, full_dataset=True)

        raw_data = {}
        for chunk in DatasetPickle.load_dict_in_chunks_static(args.data_root / "processed_data_30fps_medium_v3.pkl"):
            raw_data.update(chunk)


        train_dataset_celebv_text = DatasetPickle(args.data_root / "processed_data_30fps_medium_v3.pkl",
                                            args.data_root / 'processed_data_30fps_medium_v3_keys_train.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, 
                                            SE=False, pre_loaded_raw_dataset=raw_data,
                                            full_dataset=True)
        val_dataset_celebv_text = DatasetPickle(args.data_root / "processed_data_30fps_medium_v3.pkl",
                                            args.data_root / 'processed_data_30fps_medium_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, 
                                            SE=False, pre_loaded_raw_dataset=raw_data, 
                                            full_dataset=True)
        
        weight_ravdess_train = 1.0 / len(train_dataset_ravdess)
        weight_celebv_text_train = 1.0 / len(train_dataset_celebv_text)
        weight_ravdess_val = 1.0 / len(val_dataset_ravdess)
        weight_celebv_text_val = 1.0 / len(val_dataset_celebv_text)

        weights_train = [weight_celebv_text_train] * len(train_dataset_celebv_text) + [weight_ravdess_train] * len(train_dataset_ravdess)
        weights_val = [weight_celebv_text_val] * len(val_dataset_celebv_text) + [weight_ravdess_val] * len(val_dataset_ravdess)

        train_dataset = data.ConcatDataset([train_dataset_celebv_text, train_dataset_ravdess])
        val_dataset = data.ConcatDataset([val_dataset_celebv_text, val_dataset_ravdess])

        sampler_train = data.WeightedRandomSampler(weights_train, num_samples=len(weights_train), replacement=True)
        sampler_val = data.WeightedRandomSampler(weights_val, num_samples=len(weights_val), replacement=True)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=DatasetPickle.get_collate_fn(SE=False), sampler=sampler_train)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=DatasetPickle.get_collate_fn(SE=False), sampler=sampler_val)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    return train_dataset, val_dataset, train_loader, val_loader

def incremental_mean_and_std(train_dataset, SE=False):
    exp_sum = 0
    exp_sum_of_squares = 0
    pose_sum = 0
    pose_sum_of_squares = 0
    num_elements = 0
    for i in tqdm(range(len(train_dataset))):
        entry_i = train_dataset[i]
        if not SE:
            # Extract expression and pose tensors for both frames
            exp_0 = entry_i[1][0]['motion'][:, :64]
            exp_1 = entry_i[1][1]['motion'][:, :64]
            pose_0 = entry_i[1][0]['motion'][:, 64:]
            pose_1 = entry_i[1][1]['motion'][:, 64:]
        if SE:
            exp_0 = entry_i[0][:, :64]
            exp_1 = entry_i[1][:, :64]
            pose_0 = entry_i[0][:, 64:]
            pose_1 = entry_i[1][:, 64:]         
        # Update sum and sum of squares for expressions
        exp_sum += exp_0.sum(dim=0)
        exp_sum_of_squares += (exp_0 ** 2).sum(dim=0)
        exp_sum += exp_1.sum(dim=0)
        exp_sum_of_squares += (exp_1 ** 2).sum(dim=0)

        # Update sum and sum of squares for poses
        pose_sum += pose_0.sum(dim=0)
        pose_sum_of_squares += (pose_0 ** 2).sum(dim=0)
        pose_sum += pose_1.sum(dim=0)
        pose_sum_of_squares += (pose_1 ** 2).sum(dim=0)
        
        # Update the total number of elements processed
        num_elements += exp_0.shape[0] + exp_1.shape[0]

    # Compute the mean for expressions and poses
    exp_mean = exp_sum / num_elements
    pose_mean = pose_sum / num_elements

    # Compute the variance for expressions and poses
    exp_var = (exp_sum_of_squares / num_elements) - (exp_mean ** 2)
    pose_var = (pose_sum_of_squares / num_elements) - (pose_mean ** 2)

    # Standard deviation is the square root of variance
    exp_std = torch.sqrt(exp_var)
    pose_std = torch.sqrt(pose_var)

    return exp_mean, exp_std, pose_mean, pose_std

class DatasetPickle(data.Dataset):
    @staticmethod
    def load_dict_in_chunks_static(file_path):
        """
        Load a dictionary in chunks from a pickle file.
        """
        with open(file_path, 'rb') as f:
            while True:
                try:
                    chunk = pickle.load(f)
                    yield chunk
                except EOFError:
                    break

    def load_dict_in_chunks(self, file_path):
        """
        Load a dictionary in chunks from a pickle file.
        """
        with open(file_path, 'rb') as f:
            while True:
                try:
                    chunk = pickle.load(f)
                    yield chunk
                except EOFError:
                    break  # End of file reached
    
    def __init__(self, pkl_file, split_file, coef_stats_file=None, original_fps=30, coef_fps=25, n_motions=100,
                 rot_repr='aa', no_head_pose=False, clip_len=100, device='cpu', SE=False, full_dataset=False, 
                 pre_loaded_raw_dataset=None, celebv_text=True, random_crop=True, batch_overfit_size=-1):
        self.split_file = split_file
        self.pkl_file = pkl_file
        self.valid_id = []
        # load the valid id file (only for celebv-text)
        if celebv_text:
            with open("/data/celebv-text/keys.txt", 'r') as f:
                for line in f:
                    self.valid_id.append(line.strip())
        # load the split file
        self.file_names = []
        with open(split_file, 'r') as f:
            for line in f:
                name = line.strip()
                # self.file_names.append(line.strip())
                if celebv_text:
                    if name in self.valid_id:
                        self.file_names.append(name)
                else:
                    self.file_names.append(name)
        # if overfit_mode is not -1, only take the first overfit_mode entries
        if batch_overfit_size > 0:
            self.file_names = self.file_names[:batch_overfit_size]

        # load the data 
        if pre_loaded_raw_dataset is not None:
            raw_data = pre_loaded_raw_dataset
        elif not full_dataset:
            raw_data = pickle.load(open(pkl_file, 'rb'))
        else:
            raw_data = {}
            for chunk in self.load_dict_in_chunks(pkl_file):
                raw_data.update(chunk)
        self.data = {}
        for key in self.file_names:
            self.data[key] = raw_data[key]
        
        # resample the data to 25 fps:
        # resample the head_orientation and expression_code to 25 fps from 30 fps
        if original_fps != coef_fps:
            for key in self.file_names:
                original_dict = self.data[key]
                original_expression_code = original_dict["expression_code"]
                original_head_orientation = original_dict["head_orientation"]
                new_dict = {"audio": original_dict["audio"]}
                # resample expressioncode to 25 fps down from 30 using interp1d
                # original_expression_code.shape
                x = np.linspace(0, 1, num=original_expression_code.shape[0])
                xnew = np.linspace(0, 1, num=int(round(original_expression_code.shape[0]/original_fps*coef_fps)))
                f_exp = interp1d(x, original_expression_code, axis=0)
                new_expression_code = f_exp(xnew)
                # also resample head_orientation to 25 fps
                f_head = interp1d(x, original_head_orientation, axis=0)
                new_head_orientation = f_head(xnew)
                new_dict["expression_code"] = new_expression_code
                new_dict["head_orientation"] = new_head_orientation
                self.data[key] = new_dict
                del original_dict
        print("finished data resampling")

        if coef_stats_file is not None:
            coef_stats = dict(np.load(coef_stats_file))
            self.coef_stats = {x: torch.tensor(coef_stats[x]) for x in coef_stats}
        else:
            self.coef_stats = None
            print('Warning: No stats file found. Coef will not be normalized.')
        self.device = device
        self.coef_fps = coef_fps
        self.clip_len = clip_len
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = int(self.n_motions * 2.1)
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.random_crop = random_crop
        self.rot_representation = rot_repr
        self.no_head_pose = no_head_pose
        self.SE = SE

        # Read split file
        self.entries = self.file_names
        if coef_stats_file is None:
            exp_mean, exp_std, pose_mean, pose_std = incremental_mean_and_std(self, self.SE)
            self.coef_stats = {}
            self.coef_stats['exp_mean'] = exp_mean   
            self.coef_stats['exp_std'] = exp_std
            self.coef_stats['pose_mean'] = pose_mean
            self.coef_stats['pose_std'] = pose_std
        self.coef_stats = {x: torch.tensor(self.coef_stats[x]).float() for x in self.coef_stats}

    def __len__(self):
        return len(self.entries)
        
    def __getitem__(self, index):
        clip_dict = self.data[self.entries[index]]
        audio = clip_dict["audio"]
        expression_code = clip_dict["expression_code"]
        head_orientation = clip_dict["head_orientation"]

        # normalize the audio
        audio_mean = audio.mean() # note these are calculated before padding to ensure that the mean and std are normalized correctly.
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)
        
        # length of the goal clip
        goal_total_length = self.coef_total_len
        goal_each_clip_length = self.clip_len
        
        # length of the current clip
        current_clip_length = expression_code.shape[0]
        
        # select a starting frame to ensure that after cropping, the second clip will have at least half of goal_each_clip_length
        if self.random_crop:
            if current_clip_length > goal_total_length:
                start_frame1 = np.random.randint(0, current_clip_length - goal_total_length + 1)
                end_frame1 = start_frame1 + goal_each_clip_length
                start_frame2 = start_frame1 + goal_each_clip_length
                end_frame2 = start_frame2 + goal_each_clip_length
            elif current_clip_length == goal_total_length:
                start_frame1 = 0
                end_frame1 = goal_each_clip_length
                start_frame2 = goal_each_clip_length
                end_frame2 = goal_each_clip_length * 2
            else:
                frames_to_pad = goal_total_length - current_clip_length
                # split this down the middle randomly
                frames_to_pad_front = np.random.randint(0, frames_to_pad)
                frames_to_pad_back = frames_to_pad - frames_to_pad_front
                frames_to_pad_front = int(round(frames_to_pad_front))
                frames_to_pad_back = int(round(frames_to_pad_back))
                expression_code = np.pad(expression_code, ((frames_to_pad_front, frames_to_pad_back), (0, 0)), 'constant', constant_values=0)
                head_orientation = np.pad(head_orientation, ((frames_to_pad_front, frames_to_pad_back), (0, 0)), 'constant', constant_values=0)
                
                # audio frames to pad = frames_to_pad * audio_unit
                # note that the audio might be slightly shorter or longer than the video, so we need to pad the audio twice
                audio_frames_to_pad_front = int(round(frames_to_pad_front * self.audio_unit))
                audio_frames_to_pad_back = int(round(frames_to_pad_back * self.audio_unit))
                
                audio = np.pad(audio, ((int(audio_frames_to_pad_front), int(audio_frames_to_pad_back))), 'constant', constant_values=0)
                audio_length = audio.shape[0]

                # if the audio is still shorter than the goal length, pad it with zeros at the end (this is not elegant but what can we do......)
                audio_minimal_length = goal_total_length * self.audio_unit
                audio_minimal_length = int(round(audio_minimal_length))
                if audio_length < audio_minimal_length:
                    audio = np.pad(audio, (0, audio_minimal_length - audio_length), 'constant', constant_values=0)

                start_frame1 = 0
                end_frame1 = goal_each_clip_length
                start_frame2 = goal_each_clip_length
                end_frame2 = goal_each_clip_length * 2
            # Crop the audio and coef
        else:
            start_frame1 = 0
            end_frame1 = goal_each_clip_length
            start_frame2 = goal_each_clip_length
            end_frame2 = goal_each_clip_length * 2

            # pad the audio and coef at the end 
            expression_code = np.pad(expression_code, ((0, int(round(goal_total_length - current_clip_length))), (0, 0)), 'constant', constant_values=0)
            head_orientation = np.pad(head_orientation, ((0, int(round(goal_total_length - current_clip_length))), (0, 0)), 'constant', constant_values=0)
            audio = np.pad(audio, (0, int(round(goal_total_length * self.audio_unit)) - audio.shape[0]), 'constant', constant_values=0)

        expression_code_frame_0 = expression_code[start_frame1:end_frame1]
        expression_code_frame_1 = expression_code[start_frame2:end_frame2]
        head_orientation_frame_0 = head_orientation[start_frame1:end_frame1]
        head_orientation_frame_1 = head_orientation[start_frame2:end_frame2]
        audio_frame_0 = audio[int(start_frame1 * self.audio_unit):int(end_frame1 * self.audio_unit)]
        audio_frame_1 = audio[int(start_frame2 * self.audio_unit):int(end_frame2 * self.audio_unit)]

        # concatenate expression and head orientation
        expression_code_frame_0 = torch.tensor(expression_code_frame_0).float()
        expression_code_frame_1 = torch.tensor(expression_code_frame_1).float()
        head_orientation_frame_0 = torch.tensor(head_orientation_frame_0).float()
        head_orientation_frame_1 = torch.tensor(head_orientation_frame_1).float()

        # normalize coef if applicable
        if self.coef_stats is not None:
            expression_code_frame_0 = (expression_code_frame_0 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            expression_code_frame_1 = (expression_code_frame_1 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            head_orientation_frame_0 = (head_orientation_frame_0 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
            head_orientation_frame_1 = (head_orientation_frame_1 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)

        motion_coef_frame_0 = torch.cat([expression_code_frame_0, head_orientation_frame_0], axis=-1)
        motion_coef_frame_1 = torch.cat([expression_code_frame_1, head_orientation_frame_1], axis=-1)

        shape_frame_0 = torch.zeros((motion_coef_frame_0.shape[0], 100)).float()
        shape_frame_1 = torch.zeros((motion_coef_frame_1.shape[0], 100)).float()

        coef_dict_0 = {"shape": shape_frame_0, "motion": motion_coef_frame_0}
        coef_dict_1 = {"shape": shape_frame_1, "motion": motion_coef_frame_1}

        # turning all the numpy arrays into torch tensors
        audio_frame_0 = torch.tensor(audio_frame_0).float()
        audio_frame_1 = torch.tensor(audio_frame_1).float()
        
        if self.SE:
            return [motion_coef_frame_0, motion_coef_frame_1]
        else:
            return [audio_frame_0, audio_frame_1], [coef_dict_0, coef_dict_1], (audio_mean, audio_std)
    
    def get_k_indices_for_each_emotion(self, k=2):
        # 1 to 8
        emotions = ["01", "02", "03", "04", "05", "06", "07", "08"]
        emotion_indices = {}
        for emotion in emotions:
            emotion_indices[emotion] = []
            for count in range(0, k):
                # randomly sample from the entries
                index = np.random.randint(0, len(self.entries))
                count = 0
                while self.entries[index].split("-")[2] != emotion:
                    index = np.random.randint(0, len(self.entries))
                    count += 1
                    if count >= 100:
                        break
                if self.entries[index].split("-")[2] == emotion:
                    emotion_indices[emotion].append(index)
                else:
                    continue
        return emotion_indices

    def query_for_video(self, index):
        video_name = self.entries[index]
        if not video_name in self.file_names:
            Exception("Video name not found in the dataset")
        expression_code = self.data[video_name]["expression_code"]
        pose = self.data[video_name]["head_orientation"]
        audio = self.data[video_name]["audio"]
        # normalize the audio
        audio_mean = audio.mean() # note these are calculated before padding to ensure that the mean and std are normalized correctly.
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)
        
        # length of the current clip
        current_clip_length = expression_code.shape[0]
        
        # concatenate expression and head orientation
        expression_code = torch.tensor(expression_code).float()
        pose = torch.tensor(pose).float()
        # normalize coef if applicable
        if self.coef_stats is not None:
            expression_code = (expression_code - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            pose = (pose - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
        
        motion = torch.cat([expression_code, pose], axis=-1)
        shape = torch.zeros((motion.shape[0], 100)).float()
        coef_dict = {"shape": shape, "motion": motion}
        
        # turning all the numpy arrays into torch tensors
        audio = torch.tensor(audio).float()

        return audio, coef_dict, (audio_mean, audio_std)

    @staticmethod
    def get_collate_fn(SE):
        def pad_or_trim_audio(audio_tensor, target_length=64000):
            """Helper function to ensure all audio tensors have the same length"""
            current_length = audio_tensor.size(0)
            if current_length < target_length:
                # Pad with zeros
                padding = target_length - current_length
                return torch.nn.functional.pad(audio_tensor, (0, padding), 'constant', 0)
            elif current_length > target_length:
                # Trim to target length
                return audio_tensor[:target_length]
            return audio_tensor

        def collate_fn(batch):
            if SE:
                coef_0 = []
                coef_1 = []
                for i in range(len(batch)):
                    coef_0.append(batch[i][0])
                    coef_1.append(batch[i][1])
                coef_0 = torch.stack(coef_0, dim=0)
                coef_1 = torch.stack(coef_1, dim=0)
                return [coef_0, coef_1]
            else:
                audio_0 = []
                audio_1 = []
                motion_0 = []
                motion_1 = []
                shape_0 = []
                shape_1 = []
                audio_mean = []
                audio_std = []

                # Fixed target length for audio
                target_length = 64000

                # Process each item in batch
                for i in range(len(batch)):
                    # Pad or trim audio to target length
                    audio_0_padded = pad_or_trim_audio(batch[i][0][0], target_length)
                    audio_1_padded = pad_or_trim_audio(batch[i][0][1], target_length)
                    
                    # Append processed items
                    audio_0.append(audio_0_padded)
                    audio_1.append(audio_1_padded)
                    motion_0.append(batch[i][1][0]["motion"])
                    motion_1.append(batch[i][1][1]["motion"])
                    shape_0.append(batch[i][1][0]["shape"])
                    shape_1.append(batch[i][1][1]["shape"])
                    audio_mean.append(batch[i][2][0])
                    audio_std.append(batch[i][2][1])

                try:
                    # Stack all tensors
                    audio_0 = torch.stack(audio_0, dim=0)
                    audio_1 = torch.stack(audio_1, dim=0)
                    motion_0 = torch.stack(motion_0, dim=0)
                    motion_1 = torch.stack(motion_1, dim=0)
                    shape_0 = torch.stack(shape_0, dim=0)
                    shape_1 = torch.stack(shape_1, dim=0)
                except RuntimeError as e:
                    shapes_info = {
                        'audio_0': [x.shape for x in audio_0],
                        'audio_1': [x.shape for x in audio_1],
                        'motion_0': [x.shape for x in motion_0],
                        'motion_1': [x.shape for x in motion_1],
                        'shape_0': [x.shape for x in shape_0],
                        'shape_1': [x.shape for x in shape_1]
                    }
                    raise RuntimeError(f"Failed to stack tensors. Shapes: {shapes_info}. Original error: {str(e)}")

                # Process audio statistics
                audio_mean = torch.tensor(audio_mean).float().mean()
                audio_std = torch.tensor(audio_std).float().mean()

                # Create coefficient dictionaries
                coef_0 = {"shape": shape_0, "motion": motion_0}
                coef_1 = {"shape": shape_1, "motion": motion_1}

                return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)

        return collate_fn