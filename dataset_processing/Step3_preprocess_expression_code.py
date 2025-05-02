import pickle as pkl
import json
import os
import cv2
import argparse
from collections import defaultdict
import time
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from scipy import signal

# Replacing expressionet imports with our placeholder class
import os
import torch
import numpy as np
import math
from omegaconf import OmegaConf
import subprocess
from dataset_processing.transform import crop_v2

# Define placeholder classes for ExpressionCodeExtractor
class ExpressionCodeExtractor:
    """
    Placeholder class for expressionet functionality
    """
    def __init__(self):
        pass

    def __call__(self, image):
        # Placeholder for actual model inference
        # This should return landmarks and code
        return landmarks, code

        
def crop_img(img, bbox, transform, smooth_filter=True):
    if smooth_filter:
        h, w, cx, cy = bbox[:4]
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = w + x1 - 1
        y2 = h + y1 - 1
    else:
        x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = x1 + w // 2
        cy = y1 + h // 2
    center = np.array([cx, cy])

    scale = max(math.ceil(x2) - math.floor(x1),
                math.ceil(y2) - math.floor(y1)) / 200.0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if args.get_processed_videos_spectre:
        input, trans = crop_v2(img, center, scale * 1.15, (224, 224))
    else:
        input, trans = crop_v2(img, center, scale * 1.15, (256, 256))
    # cv2.imwrite(f"cropped_img.png", input)
    input = transform(input).unsqueeze(0)

    return input, trans

def draw_landmark(landmark, image):
    for (x, y) in (landmark + 0.5).astype(np.int32):
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    return image

def draw_landmarks(img, points, mean, std):
    img_np = img.permute(0, 2, 3, 1).cpu().contiguous().numpy()
    img_np = img_np * std + mean
    img_np = img_np * 255
    img_np = img_np.astype(np.uint8)
    images_with_landmarks = []
    for j in range(len(img)):
        image = img_np[j]
        orig_size = image.shape[0]
        this_points = points[j]
        for i, p in enumerate(this_points):
            x, y = p * orig_size
            x = int(x)
            y = int(y)

            cv2.circle(image, (x, y), 1, (255, 255, 0), -1)

        images_with_landmarks.append(image)
    return np.stack(images_with_landmarks)

def process_batch(to_process, output_video, model, mean, std, device, cfg, save_cropping_video):
    to_process = torch.concat(to_process).to(device)
    with torch.no_grad():
        if cfg.MODEL.use_gradient_reversal:
            landmarks, code, _= model(to_process)
        else:
            landmarks, code= model(to_process)
    images_with_landmarks = draw_landmarks(to_process, landmarks, mean, std)
    if save_cropping_video:
        for img in images_with_landmarks:
            output_video.write(img)
    return landmarks, code

if __name__ == "__main__":
    
    BOUNDBOX_ROOT = "/data/celebv-text/boundbox_mediapipe/"
    SHARD_ROOT = "/data/celebv-text/splitting"
    VIDEO_ROOT = "/data/celebv-text/Videos"
    OUT_ROOT = "/data/celebv-text/expression_code"
    OUT_LOG_ROOT = "/data/celebv-text/expression_code/run_log"
    
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(OUT_LOG_ROOT, exist_ok=True)

    parser = argparse.ArgumentParser(description=" Process some file paths.")
    parser.add_argument('--shard_id', type=str, required=True)
    args = parser.parse_args()
    shard_id = args.shard_id
    # shard_id = "test"
    over_crop_ratio = 1.2
    batch_size = 32

    # all smoothing types are ["", "savgol", "savgol_center_and_constant_size"]
    smoothing_type = "savgol_boundbox+smooth_expression"
    save_cropping_video = False
    output_video = None

    smoothing_type = args.smoothing_type
    
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    torch.set_grad_enabled(False)
    # Cuda
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # normalization function
    normalize = transforms.Normalize(
            mean=mean, std=std
        )
    normalize = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


    # load shard
    shard_path = os.path.join(SHARD_ROOT, f"video_split_{args.shard_id}.pkl")
    with open(shard_path, 'rb') as file:
        filenames = pkl.load(file)

    # load model:
    cfg = OmegaConf.load(args.img_encoder_config)
    model = ExpressionCodeExtractor()

    # load checkpoint
    checkpoint = torch.load(args.img_encoder_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # set up a log file
    run_log = []
    log_path = os.path.join(OUT_LOG_ROOT, f"log_{args.shard_id}.txt")

    for smoothing_type in [smoothing_type]:
    # for smoothing_type in ["", "savgol", "savgol_center_and_constant_size", "smooth_expression", "savgol_boundbox+smooth_expression"]:
        # load the log file if it exists
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                run_log = json.load(f)
        for video_i in range(len(filenames)):
            error_flags_i = {"error_extracting_landmarks_and_code": False, "error_loading_video": False, "error_saving_code": False, "error_saving_landmarks": False}
            # video_i = 6
            # all thep paths
            video_path = os.path.join(VIDEO_ROOT, filenames[video_i][0]+".mp4")
            bound_box_path = os.path.join(BOUNDBOX_ROOT, filenames[video_i][0]+".pickle")
            output_video_path = os.path.join(OUT_ROOT, filenames[video_i][0]+"_{}.mp4".format(smoothing_type))
            landmark_out_path = os.path.join(OUT_ROOT, filenames[video_i][0]+"_landmarks_{}.pkl".format(smoothing_type))
            code_out_path = os.path.join(OUT_ROOT, filenames[video_i][0]+"_code_{}.pkl".format(smoothing_type))


            # check and see if the code exist
            if os.path.exists(code_out_path) and os.path.exists(landmark_out_path):
                # load the log from the output path
                print("skipping file: ", filenames[video_i][0])
                continue

            # load the bound box
            with open(bound_box_path, 'rb') as file:
                bound_box_metadata = pkl.load(file)

            fps = bound_box_metadata["fps"]
            fps = int(round(fps))
            im_width = bound_box_metadata["frame_width"]
            im_height = bound_box_metadata["frame_height"]
            bound_box = bound_box_metadata["processed_bbox_frames"]
            bound_box = np.array(bound_box) # x, y, w, h = bound_box[i]
            # turn x and y into cx and cy
            bound_box[:, 0] = bound_box[:, 0] + bound_box[:, 2] // 2
            bound_box[:, 1] = bound_box[:, 1] + bound_box[:, 3] // 2
            
            if smoothing_type == "":
                smoothed_bound_box = bound_box

            elif smoothing_type == "savgol" or smoothing_type == "savgol_boundbox+smooth_expression":
                # smooth the bounding box in first 2 dimension using savgol_filter
                smoothed_bound_box = signal.savgol_filter(bound_box, 5, 2, axis=0)
            
            elif smoothing_type == "savgol_center_and_constant_size":
                smoothed_bound_box = signal.savgol_filter(bound_box, 5, 2, axis=0)
                # make the width and height the highest they are in all the frames
                max_width = np.max(smoothed_bound_box[:, 2])
                max_height = np.max(smoothed_bound_box[:, 3])
                smoothed_bound_box[:, 2] = max_width * args.over_crop_ratio
                smoothed_bound_box[:, 3] = max_height * args.over_crop_ratio
            
            # rearrange bbox to be h, w, cx, cy
            smoothed_bound_box = np.array([smoothed_bound_box[:, 3], smoothed_bound_box[:, 2], smoothed_bound_box[:, 0], smoothed_bound_box[:, 1]]).T
            smoothed_bound_box = smoothed_bound_box.astype(np.int32)
            # perform the cropping and process the video
            to_process = [] # for batch processing    
            all_landmarks = []
            all_code = []
            
            # output video
            if save_cropping_video:
                output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (256, 256))
            # load the video, populate to_process and process then in batch once reach sufficient batch size
            try:
                cap = cv2.VideoCapture(video_path)
                for frame_i in range(len(smoothed_bound_box)):
                    ret, frame = cap.read()
                    if frame is None: break
                    # crop the image
                    bbox = smoothed_bound_box[frame_i]
                    alignment_input, trans = crop_img(frame.copy(), bbox, normalize)
                    to_process.append(alignment_input)
                    if len(to_process) == args.batch_size:
                        landmarks, code = process_batch(to_process, output_video, model, mean, std, device, cfg, save_cropping_video)
                        all_landmarks.append(landmarks)
                        all_code.append(code)
                        to_process = []
                if len(to_process) > 0:
                    landmarks, code = process_batch(to_process, output_video, model, mean, std, device, cfg, save_cropping_video)
                    all_landmarks.append(landmarks)
                    all_code.append(code)
            except:
                error_flags_i["error_extracting_landmarks_and_code"] = True

            # save the landmarks and code
            # print(all_landmarks)
            all_landmarks = torch.cat(all_landmarks)
            all_code = torch.cat(all_code)
            if smoothing_type == "savgol_boundbox+smooth_expression":
                # apply savgol filter to the expression
                all_code = all_code.cpu().numpy()
                all_code = signal.savgol_filter(all_code, 5, 2, axis=0)
                all_code = torch.tensor(all_code).to(device)


            with open(landmark_out_path, "wb") as f:
                pkl.dump(all_landmarks, f)
            with open(code_out_path, "wb") as f:
                pkl.dump(all_code, f)
            cap.release()
            if save_cropping_video:
                output_video.release()
            # see of the landmarks and code are saved correctly
            if not os.path.exists(os.path.join(OUT_ROOT, filenames[video_i][0]+"_landmarks_{}.pkl".format(smoothing_type))):
                error_flags_i["error_saving_landmarks"] = True
            if not os.path.exists(os.path.join(OUT_ROOT, filenames[video_i][0]+"_code_{}.pkl".format(smoothing_type))):
                error_flags_i["error_saving_code"] = True
            # save the log
            run_log.append(error_flags_i)
            with open(log_path, "w") as f:
                json.dump(run_log, f)

            # check if cropped video, npy and alembic are saved correctly
            if save_cropping_video:
                if not os.path.exists(output_video_path):
                    print(f"Error: {output_video_path} not saved")