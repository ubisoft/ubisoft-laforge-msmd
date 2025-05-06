© [2025] Ubisoft Entertainment. All Rights Reserved

# Model See Model Do: Speech-Driven Facial Animation with Style Control

# Overview:
This repo contains supplementary code for the paper [MSMD](https://ubisoft-laforge.github.io/character/msmd/) to help with reproducibility. The code is based on the diffusion implementation from [Diffposetalk](https://github.com/DiffPoseTalk/DiffPoseTalk) and can be divided into two components:
* **pre-processing pipeline**: a pre-processing pipeline that can be used to process an arbitrary audiovisual dataset (like [Ravdess](https://paperswithcode.com/dataset/ravdess) or [CelebV-text](https://celebv-text.github.io/)) into a latent face representation, like FLAME parameters ([link](https://flame.is.tue.mpg.de/)) or SEREP ([link](https://arxiv.org/abs/2412.14371)) latent code. 
* **training code**: training code contains essential code like dataloader, training loop, and model architecture to aid in reproducing paper results. 

# Setup:

- build a docker image based on the dockerfile in `/docker_environment` and name it msmd
- run the docker image with the following script to set the right path

```
docker run --gpus all -it \
    --name $CONTAINER_NAME \
    --gpus all \
    -v /path_to_this_directory/:/code \
    -v /path_to_dataset/:/data \
    -v /path_to_your_choice_of_output_folder/:/experiments \
    msmd:latest \
    /bin/bash

```

# Running pre-processing pipeline:

To run the pipeline, make sure you place the dataset in this format. The video_split_*.pkl simply contains a list of file names of the videos.

```
/data/ravdess/
├── videos/
│   └── clips/         # Place your input videos here (.mp4 format)
├── audios/
│   └── clips/         # Place the input audios here (.wav/mp3/m4p)
├── splitting/
│   └── video_split_*.pkl # Files containing video splits information
```
***Note that you NEED to have a facial reconstruction model for extracting the expression code (something like the FLAME or SEREP expression parameters). We do not provide it in our code.***

To integrate the facial reconstruction model into the pipeline, simply replace the placeholder class `ExpressionCodeExtractor` in our code for your facial reconstruction model of choice. 

Once the data + model are read. Run all of the code in /dataset_processing/ in sequence (step 1 - step 6) to obtain a processed dataset.


# Training code:

To train the MSMD model as specified in the paper, first prepare both ravdess and celebv-text datasets using the pre-processing pipeline. Then run `training_spec.sh` to run the training loop with the specs used in the paper. Located inside `training_script.py`

# Inference code:

To generate outputs with the trained model, use `inference.py`. You must prepare an input audio file as a 16000hz wav file, as well as two pickle file containing the expression code and head rotation respectively. In addition, you must also prepare a picke file that stores the statistics (mean and std of the expression code + head rotation for normalizing the data, with the following keys):

- `exp_mean`: Mean values for expression coefficients (tensor)
- `exp_std`: Standard deviation values for expression coefficients (tensor)
- `pose_mean`: Mean values for head pose/rotation (tensor)
- `pose_std`: Standard deviation values for head pose/rotation (tensor)

An example input looks like this:

```
python inference.py \
  --model_root /path/to/models \
  --model_name MODEL_NAME \
  --model_iter ITERATION_NUMBER \
  --style_clip_exp_code_path /path/to/expression/code.pkl \
  --style_clip_head_rot_path /path/to/head/rotation.pkl \
  --audio_clip /path/to/audio.wav \
  --coef_dict_path /path/to/coefficient/statistics.pkl \
  --cfg_level 1.4 \
  --output_dir /path/to/output \
  --versions_of_render 1
```

The script generates:
1. Normalized expression coefficients (.pkl file)
2. Normalized head rotation parameters (.pkl file)
These outputs can be used with FLAME or similar 3D face models to render the final animated face video. You would need to implement your own. 
© [2025] Ubisoft Entertainment. All Rights Reserved
