import cv2 as cv
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
import os
import pickle as pkl
from scipy.signal import savgol_filter
import numpy as np
from scipy.signal import savgol_filter
import argparse
import time

def smooth_rotation_matrices(rotation_matrices, window_length=7, polyorder=3):
	"""
	Smooths a sequence of rotation matrices using the Savitzky-Golay filter applied to quaternions.

	Parameters:
		rotation_matrices (list or ndarray): A list or array of rotation matrices (each of shape (3, 3)).
		window_length (int): The length of the filter window (number of coefficients). Must be a positive odd integer.
		polyorder (int): The order of the polynomial used to fit the samples. Must be less than window_length.

	Returns:
		smoothed_rotation_matrices (list): A list of smoothed rotation matrices.
	"""
	# Ensure input is an array
	rotation_matrices = np.asarray(rotation_matrices)

	# Convert rotation matrices to quaternions
	quaternions = np.array([Rotation.from_matrix(R_mat).as_quat() for R_mat in rotation_matrices])

	# Ensure quaternion signs are consistent
	for i in range(1, len(quaternions)):
		if np.dot(quaternions[i], quaternions[i - 1]) < 0:
			quaternions[i] = -quaternions[i]

	# Apply Savitzky-Golay filter to quaternion components
	smoothed_quaternions = np.zeros_like(quaternions)
	for i in range(4):
		smoothed_quaternions[:, i] = savgol_filter(
			quaternions[:, i], window_length=window_length, polyorder=polyorder, mode='interp'
		)

	# Renormalize quaternions to maintain unit norm
	norms = np.linalg.norm(smoothed_quaternions, axis=1)
	smoothed_quaternions /= norms[:, np.newaxis]

	# Convert back to rotation matrices
	smoothed_rotation_matrices = [Rotation.from_quat(q).as_matrix() for q in smoothed_quaternions]

	return smoothed_rotation_matrices

def rotation_matrix_from_vectors(vec1, vec2):
	"""Find the rotation matrix that aligns vec1 to vec2."""
	a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([
		[0, -v[2], v[1]],
		[v[2], 0, -v[0]],
		[-v[1], v[0], 0]
	])
	rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
	return rotation_matrix

def procrustes_analysis(X, Y):
	"""
	Perform Procrustes analysis to find the best fitting rotation, scale, and translation.
	"""
	mu_x = X.mean(axis=1)
	mu_y = Y.mean(axis=1)
	rho2_x = X.var(axis=1).sum()
	cov_xy = (1.0 / X.shape[1]) * (Y - mu_y[:, np.newaxis]) @ (X - mu_x[:, np.newaxis]).T
	# SVD on the covariance matrix
	U, D, V_T = np.linalg.svd(cov_xy)
	# Prepare sign flipping matrix S
	S = np.identity(3)
	if np.linalg.matrix_rank(cov_xy) >= X.shape[0] - 1:
		if np.linalg.det(cov_xy) < 0:
			S[-1, -1] = -1 
	else:
		det_U = np.linalg.det(U)
		det_V = np.linalg.det(V_T)
		if det_U * det_V < 0:
			S[-1, -1] = -1  
	# Compute rotation, scale, and translation
	R = U @ S @ V_T
	c = (1.0 / rho2_x) * np.sum(D * np.diag(S))
	t = mu_y - c * R @ mu_x
	return R, c, t[:, np.newaxis]

def rotateToNeutral(neutralPose, data, staticIndices, returnRotation=False):
	"""
	Align data to neutralPose using Procrustes analysis.
	"""
	outData = np.zeros(data.shape)
	R_out = []
	t_out = []
	for i in range(data.shape[0]):
		frame_t = data[i, staticIndices]
		R, c, t = procrustes_analysis(frame_t.T, neutralPose[staticIndices].T)
		if returnRotation:
			R_out.append(R)
			t_out.append(t)
		outData[i] = (c * R @ data[i].T + t).T
	if returnRotation:
		return outData, R_out, t_out
	else:
		return outData

def compute_bounding_box(landmarks, image_width, image_height):
	"""
	Compute the bounding box of face landmarks in pixel coordinates.
	"""
	x_coords = [landmark.x * image_width for landmark in landmarks]
	y_coords = [landmark.y * image_height for landmark in landmarks]
	x_min = max(int(min(x_coords)), 0)
	x_max = min(int(max(x_coords)), image_width - 1)
	y_min = max(int(min(y_coords)), 0)
	y_max = min(int(max(y_coords)), image_height - 1)
	return (x_min, y_min, x_max - x_min, y_max - y_min)  # (x, y, w, h)

def calculate_iou(boxA, boxB):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.
	"""
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
	yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

	# Compute the area of intersection rectangle
	interWidth = max(0, xB - xA)
	interHeight = max(0, yB - yA)
	interArea = interWidth * interHeight

	# Compute the area of both bounding boxes
	boxAArea = boxA[2] * boxA[3]
	boxBArea = boxB[2] * boxB[3]

	# Compute the IoU
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def interpolate_rotation_matrices(rotation_matrices):
	"""
	Interpolates a list of rotation matrices with missing entries (None).

	Parameters:
	- rotation_matrices: list of 3x3 numpy arrays or None.

	Returns:
	- interpolated_matrices: list of 3x3 numpy arrays with interpolated rotations.
	- error_log: dictionary with processing information.
	"""
	start_time = time.time()
	num_frames = len(rotation_matrices)
	interpolated_matrices = [None] * num_frames

	# Identify valid indices
	valid_indices = [i for i, R in enumerate(rotation_matrices) if R is not None]

	error_log = {}

	# Check for the case where there are no valid frames
	if not valid_indices:
		# Return identity rotations or raise an error
		identity_matrix = np.eye(3)
		interpolated_matrices = [identity_matrix for _ in range(num_frames)]
		error_log["processing_time"] = time.time() - start_time
		error_log["longest_consecutive_missing_frames"] = num_frames
		error_log["has_missing_frames_with_surrounding"] = False
		error_log["has_missing_frames_at_start"] = True
		error_log["has_missing_frames_at_end"] = True
		return interpolated_matrices, error_log

	# Handle missing frames at the start
	has_missing_frames_at_start = valid_indices[0] > 0
	if has_missing_frames_at_start:
		for i in range(0, valid_indices[0]):
			interpolated_matrices[i] = rotation_matrices[valid_indices[0]]

	# Handle missing frames at the end
	has_missing_frames_at_end = valid_indices[-1] < num_frames - 1
	if has_missing_frames_at_end:
		for i in range(valid_indices[-1] + 1, num_frames):
			interpolated_matrices[i] = rotation_matrices[valid_indices[-1]]
			
	has_missing_frames_with_surrounding = False
	longest_consecutive_missing_frames = 0
	current_missing_streak = 0

	# Process valid frames and interpolate missing ones
	for idx in range(len(valid_indices) - 1):
		start_idx = valid_indices[idx]
		end_idx = valid_indices[idx + 1]
		gap = end_idx - start_idx - 1

		# Copy valid frames directly
		interpolated_matrices[start_idx] = rotation_matrices[start_idx]

		if gap > 0:
			has_missing_frames_with_surrounding = True
			# Update longest consecutive missing frames
			if gap > longest_consecutive_missing_frames:
				longest_consecutive_missing_frames = gap

			# Convert the surrounding rotation matrices to quaternions
			R_start = rotation_matrices[start_idx]
			R_end = rotation_matrices[end_idx]

			quat_start = Rotation.from_matrix(R_start)
			quat_end = Rotation.from_matrix(R_end)

			# Interpolate missing frames using SLERP
			times = np.linspace(0, 1, gap + 2)
			slerp = Rotation.slerp(times[0], times[-1], [quat_start, quat_end])
			interpolated_rots = slerp(times)

			# Assign interpolated rotations to the missing frames
			for i in range(1, gap + 1):
				interpolated_matrices[start_idx + i] = interpolated_rots[i].as_matrix()
		else:
			# No missing frames in this segment
			current_missing_streak = 0

	# Copy the last valid frame
	last_valid_idx = valid_indices[-1]
	interpolated_matrices[last_valid_idx] = rotation_matrices[last_valid_idx]

	# Handle any remaining missing frames (if any)
	# This also calculates the longest consecutive missing frames
	max_missing = 0
	current_missing_streak = 0
	for i in range(num_frames):
		if interpolated_matrices[i] is None:
			current_missing_streak += 1
			if current_missing_streak > max_missing:
				max_missing = current_missing_streak
			# Fill missing frames at the start or end
			if i < valid_indices[0]:
				interpolated_matrices[i] = rotation_matrices[valid_indices[0]]
			elif i > valid_indices[-1]:
				interpolated_matrices[i] = rotation_matrices[valid_indices[-1]]
		else:
			current_missing_streak = 0

	longest_consecutive_missing_frames = max_missing

	# Compute error_log information
	total_processing_time = time.time() - start_time

	error_log = {
		"processing_time": total_processing_time,
		"longest_consecutive_missing_frames": longest_consecutive_missing_frames,
		"has_missing_frames_with_surrounding": has_missing_frames_with_surrounding,
		"has_missing_frames_at_start": has_missing_frames_at_start,
		"has_missing_frames_at_end": has_missing_frames_at_end,
	}

	return interpolated_matrices, error_log

def interpolate_landmarks(facial_landmarks):
	valid_indices = [i for i, lm in enumerate(facial_landmarks) if lm is not None]
	error_log = {"missing_start_frames": False, "missing_end_frames": False, "missing_frames_in_between": False, "longest_consecutive_missing_frames": 0}
	
	# fill indices at the start of the array:
	for i in range(0, valid_indices[0]):
		facial_landmarks[i] = facial_landmarks[valid_indices[0]]
		error_log["missing_start_frames"] = True
		error_log["longest_consecutive_missing_frames"] = max(error_log["longest_consecutive_missing_frames"], valid_indices[0] - i)

	# fill indices at the end of the array:
	for i in range(valid_indices[-1] + 1, len(facial_landmarks)):
		facial_landmarks[i] = facial_landmarks[valid_indices[-1]]
		error_log["missing_end_frames"] = True
		error_log["longest_consecutive_missing_frames"] = max(error_log["longest_consecutive_missing_frames"], i - valid_indices[-1])

	# fill the rest of the indices by interpolating between nearby valid frames
	for idx in range(len(valid_indices) - 1):
		start_idx = valid_indices[idx]
		end_idx = valid_indices[idx + 1]
		gap = end_idx - start_idx - 1
		if gap > 0:
			in_between_indices = np.linspace(start_idx+1, end_idx-1, gap).astype(int)
			# interpolate between the two valid frames
			for i, in_between_idx in enumerate(in_between_indices):
				t = (i + 1) / (gap + 2)
				facial_landmarks[in_between_idx] = (1 - t) * facial_landmarks[start_idx] + t * facial_landmarks[end_idx]
			error_log["missing_frames_in_between"] = True
			error_log["longest_consecutive_missing_frames"] = max(error_log["longest_consecutive_missing_frames"], gap)
		
	return facial_landmarks, error_log





class ObjLoader(object):
	def __init__(self, fileName):
		self.vertices = []
		self.faces = []
		self.transformed_vertices = []
		self.transformed_faces = []
		try:
			with open(fileName) as f:
				for line in f:
					if line.startswith("v "):
						parts = line.strip().split()
						vertex = tuple(map(float, parts[1:4]))
						self.vertices.append(vertex)
					elif line.startswith("f"):
						parts = line.strip().split()
						face = []
						for part in parts[1:]:
							indices = part.replace('//', '/').split('/')
							face.append(int(indices[0]) - 1)  # OBJ indices start at 1
						self.faces.append(tuple(face))
		except IOError:
			print(".obj file not found.")
		self.vertices = np.array(self.vertices)
		self.faces = np.array(self.faces)
		self.transformed_vertices = np.array(self.vertices)
		self.transformed_faces = np.array(self.faces)

	def transform(self, R, c, t):
		self.transformed_vertices = (c * R @ self.vertices.T + t).T

if __name__ == "__main__":
	VIDEO_ROOT = "/data/celebv-text/Videos/celebvtext_6"
	OUTPUT_ROOT = "/data/celebv-text/head_orientations"
	BOUNDBOX_ROOT = "/data/celebv-text/boundbox_mediapipe"
	SHARD_ROOT = "/data/celebv-text/splitting"
	LOG_ROOT = "/data/celebv-text/head_orientations/runlog"

	MEDIAPIPE_MAPPING_PATH = "/code/dataset_processing/models/mediapipe_geometries/mediapipe_emantic_mapping.json"
	MEDIAPIPE_CANONICAL_FACE_PATH = "/code/dataset_processing/models/mediapipe_geometries/mediapipe_canonical_face_mesh.obj"
	
	os.makedirs(OUTPUT_ROOT, exist_ok=True)
	os.makedirs(LOG_ROOT, exist_ok=True)

	# inputs
	parser = argparse.ArgumentParser()
	parser.add_argument("--shard_id", type=str, required=True)
	args = parser.parse_args()
	
	shard_id = args.shard_id

	# load shard
	shard_path = os.path.join(SHARD_ROOT, "video_split_{}.pkl".format(shard_id))
	with open(shard_path, 'rb') as file:
		filenames = pkl.load(file)
	# runlog
	runlog_path = os.path.join(LOG_ROOT, "runlog_{}.json".format(shard_id))

	# Set up mediapipe
	mp_face_mesh = mp.solutions.face_mesh
	with open(MEDIAPIPE_MAPPING_PATH, "r") as f:
		mapping = json.load(f)
	staticLandmarkIndices = mapping["nose"]["dorsum"] + mapping["nose"]["tipLower"] + mapping["additional_anchors"]
	keypointIndices = (
		mapping["nose"]["dorsum"] + mapping["nose"]["tipLower"] + mapping["additional_anchors"] +
		mapping["brow"]["rightLower"] + mapping["brow"]["rightUpper"] +
		mapping["brow"]["leftUpper"] + mapping["brow"]["leftLower"] +
		mapping["eye"]["right"] + mapping["eye"]["left"] +
		mapping["lips"]["inner"] + mapping["lips"]["outer"]
	)
	# Import the canonical face as a plane of reference. 
	face = ObjLoader(MEDIAPIPE_CANONICAL_FACE_PATH)

	runlog = []
	for i in range(0, len(filenames)):
		# try:
		video_name = filenames[i][0]  # this is a list because of legacy code issues
		# check and see if the output video file exists
		if os.path.exists(os.path.join(OUTPUT_ROOT, f"{video_name}.pkl")):
			print(f"Output video file {video_name}.pkl exists")
			found = False
			# load the runlog json
			if os.path.exists(runlog_path):
				with open(runlog_path, "r") as f:
					prev_runlog = json.load(f)
				# find the entry for this video
				for entry in prev_runlog:
					if entry["video_name"] == video_name:
						runlog.append(entry)
						found = True
						break	
				# if there is no log, re-run it still
				if found:
					continue
				else:
					print("video exists but log does not, re-running")
					pass
			else:
				print("video exists but log does not, re-running")
				pass

		# Output storing
		raw_landmark_output = []
		log_i = {"video_name": video_name, "error_too_many_missing_frames": False,"error_missing_landmark_detection": False, "error_cant_open_video": False, "error_cant_open_boundbox": False, "error_unknown": False}
		# Load the video
		cap = cv.VideoCapture(os.path.join(VIDEO_ROOT, f"{video_name}.mp4"))
		# Load the bounding boxes (x, y, w, h)
		with open(os.path.join(BOUNDBOX_ROOT, f"{video_name}.pickle"), "rb") as f:
			boundbox_list = pkl.load(f)["processed_bbox_frames"]
		with mp_face_mesh.FaceMesh(
			static_image_mode=False,
			max_num_faces=10,  # Increase max_num_faces to detect multiple faces
			min_detection_confidence=0.3,
			refine_landmarks=True) as face_mesh:
			# Set progress tracking before iterating through all frames
			pbar = tqdm(total=int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
			boundbox_counter = 0
			while cap.isOpened():
				# Get image
				ret, image = cap.read()
				if not ret:
					break
				
				if boundbox_counter >= len(boundbox_list):
					break

				# Get the bounding box
				boundbox = boundbox_list[boundbox_counter]
				x, y, w, h = boundbox

				# Crop the image
				scale = 1.2
				# Convert bounding box from (x, y, w, h) to (xc, yc, w/2, h/2)
				boundbox_center = (
					x + w // 2,
					y + h // 2,
					w // 2,
					h // 2
				)
				# Scale the bounding box
				boundbox_scaled = (
					int(boundbox_center[0]),
					int(boundbox_center[1]),
					int(round(boundbox_center[2] * scale)),
					int(round(boundbox_center[3] * scale))
				)
				x_min = max(boundbox_scaled[0] - boundbox_scaled[2], 0)
				x_max = min(boundbox_scaled[0] + boundbox_scaled[2], image.shape[1])
				y_min = max(boundbox_scaled[1] - boundbox_scaled[3], 0)
				y_max = min(boundbox_scaled[1] + boundbox_scaled[3], image.shape[0])

				cropped_image = image[y_min:y_max, x_min:x_max]
				# cropped_image = image
				# Convert the BGR image to RGB before processing.
				results = face_mesh.process(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))
				# Process face mesh landmarks on the image.
				if not results.multi_face_landmarks:
					log_i["error_missing_landmark_detection"] = True
					raw_landmark_output.append(None)
					pbar.update(1)
					boundbox_counter += 1
					continue

				image_height, image_width = cropped_image.shape[:2]
				input_bbox = (0, 0, image_width, image_height)  # Since the cropped image is based on the bounding box

				# Compute IoU for each detected face and select the one with the largest IoU
				max_iou = 0
				selected_landmarks = None
				for face_landmarks in results.multi_face_landmarks:
					# Compute face bounding box
					face_bbox = compute_bounding_box(face_landmarks.landmark, image_width, image_height)
					# Adjust face_bbox coordinates relative to the original image
					face_bbox_in_original = (
						x_min + face_bbox[0],
						y_min + face_bbox[1],
						face_bbox[2],
						face_bbox[3]
					)
					# Calculate IoU with the input bounding box
					# print(face_bbox, face_bbox_in_original)
					iou = calculate_iou(boundbox, face_bbox_in_original)
					if iou > max_iou:
						max_iou = iou
						selected_landmarks = face_landmarks

				if selected_landmarks is None:
					# If no face matches, skip the frame
					log_i["error_missing_landmark_detection"] = True
					raw_landmark_output.append(None)
					pbar.update(1)
					boundbox_counter += 1
					continue

				# Process the selected face landmarks
				land_mark_matrix_pts = np.zeros((478, 3))
				for idx in range(len(selected_landmarks.landmark)):
					land_mark_matrix_pts[idx, 0] = selected_landmarks.landmark[idx].x
					land_mark_matrix_pts[idx, 1] = selected_landmarks.landmark[idx].y
					land_mark_matrix_pts[idx, 2] = selected_landmarks.landmark[idx].z
				raw_landmark_output.append(land_mark_matrix_pts)
				pbar.update(1)
				boundbox_counter += 1

			# if there are no valid frames, set the error flag, and just continue
			if boundbox_counter == 0:
				log_i["error_cant_open_video"] = True
				pbar.close()
				cap.release()
				raw_landmark_output = np.array(raw_landmark_output)
				runlog.append(log_i)
				with open(runlog_path, "w") as f:
					json.dump(runlog, f)

			pbar.close()
			cap.release()
			# check the number of Nones:
			num_nones = len([x for x in raw_landmark_output if x is None])
			if num_nones >= len(raw_landmark_output) // 2:
				log_i["error_too_many_missing_frames"] = True
				print(f"Too many missing frames in video {video_name}, missing {num_nones} out of {len(raw_landmark_output)}")
				runlog.append(log_i)
				with open(runlog_path, "w") as f:
					json.dump(runlog, f)
				continue

			interpolated_landmarks, detailed_error_log_i = interpolate_landmarks(raw_landmark_output)

			raw_landmark_output = np.array(raw_landmark_output)



		# Smooth landmarks if needed
		landmark_norm = np.linalg.norm(raw_landmark_output, axis=2).sum(axis=1)
		smoothed_landmark_output = np.copy(raw_landmark_output)
		for i in range(1, raw_landmark_output.shape[0]):
			if landmark_norm[i] == 0:
				smoothed_landmark_output[i] = smoothed_landmark_output[i-1]

		# Align landmarks to neutral pose and get rotation and translation
		___, R_matrices, t_vectors = rotateToNeutral(
			face.vertices, smoothed_landmark_output, staticLandmarkIndices, returnRotation=True
		)
		rotation_matrices = R_matrices
		translation_vectors = t_vectors
		# apply smoothing to the rotation matrices
		rotation_matrices = smooth_rotation_matrices(rotation_matrices, window_length=5, polyorder=2)
		# normalize the angles to make forward (0, 0, 0)
		R_adjust = Rotation.from_euler('X', 180, degrees=True).as_matrix()
		adjusted_rotation_matrices = []
		for R_current in rotation_matrices:
			R_adj = R_adjust @ R_current
			adjusted_rotation_matrices.append(R_adj)
		rotation_matrices = adjusted_rotation_matrices

		# collect yaw pitch roll angles
		yall_pitch_roll = []
		for i in range(0, len(rotation_matrices)):
			R = rotation_matrices[i]
			rotations = Rotation.from_matrix(R)
			yaw, pitch, roll = rotations.as_euler('YXZ', degrees=True)  # Use radians for internal calculations
			# Adjust the yaw angle
			# pitch = -pitch  # Flip the yaw angle
			roll = -roll  # Flip the roll angle
			# store it 
			yall_pitch_roll.append([yaw, pitch, roll])
			rotations_modified = Rotation.from_euler('YXZ', [yaw, pitch, roll], degrees=True)
			R_modified = rotations_modified.as_matrix()
			rotation_matrices[i] = R_modified
		for i in range(0, 8):
			print(i, yall_pitch_roll[i])
		# Proceed to visualize without projection

		# Open the video again for writing the output video with arrows
		cap = cv.VideoCapture(os.path.join(VIDEO_ROOT, f"{video_name}.mp4"))
		out_video_path = os.path.join(OUTPUT_ROOT, f"{video_name}.mp4")
		fourcc = cv.VideoWriter_fourcc(*'mp4v')
		fps = cap.get(cv.CAP_PROP_FPS)
		frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
		out = cv.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

		pbar = tqdm(total=int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
		frame_idx = 0
		boundbox_counter = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret or frame_idx >= len(rotation_matrices):
				break

			if boundbox_counter >= len(boundbox_list):
				break

			R = rotation_matrices[frame_idx]
			t = translation_vectors[frame_idx]

			if R is not None:
				# Get the bounding box
				boundbox = boundbox_list[boundbox_counter]
				x, y, w, h = boundbox

				# Compute the center of the bounding box
				center_x = x + w // 2
				center_y = y + h // 2

				# Define 3D axes
				axis_length = 200  # Adjust length as needed
				axes_3D = np.float32([
					[0, 0, 0],
					[axis_length, 0, 0],  # X-axis
					[0, axis_length, 0],  # Y-axis
					[0, 0, axis_length]   # Z-axis
				])

				# Rotate the axes using the rotation matrix
				rotated_axes = R @ axes_3D.T
				# Correct the signs to match the viewer's perspective
				# Invert the Z-axis to flip the yaw direction
				# rotated_axes[2, :] *= -1  # Invert Z-axis to flip yaw

				# Project the rotated axes to 2D by ignoring the Z-coordinate (orthographic projection)
				projected_axes = rotated_axes[:2, :].T  # Shape: (4, 2)

				# Shift the axes to the center of the bounding box
				projected_axes[:, 0] += center_x
				projected_axes[:, 1] += center_y

				# Convert coordinates to integers
				projected_axes = projected_axes.astype(int)

				# Draw the arrows on the frame
				origin = tuple(projected_axes[0])
				cv.arrowedLine(frame, origin, tuple(projected_axes[1]), (0, 0, 255), 2, tipLength=0.2)  # X-axis in red
				cv.arrowedLine(frame, origin, tuple(projected_axes[2]), (0, 255, 0), 2, tipLength=0.2)  # Y-axis in green
				cv.arrowedLine(frame, origin, tuple(projected_axes[3]), (255, 0, 0), 2, tipLength=0.2)  # Z-axis in blue

				# Display the yaw, pitch, and roll angles
				rotations = Rotation.from_matrix(R)
				# Adjust the axes order and signs to match the correct conventions
				yaw, pitch, roll = rotations.as_euler('YXZ', degrees=True)

				text = f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}"
				x = int(round(x))
				y = int(round(y))
				cv.putText(frame, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

			out.write(frame)
			pbar.update(1)
			frame_idx += 1
			boundbox_counter += 1

		cap.release()
		out.release()
		pbar.close()

		# this must be done after the video is written, because the video is written to in the loop, so if there is an error, the video will be incomplete
		# but then the error will be logged and the video will still be skipped 
		yall_pitch_roll = np.array(yall_pitch_roll)
		with open(os.path.join(OUTPUT_ROOT, f"{video_name}.pkl"), "wb") as f:
			pkl.dump(yall_pitch_roll, f)
		log_i["error_details"] = detailed_error_log_i
		runlog.append(log_i)
		with open(runlog_path, "w") as f:
			json.dump(runlog, f)
