import os
import h5py
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import av

AVALOHA_DATASET_PATH = "data/av_aloha/dataset"
OUTPUT_HDF5_PATH = "data/av_aloha/hdf5"

TASK_NAMES = [
    "av_aloha_sim_cube_transfer",
    "av_aloha_sim_hook_package",
    "av_aloha_sim_peg_insertion",
    "av_aloha_sim_pour_test_tube",
    "av_aloha_sim_slot_insertion",
    "av_aloha_sim_thread_needle",
]


def load_episode_parquet(dataset_path: Path, episode_idx: int):
    """读取单个episode的parquet数据"""
    chunk_idx = episode_idx // 1000
    parquet_path = dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
    df = pd.read_parquet(parquet_path)
    return df


def load_video_frames(video_path: Path) -> np.ndarray:
    """读取mp4视频的所有帧"""
    frames = []
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)
    container.close()
    return np.array(frames)


def images_encoding(imgs):
    """将图像序列编码为JPEG格式"""
    encode_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    padded_data = []
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return padded_data, max_len


def convert_gaze_to_pixel(left_eye, right_eye, img_height=480, img_width=640):
    """
    将归一化眼动坐标转换为像素坐标
    AVAloha: left_eye/right_eye 是 [x, y] 归一化坐标 (-1到1范围)
    转换公式: pixel = (normalized + 1) / 2 * size
    输出: gaze [y, x] 像素坐标
    """
    avg_x = (left_eye[0] + right_eye[0]) / 2
    avg_y = (left_eye[1] + right_eye[1]) / 2
    pixel_x = (avg_x + 1) / 2 * img_width
    pixel_y = (avg_y + 1) / 2 * img_height
    return np.array([pixel_y, pixel_x])


def data_transform(dataset_path: Path, save_path: Path, episode_num: int):
    """转换单个任务的所有episode"""
    
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    with open(tasks_path, 'r') as f:
        task_info = json.loads(f.readline())
    task_description = task_info.get('task', 'No task description')
    
    for i in tqdm(range(episode_num), desc=f"Converting {dataset_path.name}"):
        episode_dir = save_path / f"episode_{i}"
        episode_dir.mkdir(exist_ok=True)
        
        instructions = {"instructions": [task_description]}
        with open(episode_dir / "instructions.json", "w") as f:
            json.dump(instructions, f, indent=2)
        
        df = load_episode_parquet(dataset_path, i)
        chunk_idx = i // 1000
        video_base = dataset_path / "videos" / f"chunk-{chunk_idx:03d}"
        cam_high = load_video_frames(video_base / "observation.images.zed_cam_left" / f"episode_{i:06d}.mp4")
        cam_left = load_video_frames(video_base / "observation.images.wrist_cam_left" / f"episode_{i:06d}.mp4")
        cam_right = load_video_frames(video_base / "observation.images.wrist_cam_right" / f"episode_{i:06d}.mp4")
        num_frames = len(df)
        states = np.array([df['observation.state'].iloc[j] for j in range(num_frames)], dtype=np.float32)
        
        gazes = []
        for j in range(num_frames):
            left_eye = np.array(df['left_eye'].iloc[j])
            right_eye = np.array(df['right_eye'].iloc[j])
            gaze = convert_gaze_to_pixel(left_eye, right_eye)
            gazes.append(gaze)
        gazes = np.array(gazes, dtype=np.float32)
        qpos = states[:-1]
        action = states[1:]
        gaze = gazes[:-1]
        cam_high = cam_high[:-1]
        cam_left = cam_left[:-1]
        cam_right = cam_right[:-1]
        cam_high_enc, len_high = images_encoding(cam_high)
        cam_left_enc, len_left = images_encoding(cam_left)
        cam_right_enc, len_right = images_encoding(cam_right)
        
        hdf5_path = episode_dir / f"episode_{i}.hdf5"
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("action", data=action)
            f.create_dataset("gaze", data=gaze)
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=qpos)
            images = obs.create_group("images")
            images.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            images.create_dataset("cam_left_wrist", data=cam_left_enc, dtype=f"S{len_left}")
            images.create_dataset("cam_right_wrist", data=cam_right_enc, dtype=f"S{len_right}")


if __name__ == "__main__":
    for task_name in TASK_NAMES:
        load_dir = Path(AVALOHA_DATASET_PATH) / task_name
        target_dir = Path(OUTPUT_HDF5_PATH) / task_name
        
        if not load_dir.exists():
            print(f"Skipping {task_name}: directory not found")
            continue
        
        episodes_path = load_dir / "meta" / "episodes.jsonl"
        episode_num = sum(1 for _ in open(episodes_path))
        
        print(f'Processing: {task_name}, episodes: {episode_num}')
        data_transform(load_dir, target_dir, episode_num)
    
    print("All tasks completed!")

