import argparse
import logging
import os
import json
import glob
import gc
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import tqdm
import tensorflow_datasets as tfds
from tensorflow_datasets.core.file_adapters import FileFormat
from tensorflow_datasets.rlds import rlds_base

# 禁用GCS和CUDA
os.environ["NO_GCE_CHECK"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
tfds.core.utils.gcs_utils._is_gcs_disabled = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", 
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    "head_pan", "head_tilt", "head_roll", "head_4", "head_5", "head_6", "head_7",
]

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

AVALOHA_DATASET_PATH = "/data/lichy/root/Datasets/LAF"

ORIGIN_IMAGE_SIZE = (480, 640)
TARGET_IMAGE_SIZE = (224, 224)


def image_resize_with_pad(image, target_size=TARGET_IMAGE_SIZE):
    """图像缩放并保持宽高比，使用黑色padding"""
    batch_size, h, w, c = image.shape
    target_h, target_w = target_size
    
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized_images = []
    for i in range(batch_size):
        resized = cv2.resize(image[i], (new_w, new_h))
        resized_images.append(resized)
    
    result = np.zeros((batch_size, target_h, target_w, c), dtype=image.dtype)
    
    start_h = (target_h - new_h) // 2
    start_w = (target_w - new_w) // 2
    
    for i in range(batch_size):
        result[i, start_h:start_h+new_h, start_w:start_w+new_w] = resized_images[i]
    
    return result


def gaze_resize_with_pad(gaze_pixel, origin_size=ORIGIN_IMAGE_SIZE, target_size=TARGET_IMAGE_SIZE):
    """同步调整gaze坐标"""
    gaze_pixel = np.array(gaze_pixel)
    origin_h, origin_w = origin_size
    target_h, target_w = target_size
    
    scale = min(target_h / origin_h, target_w / origin_w)
    new_h, new_w = int(origin_h * scale), int(origin_w * scale)
    
    start_h = (target_h - new_h) // 2
    start_w = (target_w - new_w) // 2
    
    scaled_gaze = gaze_pixel * scale
    transformed_gaze = scaled_gaze + np.array([start_h, start_w])
    return transformed_gaze.astype(np.int64)


def generate_rlds_config(encoding_format="jpeg", **kwargs):
    """生成RLDS数据集配置"""
    
    action_info = {
        "action": tfds.features.Tensor(
            shape=(len(MOTORS),), 
            dtype=np.float32, 
            doc="Robot joint action"
        ),
        "action_mask": tfds.features.Tensor(
            shape=(1,), 
            dtype=np.int64, 
            doc="Action mask (1: valid action, 0: invalid action)"
        ),
        "gaze": tfds.features.Tensor(
            shape=(2,), 
            dtype=np.int64, 
            doc="Gaze coordinates [h, w]"
        ),
        "gaze_mask": tfds.features.Tensor(
            shape=(1,), 
            dtype=np.int64, 
            doc="Gaze mask (1: valid gaze, 0: invalid gaze)"
        )
    }
    
    observation_info = {}
    
    for cam in CAMERAS:
        observation_info[cam] = tfds.features.Image(
            shape=(224, 224, 3),
            dtype=np.uint8,
            encoding_format=encoding_format,
            doc=f"Camera {cam}"
        )
    
    observation_info["state"] = tfds.features.Tensor(
        shape=(len(MOTORS),),
        dtype=np.float32,
        doc=MOTORS
    )
    
    return dict(
        observation_info=observation_info,
        action_info=action_info,
        step_metadata_info={
            "language_instruction": tfds.features.Text(),
        },
        citation=kwargs.get("citation", ""),
        homepage=kwargs.get("homepage", ""),
        overall_description=kwargs.get("overall_description", ""),
        description=kwargs.get("description", ""),
    )


def load_hdf5_episode(ep_path: Path):
    """加载单个HDF5 episode"""
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        gaze = torch.from_numpy(ep["/gaze"][:])
        
        imgs_per_cam = {}
        for camera in CAMERAS:
            if f"/observations/images/{camera}" in ep:
                imgs_array = []
                for data in ep[f"/observations/images/{camera}"]:
                    data = np.frombuffer(data, np.uint8)
                    imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
                imgs_array = np.array(imgs_array)
                # 图像resize
                imgs_per_cam[camera] = image_resize_with_pad(imgs_array, TARGET_IMAGE_SIZE)
    
    # gaze坐标resize
    gaze = gaze_resize_with_pad(gaze.numpy())
    gaze = torch.from_numpy(gaze)
    
    dir_path = os.path.dirname(ep_path)
    json_path = f"{dir_path}/instructions.json"
    
    instruction = "No instruction available"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            instruction_dict = json.load(f)
            instructions = instruction_dict['instructions']
            instruction = np.random.choice(instructions)
    
    return {
        'state': state,
        'action': action,
        'gaze': gaze,
        'images': imgs_per_cam,
        'instruction': instruction,
        'num_frames': state.shape[0]
    }


def process_episode_to_rlds(episode_data):
    """转换episode数据为RLDS格式"""
    episode_steps = []
    
    for i in range(episode_data['num_frames']):
        observation = {
            "state": episode_data['state'][i].numpy().astype(np.float32)
        }
        
        for camera, img_array in episode_data['images'].items():
            img = img_array[i]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            observation[camera] = img
        
        action = {
            "action": episode_data['action'][i].numpy().astype(np.float32),
            "action_mask": np.array([1], dtype=np.int64),
            "gaze": episode_data['gaze'][i].numpy().astype(np.int64),
            "gaze_mask": np.array([1], dtype=np.int64)
        }
        
        step = {
            "observation": observation,
            "action": action,
            "language_instruction": episode_data['instruction'],
            "is_first": i == 0,
            "is_last": i == episode_data['num_frames'] - 1,
            "is_terminal": i == episode_data['num_frames'] - 1,
        }
        
        episode_steps.append(step)
    
    return episode_steps


class DirectHDF5ToRLDSBuilder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    """TFDS builder for HDF5 to RLDS conversion"""
    
    def __init__(self, raw_dir, name, dataset_config, hdf5_files, *, file_format=None, **kwargs):
        self.name = name
        self.VERSION = kwargs["version"]
        self.raw_dir = raw_dir
        self.dataset_config = dataset_config
        self.hdf5_files = hdf5_files
        self.__module__ = "hdf5_to_rlds_direct"
        super().__init__(file_format=file_format, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return rlds_base.build_info(
            rlds_base.DatasetConfig(name=self.name, **self.dataset_config),
            self,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        dl_manager._download_dir.rmtree(missing_ok=True)
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        def _generate_examples_regular():
            for episode_index, ep_path in enumerate(tqdm.tqdm(self.hdf5_files, desc="Processing episodes")):
                logging.info(f"Processing episode {episode_index}: {ep_path}")
                
                try:
                    episode_data = load_hdf5_episode(Path(ep_path))
                    episode_steps = process_episode_to_rlds(episode_data)
                    del episode_data
                    gc.collect()
                    yield f"{episode_index}", {"steps": episode_steps}
                except Exception as e:
                    logging.error(f"Error processing episode {episode_index}: {e}")
                    yield f"{episode_index}", {"steps": []}

        return _generate_examples_regular()


def collect_hdf5_files(raw_dir: Path, task_list: str):
    """收集指定任务的所有HDF5文件"""
    hdf5_files = []
    
    for task_name in task_list:
        dir_path = raw_dir / task_name
        if dir_path.exists():
            pattern = str(dir_path / "**" / "*.hdf5")
            files = glob.glob(pattern, recursive=True)
            hdf5_files.extend(sorted(files))
            logging.info(f"Found {len(files)} files in {dir_path}")
        else:
            logging.warning(f"Directory {dir_path} does not exist")
    
    logging.info(f"Total HDF5 files to process: {len(hdf5_files)}")
    return hdf5_files


def main(raw_dir, output_dir, task_list, version, encoding_format, split_ratio=0.9, **kwargs):
    """主转换函数"""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    
    hdf5_files = collect_hdf5_files(raw_dir, task_list)
    dataset_config = generate_rlds_config(encoding_format, **kwargs)
    
    random.shuffle(hdf5_files)
    split_idx = int(len(hdf5_files) * split_ratio)
    train_hdf5_files = hdf5_files[:split_idx]
    test_hdf5_files = hdf5_files[split_idx:]
    logging.info(f"Split: {len(train_hdf5_files)} train, {len(test_hdf5_files)} test")
    
    split_info = {
        "task_name": "all",
        "split_ratio": split_ratio,
        "total_files": len(hdf5_files),
        "train_files": train_hdf5_files,
        "test_files": test_hdf5_files
    }
    
    task_output_dir = output_dir / "all"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    split_json_path = task_output_dir / "split.json"
    with open(split_json_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    logging.info(f"Split info saved to: {split_json_path}")
    
    # # train
    # train_builder = DirectHDF5ToRLDSBuilder(
    #     raw_dir=raw_dir,
    #     name=f"train",
    #     data_dir=task_output_dir,
    #     version=version,
    #     dataset_config=dataset_config,
    #     hdf5_files=train_hdf5_files,
    #     file_format=FileFormat.TFRECORD,
    # )
    # train_builder.download_and_prepare(
    #     download_config=tfds.download.DownloadConfig(
    #         try_download_gcs=False,
    #         verify_ssl=False,
    #         beam_options=None,
    #         beam_runner=None,
    #     ),
    # )
    # logging.info(f"Train dataset successfully created at: {task_output_dir}")
    
    # # test
    # test_builder = DirectHDF5ToRLDSBuilder(
    #     raw_dir=raw_dir,
    #     name=f"test",
    #     data_dir=task_output_dir,
    #     version=version,
    #     dataset_config=dataset_config,
    #     hdf5_files=test_hdf5_files,
    #     file_format=FileFormat.TFRECORD,
    # )
    # test_builder.download_and_prepare(
    #     download_config=tfds.download.DownloadConfig(
    #         try_download_gcs=False,
    #         verify_ssl=False,
    #         beam_options=None,
    #         beam_runner=None,
    #     ),
    # )
    # logging.info(f"Test dataset successfully created at: {task_output_dir}")

    # all
    all_builder = DirectHDF5ToRLDSBuilder(
        raw_dir=raw_dir,
        name=f"all",
        data_dir=task_output_dir,
        version=version,
        dataset_config=dataset_config,
        hdf5_files=hdf5_files,
        file_format=FileFormat.TFRECORD,
    )
    all_builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            try_download_gcs=False,
            verify_ssl=False,
            beam_options=None,
            beam_runner=None,
        ),
    )
    logging.info(f"All dataset successfully created at: {task_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDF5 to RLDS conversion for AVAloha datasets")
    
    parser.add_argument("--raw-dir", type=Path, default="/data/lichy/root/Datasets/LAF/hdf5",
                        help="Path to the raw HDF5 dataset directory")
    parser.add_argument("--output-dir", type=Path, default="/data/lichy/root/Datasets/LAF/rlds",
                        help="Path to the output RLDS directory")
    parser.add_argument("--encoding-format", type=str, choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--version", type=str, default="1.0.0", help="Dataset version (x.y.z)")
    parser.add_argument("--split-ratio", type=float, default=0.9, help="Train/test split ratio (default: 0.9)")
    
    args = parser.parse_args()
    
    task_list = [
        "av_aloha_sim_cube_transfer",
        "av_aloha_sim_hook_package",
        "av_aloha_sim_peg_insertion",
        "av_aloha_sim_pour_test_tube",
        "av_aloha_sim_slot_insertion",
        "av_aloha_sim_thread_needle",
    ]
    
    main(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        task_list=task_list,
        version=args.version,
        encoding_format=args.encoding_format,
        split_ratio=args.split_ratio,
    )
    
    print("All tasks completed!")

