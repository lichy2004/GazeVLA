"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from enum import Enum
from enum import auto
import json
import logging
from pathlib import Path

import tqdm

import openpi.shared.download as download
from openpi.utils import transforms as tfs


def matrix_to_rotation_6d(matrix):
    """Convert rotation matrix to 6D rotation representation (TensorFlow version).
    
    Args:
        matrix: Rotation matrix with shape (..., 3, 3)
    
    Returns:
        6D rotation representation with shape (..., 6)
    """
    import tensorflow as tf
    
    batch_shape = tf.shape(matrix)[:-2]
    rotation_6d = tf.reshape(matrix[..., :2, :], tf.concat([batch_shape, [6]], axis=0))
    
    return rotation_6d


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        filter_dict_path=None,  # Path to json file with indices to sample during training
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )

        # # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        # Load the filter dictionary if provided.
        # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
        # (e.g.,
        # {
        #     "<episode key>": [[0, 100], [200, 300]]
        # }
        # means keep frames 0-99 and 200-299).
        if filter_dict_path is not None:
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)

            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []

            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
            )
            logging.info("Filter hash table initialized")
        else:
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
            )

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    (
                        traj["action_dict"]["joint_position"]
                        if action_space == DroidActionSpace.JOINT_POSITION
                        else traj["action_dict"]["joint_velocity"]
                    ),
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            exterior_img = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            # Randomly sample one of the three language instructions
            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]

            traj_len = tf.shape(traj["action"])[0]
            indices = tf.as_string(tf.range(traj_len))

            # Data filtering:
            # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
            # and each step's time step index. This will index into the filter hash table, and if it returns true,
            # then the frame passes the filter.
            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + indices
            )
            passes_filter = self.filter_table.lookup(step_id)

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
                "step_id": step_id,
                "passes_filter": passes_filter,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Filter data that doesn't pass the filter
        def filter_from_dict(frame):
            return frame["passes_filter"]

        dataset = dataset.filter(filter_from_dict)

        # Remove "passes_filter" key from output
        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(remove_passes_filter)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            return traj

        dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000



class RobotwinRldsDataset:
    """RLDS data loader designed for RobotWin dataset."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        action_chunk_size: int = 50,
        down_sample: int = 1,
        shuffle_buffer_size: int = 1_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        filter_dict_path=None,
        dataset_name: str = "robotwin_full_hard",
        dataset_version: str = "1.0.0",
    ):
        """RLDS data loader for RobotWin dataset.
        
        Args:
            data_dir: Path to RobotWin RLDS data directory
            batch_size: Batch size
            action_chunk_size: Action sequence length (default: 50)
            dataset_name: RobotWin dataset name
        """
        # Import tensorflow here to not make it mandatory
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
        
        # Configure Tensorflow with *no GPU devices*
        tf.config.set_visible_devices([], "GPU")
        
        logging.info(f"Loading RobotWin RLDS dataset: {dataset_name} from {data_dir}")
        try:
            print(f"Loading RobotWin RLDS dataset: {dataset_name} from {data_dir}")
            builder = tfds.builder(dataset_name, data_dir=data_dir, version=dataset_version)
            dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)
            print(f"Finish loading RobotWin RLDS dataset: {dataset_name} from {data_dir}")
        except Exception as e:
            logging.error(f"Failed to load robotwin dataset {dataset_name}: {e}")
            raise RuntimeError(f"Failed to load RobotWin RLDS dataset from {data_dir}.") from e
        
        dataset = dataset.repeat()
        
        self.use_filter = filter_dict_path is not None
        if self.use_filter:
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)
            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")
        
        def restructure_robotwin(traj):
            """Restructure RobotWin trajectory data."""
            
            # RobotWin data: Extract action components from action dict
            action_dict = traj["action"]
            actions = action_dict["action"]           # [T, 14]
            action_mask = action_dict["action_mask"]  # [T, 1]
            gaze = action_dict["gaze"]                # [T, 2]
            gaze_mask = action_dict["gaze_mask"]      # [T, 1]
            
            observation = traj["observation"]
            cam_high = observation["cam_high"]
            cam_left_wrist = observation["cam_left_wrist"]
            cam_right_wrist = observation["cam_right_wrist"]
            robot_state = observation["state"]
            language_instruction = traj["language_instruction"]
            
            traj_len = tf.shape(actions)[0]
            passes_filter = tf.ones([traj_len], dtype=tf.bool)
            
            return {
                "actions": actions,
                "observation": {
                    "image": cam_high,
                    "wrist_image": cam_left_wrist,
                    "wrist_image_right": cam_right_wrist,
                    "state": robot_state,
                    "gaze": gaze,
                    "action_mask": action_mask,
                    "gaze_mask": gaze_mask,
                },
                "prompt": language_instruction,
                "passes_filter": passes_filter,
            }
        
        dataset = dataset.traj_map(restructure_robotwin, num_parallel_calls)
        
        def chunk_actions_robotwin(traj):
            """Create action chunks for RobotWin data."""
            traj_len = tf.shape(traj["actions"])[0]
            
            action_offsets = tf.range(action_chunk_size) * down_sample
            action_chunk_indices = tf.broadcast_to(
                action_offsets[None, :],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )
            
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            
            # Chunk action_mask to match actions dimensions
            action_mask = traj["observation"]["action_mask"]  # [T, 1]
            action_mask_chunked = tf.gather(action_mask, action_chunk_indices)  # [T, chunk_size, 1]
            traj["observation"]["action_mask"] = action_mask_chunked
            
            return traj
        
        dataset = dataset.traj_map(chunk_actions_robotwin, num_parallel_calls)
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(lambda frame: frame["passes_filter"])
        dataset = dataset.map(lambda frame: {k: v for k, v in frame.items() if k != "passes_filter"})
        
        def decode_robotwin_images(traj):
            """Decode RobotWin image data."""
            obs = traj["observation"]
            obs["image"] = tf.io.decode_image(obs["image"], expand_animations=False, dtype=tf.uint8)
            obs["wrist_image"] = tf.io.decode_image(obs["wrist_image"], expand_animations=False, dtype=tf.uint8)
            obs["wrist_image_right"] = tf.io.decode_image(obs["wrist_image_right"], expand_animations=False, dtype=tf.uint8)
            return traj
        
        dataset = dataset.frame_map(decode_robotwin_images, num_parallel_calls)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.with_ram_budget(1)
        
        self.dataset = dataset
        
        logging.info(f"RobotWin RLDS dataset initialized: batch_size={batch_size}, action_chunk_size={action_chunk_size}")
        
    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()
        
    def __len__(self):
        return 100_000

TRANSFORM_JOINTS = [
    'camera',
    'left_wrist', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_little',
    'right_wrist', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little',
]
class HumanRldsDataset:
    """RLDS data loader for Human dataset with hand (48-dim) and gaze (2-dim) actions.
    
    Supports loading and mixing multiple dataset directories.
    """
    
    def __init__(
        self,
        data_dir: str | list[str],
        batch_size: int,
        *,
        shuffle: bool = True,
        action_chunk_size: int = 30,
        down_sample: int = 1,
        shuffle_buffer_size: int = 200,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        filter_dict_path=None,
        dataset_name: str = "Human",
        dataset_version: str = "1.0.0",
        dataset_weights: list[float] | None = None,
    ):
        """RLDS data loader for Human dataset.
        
        Args:
            data_dir: Path to Human RLDS data directory (or list of paths)
            batch_size: Batch size
            action_chunk_size: Action sequence length
            dataset_name: Human dataset name
        """
        # Import tensorflow here to not make it mandatory
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
        
        # Configure Tensorflow with *no GPU devices*
        tf.config.set_visible_devices([], "GPU")
        
        if isinstance(data_dir, str):
            data_dirs = [data_dir]
        else:
            data_dirs = data_dir
        
        datasets = []
        for idx, single_data_dir in enumerate(data_dirs):
            logging.info(f"Loading Human RLDS dataset {idx+1}/{len(data_dirs)}: {dataset_name} from {single_data_dir}")
            print(f"Loading Human RLDS dataset {idx+1}/{len(data_dirs)}: {dataset_name} from {single_data_dir}")
            builder = tfds.builder(dataset_name, data_dir=single_data_dir, version=dataset_version)
            single_dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)
            datasets.append(single_dataset)
            print(f"Finished loading dataset from {single_data_dir}")
        
        if not datasets:
            raise RuntimeError(f"Failed to load any Human RLDS dataset. Please check paths.")
        
        if dataset_weights is None:
            weights = [1.0] * len(datasets)
        else:
            weights = dataset_weights
        logging.info(f"Using dataset weights: {weights}")
        
        self.use_filter = filter_dict_path is not None
        if self.use_filter:
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)
            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")
        
        def restructure_human(traj):
            """Restructure human trajectory data with hand+gaze action format."""
            
            # Human data structure: hand [T, 48], gaze [T, 2], state [T, 48]
            hand = traj["action"]["hand"]
            gaze = traj["action"]["gaze"]
            action_mask = traj["action"]["hand_mask"]  # Read from RLDS data (field name unchanged)
            gaze_mask = traj["action"]["gaze_mask"]

            transforms = {}
            for joint_name in TRANSFORM_JOINTS:
                transform_key = f"transform_{joint_name}"
                transforms[joint_name] = traj["observation"][transform_key]  # [T, 4, 4]
            
            # Extract observation data
            cam_high = traj["observation"]["cam_high"]
            state = traj["observation"]["state"]

            # Language instruction
            instruction = traj["language_instruction"]
            
            traj_len = tf.shape(hand)[0]
            passes_filter = tf.ones([traj_len], dtype=tf.bool)

            return {
                "actions": hand,
                "observation": {
                    "transforms": transforms,
                    "image": cam_high,
                    "state": state,
                    "gaze": gaze,
                    "action_mask": action_mask,
                    "gaze_mask": gaze_mask,
                },
                "prompt": instruction,
                "passes_filter": passes_filter,
            }
        
        def chunk_actions_human(traj):
            # chunk indices
            traj_len = tf.shape(traj["actions"])[0]
            action_offsets = tf.range(action_chunk_size) * down_sample
            action_chunk_indices = tf.broadcast_to(
                action_offsets[None, :],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            transforms = traj["observation"]["transforms"]
            camera_transform = transforms['camera'][0:1, :, :]          # [1, 4, 4]
            camera_first_frame = camera_transform[:, None, :, :]        # [1, 1, 4, 4]
            camera_first_frame_inv = tf.linalg.inv(camera_first_frame)  # [1, 1, 4, 4]
            
            for joint_name in TRANSFORM_JOINTS:
                if 'camera' in joint_name:
                    continue
                joint_transform_chunked = tf.gather(transforms[joint_name], action_chunk_indices)       # [len, chunk, 4, 4]
                joint_transform_relative = tf.matmul(camera_first_frame_inv, joint_transform_chunked)   # [len, chunk, 4, 4]
                transforms[joint_name] = joint_transform_relative

            joint_features = []
            for joint_name in TRANSFORM_JOINTS:
                if 'camera' in joint_name:
                    continue
                joint_transform = transforms[joint_name]  # [len, chunk, 4, 4]
                position = joint_transform[..., :3, 3]
                joint_features.append(position)
                if 'wrist' in joint_name:
                    rotation_matrix = joint_transform[..., :3, :3]
                    rotation_6d = matrix_to_rotation_6d(rotation_matrix)
                    joint_features.append(rotation_6d)
            qpos = tf.concat(joint_features, axis=-1)   # [len, chunk, 48]
            actions = qpos             # [len, chunk, 48]
            state = qpos[:, 0]         # [len, 48]
            traj["actions"] = actions
            traj["observation"]["state"] = state
            
            action_mask = traj["observation"]["action_mask"]  # [len, 2]
            action_mask_chunked = tf.gather(action_mask, action_chunk_indices)  # [len, chunk, 2]
            traj["observation"]["action_mask"] = action_mask_chunked
            
            return traj
        
        def decode_images(traj):
            obs = traj["observation"]
            obs["image"] = tf.io.decode_image(obs["image"], expand_animations=False, dtype=tf.uint8)
            return traj
        
        def process_single_dataset(ds):
            ds = ds.traj_map(restructure_human, num_parallel_calls)
            ds = ds.traj_map(chunk_actions_human, num_parallel_calls)
            ds = ds.flatten(num_parallel_calls=num_parallel_calls)
            ds = ds.filter(lambda frame: frame["passes_filter"])
            ds = ds.map(lambda frame: {k: v for k, v in frame.items() if k != "passes_filter"})
            ds = ds.frame_map(decode_images, num_parallel_calls)
            return ds
        
        if len(datasets) == 1:
            dataset = process_single_dataset(datasets[0])
            dataset = dataset.repeat()
            logging.info(f"Using single dataset from {data_dirs[0]}")
            dataset = dataset.shuffle(shuffle_buffer_size)
            dataset = dataset.batch(batch_size)
            dataset = dataset.with_ram_budget(1)
        else:
            logging.info(f"Processing {len(datasets)} datasets before mixing...")
            prepared_datasets = []
            per_dataset_buffer = max(100, shuffle_buffer_size // (len(datasets) * 2))
            for idx, ds in enumerate(datasets):
                ds_processed = process_single_dataset(ds)
                ds_prepared = ds_processed.repeat().shuffle(per_dataset_buffer)
                ds_prepared = ds_prepared.with_ram_budget(1)
                prepared_datasets.append(ds_prepared)
                logging.info(f"  Processed dataset {idx+1}/{len(datasets)} from {data_dirs[idx]}")
            
            dataset = tf.data.Dataset.sample_from_datasets(prepared_datasets, weights=weights)
            logging.info(f"Merged {len(datasets)} datasets using sample_from_datasets with weights {weights}")
            final_shuffle_buffer = min(shuffle_buffer_size, 200)
            dataset = dataset.shuffle(final_shuffle_buffer)
            dataset = dataset.batch(batch_size)
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        logging.info(f"Human RLDS dataset initialized: batch_size={batch_size}, action_chunk_size={action_chunk_size}")
        
    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()
        
    def __len__(self):
        '''
            100_000     for 1 hour (30fps * 60s * 60min)
            1_000_000   for 10 hour
            20_000_000  for 200 hour (gaze)
            100_000_000 for 1000 hour (hand)

        '''
        # return 100_000_000
        return 1_000_000
