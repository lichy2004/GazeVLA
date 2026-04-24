# GazeVLA: Learning Human Intention for Robotic Manipulation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red?style=flat&logo=arxiv&logoColor=white)]()
[![Project Page](https://img.shields.io/badge/Project%20Page-Website-yellow?style=flat&logo=googlechrome&logoColor=white)](https://gazevla.github.io/)
[![Video YouTube](https://img.shields.io/badge/Video-YouTube-FF0000?style=flat&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=a1-jS9uWwAI&t=20s)
[![Data](https://img.shields.io/badge/Data-Dataset-green?style=flat&logo=databricks&logoColor=white)](https://github.com/lichy2004/GazeVLA-Data)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

</details>

![](figures/teaser.png)
We propose a novel learning-from-human framework that explicitly models intention to capture the causal structure of manipulation behavior.

## 📋 Table of Contents
- [🛠 Environment Setup](#-environment-setup)
- [🧩 Model Architecture](#-model-architecture)
- [💡 Experiments](#-experiments)
- [🙏 Acknowledgements](#-acknowledgements)
- [✍️ Citation](#️-citation)


## 🛠 Environment Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/lichy2004/GazeVLA-Code.git GazeVLA
cd GazeVLA
```

### Step 2: Set Up Python Environment
```bash
# Create a conda environment
conda create -n gazevla python=3.11 -y
conda activate gazevla

# Install requirements
pip install -r requirements.txt

# Install openpi-client
cd packages/openpi-client
pip install -e .
cd ../..

# Install GazeVLA
pip install -e .
```

## 🧩 Model Architecture

<div align="center">
<img src="./figures/pipeline.png" width="700" alt="VLIA Architecture">
</div>

Our model receives a task description, an egocentric observation, and the human or robot state as inputs. It first predicts discrete intention tokens, followed by continuous action generation via an intention–action reasoning chain. By explicitly modeling intention as an intermediate representation, the framework bridges high-level task understanding and low-level control. We instantiate intention as gaze, parameterized as 2D image coordinates.


## 💡 Experiments

### 📦 Data Preparation
Download the [AV-ALOHA](https://github.com/Soltanilara/giava) dataset and place it in `./data/av-aloha/dataset`. 
The directory structure should look like this:

```text
./data/av-aloha
├── av_aloha_sim_cube_transfer
├── av_aloha_sim_peg_insertion
├── av_aloha_sim_slot_insertion
├── av_aloha_sim_hook_package
├── av_aloha_sim_pour_test_tube
└── av_aloha_sim_thread_needle
```

Then run `examples/process_data_1.py` and `examples/process_data_2.py` to process the data: the first step converts the raw data into HDF5 files, and the second step saves the results in RLDS format.


### 🔥 Training

Please run the following command for training, and you can modify the configuration in `src/openpi/training/config.py`.

```bash
python scripts/compute_norm_stats.py --config-name lfa-cam1-chunk25
CUDA_VISIBLE_DEVICES=0 \
    torchrun --nnodes=1 --nproc_per_node=1 --master_port=29500 \
    scripts/train/train_human_gaze_accelerate.py lfa-cam1-chunk25 \
    --exp_name=lfa-cam1-chunk25
```

### 🧪 Evaluation

You need to first download the official [LFA](https://github.com/Soltanilara/giava.git) repository, make some modifications.

## TODO

The following features are planned for future implementation:

- ✅ Training code.
- [ ] Pre-trained model checkpoints.
- [ ] Evaluation scripts.

##  🙏 Acknowledgements

Our code is built upon [Openpi](https://github.com/Physical-Intelligence/openpi) and [GraspVLA](https://github.com/PKU-EPIC/GraspVLA). These code serve as an essential foundation for our implementation, and we deeply appreciate the time, effort, and expertise they shared with the community.

## ✍️ Citation


If you find our work useful, please cite us:


```
@article{}
```

## License

 This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

 [![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

 [cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
 [cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

<!-- *Chart updates automatically. Click to interact with the full timeline.* -->