# PilotANN
> Memory-bounded GPU acceleration for graph-based approximate nearest neighbor search.

## Updates
**October 17, 2025**: 
- Fixed build issues caused by dependency conflicts:
  - For numpy conflicts, install numpy 1.26: `pip install numpy==1.26`
  - For "torch not found" errors during setup.py, install setuptools 75.8 (Feb 27, 2025 version): `pip install setuptools==75.8`

## Build
### RHEL/CentOS Setup
```
# Install development tools
sudo dnf install @'Development Tools'

# Install additional dependencies
sudo dnf install epel-release
sudo dnf config-manager --set-enabled crb
sudo dnf install cmake git swig lapack lapack-devel
```
### CUDA Setup
```
# Enable NVIDIA repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Install CUDA
sudo dnf clean all
sudo dnf -y install cuda-toolkit-11-8
sudo dnf module install nvidia-driver:latest-dkms

# Set environment variables (add to ~/.bashrc):
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
### Python Setup (Anaconda or Miniconda)
```
# The code is built on top of LibTorch
conda create -n pytorch python=3.10 && conda activate pytorch
conda install pytorch=2.1.2 torchvision torchaudio pytorch-cuda=11.8 numpy -c pytorch -c nvidia
```
### Build FAISS (Under Current Python Env)
```
# Download source
git clone https://github.com/facebookresearch/faiss/archive/refs/tags/v1.8.0.tar.gz
tar -xvf v1.8.0.tar.gz && cd faiss-*

# Build faiss
mkdir build && cd build
cmake .. -DFAISS_OPT_LEVEL='avx2' \
         -DFAISS_ENABLE_GPU=OFF   \
         -DFAISS_ENABLE_PYTHON=ON \
         -DCMAKE_BUILD_TYPE=Release

# Install
cd faiss/python && python3 ./setup.py install
```
### Build PilotANN (Under Current Python Env)
```
cd pilot_ann_src
python3 ./setup.py develop
```


## Dataset Setup
+ Expected directory structure:
```
.datasets/
├── deep-1m/
├── deep-100m/
└── laion-100m/
```


## Run Benchmark
```
# Using 16 CPUs
export OMP_THREAD_LIMIT=16

# Run benchmark
python3 script/bench_1.py --dataset='laion-1m' --sample_ratio=0.25 --d_principle=128
python3 script/bench_1.py --dataset='laion-100m' --sample_ratio=0.25 --d_principle=128
```
### Available benchmark options
+ --top_k: Set `k` (any int <= 100, default 10)
+ --dataset: Choose dataset (deep-1m, text2img-1m, wiki-1m, laion-1m)
+ --sample_ratio: Set sampling ratio (any float in (0.0, 1.0])
+ --d_principle: Set SVD reduction dimension (any int, prefer numbers divisible by 32)
+ --algorithm: Choose graph angorithm (hnsw, nsg)
+ --router: Entry selection method (random32, random64, router32x32, router64x64, router means FES)
+ --n_queries: Set number of queries to process (default 1024)


Benchmark Results
+ Example output
```
[hnsw] dataset: laion-1m, d_model: 768
svd 128
build nsw
sampling 0.25
[INFO] building entry

evaluate:   0%|          | 0/22 [00:00<?, ?it/s][INFO] init cuda streams
[INFO] creating bitmask_pool

evaluate:   5%|▍         | 1/22 [00:02<00:55,  2.62s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:45,  2.27s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:40,  2.16s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:37,  2.10s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:35,  2.07s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:32,  2.06s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:30,  2.05s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:28,  2.04s/it]
evaluate:  41%|████      | 9/22 [00:18<00:26,  2.03s/it]
evaluate:  45%|████▌     | 10/22 [00:20<00:24,  2.03s/it]
evaluate:  50%|█████     | 11/22 [00:22<00:22,  2.03s/it]
evaluate:  55%|█████▍    | 12/22 [00:24<00:20,  2.03s/it]
evaluate:  59%|█████▉    | 13/22 [00:26<00:18,  2.02s/it]
evaluate:  64%|██████▎   | 14/22 [00:28<00:16,  2.02s/it]
evaluate:  68%|██████▊   | 15/22 [00:30<00:14,  2.02s/it]
evaluate:  73%|███████▎  | 16/22 [00:32<00:12,  2.02s/it]
evaluate:  77%|███████▋  | 17/22 [00:34<00:10,  2.02s/it]
evaluate:  82%|████████▏ | 18/22 [00:36<00:08,  2.02s/it]
evaluate:  86%|████████▋ | 19/22 [00:39<00:06,  2.02s/it]
evaluate:  91%|█████████ | 20/22 [00:41<00:04,  2.02s/it]
evaluate:  95%|█████████▌| 21/22 [00:43<00:02,  2.02s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.02s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.05s/it]
search-pilot: recall=0.904, duration=15.627ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:42,  2.03s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:40,  2.03s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:38,  2.03s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:36,  2.03s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:34,  2.03s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:32,  2.03s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:30,  2.03s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:28,  2.03s/it]
evaluate:  41%|████      | 9/22 [00:18<00:26,  2.03s/it]
evaluate:  45%|████▌     | 10/22 [00:20<00:24,  2.03s/it]
evaluate:  50%|█████     | 11/22 [00:22<00:22,  2.03s/it]
evaluate:  55%|█████▍    | 12/22 [00:24<00:20,  2.03s/it]
evaluate:  59%|█████▉    | 13/22 [00:26<00:18,  2.03s/it]
evaluate:  64%|██████▎   | 14/22 [00:28<00:16,  2.03s/it]
evaluate:  68%|██████▊   | 15/22 [00:30<00:14,  2.03s/it]
evaluate:  73%|███████▎  | 16/22 [00:32<00:12,  2.03s/it]
evaluate:  77%|███████▋  | 17/22 [00:34<00:10,  2.03s/it]
evaluate:  82%|████████▏ | 18/22 [00:36<00:08,  2.03s/it]
evaluate:  86%|████████▋ | 19/22 [00:38<00:06,  2.03s/it]
evaluate:  91%|█████████ | 20/22 [00:40<00:04,  2.03s/it]
evaluate:  95%|█████████▌| 21/22 [00:42<00:02,  2.03s/it]
evaluate: 100%|██████████| 22/22 [00:44<00:00,  2.03s/it]
evaluate: 100%|██████████| 22/22 [00:44<00:00,  2.03s/it]
search-pilot: recall=0.939, duration=24.763ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:42,  2.04s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:40,  2.04s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:38,  2.04s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:36,  2.04s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:34,  2.04s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:32,  2.04s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:30,  2.04s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:28,  2.04s/it]
evaluate:  41%|████      | 9/22 [00:18<00:26,  2.04s/it]
evaluate:  45%|████▌     | 10/22 [00:20<00:24,  2.04s/it]
evaluate:  50%|█████     | 11/22 [00:22<00:22,  2.04s/it]
evaluate:  55%|█████▍    | 12/22 [00:24<00:20,  2.04s/it]
evaluate:  59%|█████▉    | 13/22 [00:26<00:18,  2.04s/it]
evaluate:  64%|██████▎   | 14/22 [00:28<00:16,  2.04s/it]
evaluate:  68%|██████▊   | 15/22 [00:30<00:14,  2.04s/it]
evaluate:  73%|███████▎  | 16/22 [00:32<00:12,  2.04s/it]
evaluate:  77%|███████▋  | 17/22 [00:34<00:10,  2.04s/it]
evaluate:  82%|████████▏ | 18/22 [00:36<00:08,  2.04s/it]
evaluate:  86%|████████▋ | 19/22 [00:38<00:06,  2.04s/it]
evaluate:  91%|█████████ | 20/22 [00:40<00:04,  2.04s/it]
evaluate:  95%|█████████▌| 21/22 [00:42<00:02,  2.04s/it]
evaluate: 100%|██████████| 22/22 [00:44<00:00,  2.04s/it]
evaluate: 100%|██████████| 22/22 [00:44<00:00,  2.04s/it]
search-pilot: recall=0.948, duration=32.539ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:43,  2.05s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:40,  2.05s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:38,  2.05s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:36,  2.05s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:34,  2.05s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:32,  2.05s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:30,  2.05s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:28,  2.05s/it]
evaluate:  41%|████      | 9/22 [00:18<00:26,  2.05s/it]
evaluate:  45%|████▌     | 10/22 [00:20<00:24,  2.05s/it]
evaluate:  50%|█████     | 11/22 [00:22<00:22,  2.05s/it]
evaluate:  55%|█████▍    | 12/22 [00:24<00:20,  2.05s/it]
evaluate:  59%|█████▉    | 13/22 [00:26<00:18,  2.05s/it]
evaluate:  64%|██████▎   | 14/22 [00:28<00:16,  2.05s/it]
evaluate:  68%|██████▊   | 15/22 [00:30<00:14,  2.05s/it]
evaluate:  73%|███████▎  | 16/22 [00:32<00:12,  2.05s/it]
evaluate:  77%|███████▋  | 17/22 [00:34<00:10,  2.05s/it]
evaluate:  82%|████████▏ | 18/22 [00:36<00:08,  2.05s/it]
evaluate:  86%|████████▋ | 19/22 [00:38<00:06,  2.05s/it]
evaluate:  91%|█████████ | 20/22 [00:40<00:04,  2.05s/it]
evaluate:  95%|█████████▌| 21/22 [00:42<00:02,  2.05s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.05s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.05s/it]
search-pilot: recall=0.956, duration=40.757ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:43,  2.06s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:41,  2.06s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:39,  2.06s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:37,  2.06s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:35,  2.06s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:32,  2.06s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:31,  2.07s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:28,  2.07s/it]
evaluate:  41%|████      | 9/22 [00:18<00:26,  2.07s/it]
evaluate:  45%|████▌     | 10/22 [00:20<00:24,  2.07s/it]
evaluate:  50%|█████     | 11/22 [00:22<00:22,  2.06s/it]
evaluate:  55%|█████▍    | 12/22 [00:24<00:20,  2.06s/it]
evaluate:  59%|█████▉    | 13/22 [00:26<00:18,  2.06s/it]
evaluate:  64%|██████▎   | 14/22 [00:28<00:16,  2.06s/it]
evaluate:  68%|██████▊   | 15/22 [00:30<00:14,  2.06s/it]
evaluate:  73%|███████▎  | 16/22 [00:33<00:12,  2.06s/it]
evaluate:  77%|███████▋  | 17/22 [00:35<00:10,  2.06s/it]
evaluate:  82%|████████▏ | 18/22 [00:37<00:08,  2.06s/it]
evaluate:  86%|████████▋ | 19/22 [00:39<00:06,  2.06s/it]
evaluate:  91%|█████████ | 20/22 [00:41<00:04,  2.06s/it]
evaluate:  95%|█████████▌| 21/22 [00:43<00:02,  2.06s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.06s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.06s/it]
search-pilot: recall=0.964, duration=55.374ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:43,  2.08s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:41,  2.08s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:39,  2.08s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:37,  2.08s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:35,  2.08s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:33,  2.08s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:31,  2.08s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:29,  2.08s/it]
evaluate:  41%|████      | 9/22 [00:18<00:26,  2.08s/it]
evaluate:  45%|████▌     | 10/22 [00:20<00:24,  2.08s/it]
evaluate:  50%|█████     | 11/22 [00:22<00:22,  2.08s/it]
evaluate:  55%|█████▍    | 12/22 [00:24<00:20,  2.08s/it]
evaluate:  59%|█████▉    | 13/22 [00:26<00:18,  2.08s/it]
evaluate:  64%|██████▎   | 14/22 [00:29<00:16,  2.08s/it]
evaluate:  68%|██████▊   | 15/22 [00:31<00:14,  2.08s/it]
evaluate:  73%|███████▎  | 16/22 [00:33<00:12,  2.08s/it]
evaluate:  77%|███████▋  | 17/22 [00:35<00:10,  2.08s/it]
evaluate:  82%|████████▏ | 18/22 [00:37<00:08,  2.08s/it]
evaluate:  86%|████████▋ | 19/22 [00:39<00:06,  2.08s/it]
evaluate:  91%|█████████ | 20/22 [00:41<00:04,  2.08s/it]
evaluate:  95%|█████████▌| 21/22 [00:43<00:02,  2.08s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.08s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.08s/it]
search-pilot: recall=0.968, duration=69.943ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:44,  2.11s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:42,  2.11s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:40,  2.11s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:37,  2.11s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:35,  2.11s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:33,  2.11s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:31,  2.11s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:29,  2.11s/it]
evaluate:  41%|████      | 9/22 [00:18<00:27,  2.11s/it]
evaluate:  45%|████▌     | 10/22 [00:21<00:25,  2.11s/it]
evaluate:  50%|█████     | 11/22 [00:23<00:23,  2.11s/it]
evaluate:  55%|█████▍    | 12/22 [00:25<00:21,  2.11s/it]
evaluate:  59%|█████▉    | 13/22 [00:27<00:18,  2.11s/it]
evaluate:  64%|██████▎   | 14/22 [00:29<00:16,  2.11s/it]
evaluate:  68%|██████▊   | 15/22 [00:31<00:14,  2.11s/it]
evaluate:  73%|███████▎  | 16/22 [00:33<00:12,  2.11s/it]
evaluate:  77%|███████▋  | 17/22 [00:35<00:10,  2.11s/it]
evaluate:  82%|████████▏ | 18/22 [00:37<00:08,  2.11s/it]
evaluate:  86%|████████▋ | 19/22 [00:40<00:06,  2.11s/it]
evaluate:  91%|█████████ | 20/22 [00:42<00:04,  2.11s/it]
evaluate:  95%|█████████▌| 21/22 [00:44<00:02,  2.11s/it]
evaluate: 100%|██████████| 22/22 [00:46<00:00,  2.11s/it]
evaluate: 100%|██████████| 22/22 [00:46<00:00,  2.11s/it]
search-pilot: recall=0.971, duration=98.985ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:44,  2.13s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:42,  2.13s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:40,  2.13s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:38,  2.13s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:36,  2.13s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:34,  2.13s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:31,  2.13s/it]
evaluate:  36%|███▋      | 8/22 [00:17<00:29,  2.13s/it]
evaluate:  41%|████      | 9/22 [00:19<00:27,  2.13s/it]
evaluate:  45%|████▌     | 10/22 [00:21<00:25,  2.13s/it]
evaluate:  50%|█████     | 11/22 [00:23<00:23,  2.13s/it]
evaluate:  55%|█████▍    | 12/22 [00:25<00:21,  2.13s/it]
evaluate:  59%|█████▉    | 13/22 [00:27<00:19,  2.13s/it]
evaluate:  64%|██████▎   | 14/22 [00:29<00:17,  2.13s/it]
evaluate:  68%|██████▊   | 15/22 [00:31<00:14,  2.13s/it]
evaluate:  73%|███████▎  | 16/22 [00:34<00:12,  2.13s/it]
evaluate:  77%|███████▋  | 17/22 [00:36<00:10,  2.13s/it]
evaluate:  82%|████████▏ | 18/22 [00:38<00:08,  2.13s/it]
evaluate:  86%|████████▋ | 19/22 [00:40<00:06,  2.13s/it]
evaluate:  91%|█████████ | 20/22 [00:42<00:04,  2.13s/it]
evaluate:  95%|█████████▌| 21/22 [00:44<00:02,  2.13s/it]
evaluate: 100%|██████████| 22/22 [00:46<00:00,  2.13s/it]
evaluate: 100%|██████████| 22/22 [00:46<00:00,  2.13s/it]
search-pilot: recall=0.974, duration=125.944ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:43,  2.05s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:41,  2.05s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:39,  2.05s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:36,  2.05s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:34,  2.05s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:32,  2.05s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:30,  2.05s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:28,  2.05s/it]
evaluate:  41%|████      | 9/22 [00:18<00:26,  2.05s/it]
evaluate:  45%|████▌     | 10/22 [00:20<00:24,  2.05s/it]
evaluate:  50%|█████     | 11/22 [00:22<00:22,  2.05s/it]
evaluate:  55%|█████▍    | 12/22 [00:24<00:20,  2.05s/it]
evaluate:  59%|█████▉    | 13/22 [00:26<00:18,  2.05s/it]
evaluate:  64%|██████▎   | 14/22 [00:28<00:16,  2.05s/it]
evaluate:  68%|██████▊   | 15/22 [00:30<00:14,  2.05s/it]
evaluate:  73%|███████▎  | 16/22 [00:32<00:12,  2.05s/it]
evaluate:  77%|███████▋  | 17/22 [00:34<00:10,  2.05s/it]
evaluate:  82%|████████▏ | 18/22 [00:36<00:08,  2.05s/it]
evaluate:  86%|████████▋ | 19/22 [00:38<00:06,  2.05s/it]
evaluate:  91%|█████████ | 20/22 [00:41<00:04,  2.05s/it]
evaluate:  95%|█████████▌| 21/22 [00:43<00:02,  2.05s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.05s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.05s/it]
faiss: recall=0.832, duration=46.139ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:43,  2.07s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:41,  2.07s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:39,  2.07s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:37,  2.07s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:35,  2.07s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:33,  2.07s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:31,  2.07s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:29,  2.07s/it]
evaluate:  41%|████      | 9/22 [00:18<00:26,  2.07s/it]
evaluate:  45%|████▌     | 10/22 [00:20<00:24,  2.07s/it]
evaluate:  50%|█████     | 11/22 [00:22<00:22,  2.07s/it]
evaluate:  55%|█████▍    | 12/22 [00:24<00:20,  2.07s/it]
evaluate:  59%|█████▉    | 13/22 [00:26<00:18,  2.07s/it]
evaluate:  64%|██████▎   | 14/22 [00:29<00:16,  2.07s/it]
evaluate:  68%|██████▊   | 15/22 [00:31<00:14,  2.07s/it]
evaluate:  73%|███████▎  | 16/22 [00:33<00:12,  2.07s/it]
evaluate:  77%|███████▋  | 17/22 [00:35<00:10,  2.07s/it]
evaluate:  82%|████████▏ | 18/22 [00:37<00:08,  2.07s/it]
evaluate:  86%|████████▋ | 19/22 [00:39<00:06,  2.07s/it]
evaluate:  91%|█████████ | 20/22 [00:41<00:04,  2.07s/it]
evaluate:  95%|█████████▌| 21/22 [00:43<00:02,  2.07s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.07s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.07s/it]
faiss: recall=0.903, duration=65.952ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:43,  2.09s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:41,  2.09s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:39,  2.09s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:37,  2.09s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:35,  2.09s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:33,  2.09s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:31,  2.09s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:29,  2.09s/it]
evaluate:  41%|████      | 9/22 [00:18<00:27,  2.09s/it]
evaluate:  45%|████▌     | 10/22 [00:20<00:25,  2.09s/it]
evaluate:  50%|█████     | 11/22 [00:22<00:22,  2.09s/it]
evaluate:  55%|█████▍    | 12/22 [00:25<00:20,  2.09s/it]
evaluate:  59%|█████▉    | 13/22 [00:27<00:18,  2.09s/it]
evaluate:  64%|██████▎   | 14/22 [00:29<00:16,  2.09s/it]
evaluate:  68%|██████▊   | 15/22 [00:31<00:14,  2.09s/it]
evaluate:  73%|███████▎  | 16/22 [00:33<00:12,  2.09s/it]
evaluate:  77%|███████▋  | 17/22 [00:35<00:10,  2.09s/it]
evaluate:  82%|████████▏ | 18/22 [00:37<00:08,  2.09s/it]
evaluate:  86%|████████▋ | 19/22 [00:39<00:06,  2.09s/it]
evaluate:  91%|█████████ | 20/22 [00:41<00:04,  2.09s/it]
evaluate:  95%|█████████▌| 21/22 [00:43<00:02,  2.09s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.09s/it]
evaluate: 100%|██████████| 22/22 [00:45<00:00,  2.09s/it]
faiss: recall=0.927, duration=84.055ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:44,  2.11s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:42,  2.11s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:40,  2.11s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:37,  2.11s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:35,  2.11s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:33,  2.11s/it]
evaluate:  32%|███▏      | 7/22 [00:14<00:31,  2.11s/it]
evaluate:  36%|███▋      | 8/22 [00:16<00:29,  2.11s/it]
evaluate:  41%|████      | 9/22 [00:18<00:27,  2.11s/it]
evaluate:  45%|████▌     | 10/22 [00:21<00:25,  2.11s/it]
evaluate:  50%|█████     | 11/22 [00:23<00:23,  2.11s/it]
evaluate:  55%|█████▍    | 12/22 [00:25<00:21,  2.11s/it]
evaluate:  59%|█████▉    | 13/22 [00:27<00:18,  2.11s/it]
evaluate:  64%|██████▎   | 14/22 [00:29<00:16,  2.11s/it]
evaluate:  68%|██████▊   | 15/22 [00:31<00:14,  2.11s/it]
evaluate:  73%|███████▎  | 16/22 [00:33<00:12,  2.11s/it]
evaluate:  77%|███████▋  | 17/22 [00:35<00:10,  2.11s/it]
evaluate:  82%|████████▏ | 18/22 [00:37<00:08,  2.11s/it]
evaluate:  86%|████████▋ | 19/22 [00:40<00:06,  2.11s/it]
evaluate:  91%|█████████ | 20/22 [00:42<00:04,  2.11s/it]
evaluate:  95%|█████████▌| 21/22 [00:44<00:02,  2.11s/it]
evaluate: 100%|██████████| 22/22 [00:46<00:00,  2.11s/it]
evaluate: 100%|██████████| 22/22 [00:46<00:00,  2.11s/it]
faiss: recall=0.941, duration=102.547ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:45,  2.15s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:42,  2.15s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:40,  2.15s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:38,  2.14s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:36,  2.14s/it]
evaluate:  27%|██▋       | 6/22 [00:12<00:34,  2.14s/it]
evaluate:  32%|███▏      | 7/22 [00:15<00:32,  2.15s/it]
evaluate:  36%|███▋      | 8/22 [00:17<00:30,  2.15s/it]
evaluate:  41%|████      | 9/22 [00:19<00:27,  2.15s/it]
evaluate:  45%|████▌     | 10/22 [00:21<00:25,  2.15s/it]
evaluate:  50%|█████     | 11/22 [00:23<00:23,  2.15s/it]
evaluate:  55%|█████▍    | 12/22 [00:25<00:21,  2.15s/it]
evaluate:  59%|█████▉    | 13/22 [00:27<00:19,  2.14s/it]
evaluate:  64%|██████▎   | 14/22 [00:30<00:17,  2.15s/it]
evaluate:  68%|██████▊   | 15/22 [00:32<00:15,  2.15s/it]
evaluate:  73%|███████▎  | 16/22 [00:34<00:12,  2.15s/it]
evaluate:  77%|███████▋  | 17/22 [00:36<00:10,  2.15s/it]
evaluate:  82%|████████▏ | 18/22 [00:38<00:08,  2.15s/it]
evaluate:  86%|████████▋ | 19/22 [00:40<00:06,  2.15s/it]
evaluate:  91%|█████████ | 20/22 [00:42<00:04,  2.15s/it]
evaluate:  95%|█████████▌| 21/22 [00:45<00:02,  2.15s/it]
evaluate: 100%|██████████| 22/22 [00:47<00:00,  2.15s/it]
evaluate: 100%|██████████| 22/22 [00:47<00:00,  2.15s/it]
faiss: recall=0.953, duration=138.678ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:45,  2.18s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:43,  2.18s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:41,  2.18s/it]
evaluate:  18%|█▊        | 4/22 [00:08<00:39,  2.18s/it]
evaluate:  23%|██▎       | 5/22 [00:10<00:37,  2.18s/it]
evaluate:  27%|██▋       | 6/22 [00:13<00:34,  2.18s/it]
evaluate:  32%|███▏      | 7/22 [00:15<00:32,  2.18s/it]
evaluate:  36%|███▋      | 8/22 [00:17<00:30,  2.18s/it]
evaluate:  41%|████      | 9/22 [00:19<00:28,  2.18s/it]
evaluate:  45%|████▌     | 10/22 [00:21<00:26,  2.18s/it]
evaluate:  50%|█████     | 11/22 [00:24<00:24,  2.18s/it]
evaluate:  55%|█████▍    | 12/22 [00:26<00:21,  2.18s/it]
evaluate:  59%|█████▉    | 13/22 [00:28<00:19,  2.18s/it]
evaluate:  64%|██████▎   | 14/22 [00:30<00:17,  2.18s/it]
evaluate:  68%|██████▊   | 15/22 [00:32<00:15,  2.18s/it]
evaluate:  73%|███████▎  | 16/22 [00:34<00:13,  2.18s/it]
evaluate:  77%|███████▋  | 17/22 [00:37<00:10,  2.18s/it]
evaluate:  82%|████████▏ | 18/22 [00:39<00:08,  2.18s/it]
evaluate:  86%|████████▋ | 19/22 [00:41<00:06,  2.18s/it]
evaluate:  91%|█████████ | 20/22 [00:43<00:04,  2.18s/it]
evaluate:  95%|█████████▌| 21/22 [00:45<00:02,  2.18s/it]
evaluate: 100%|██████████| 22/22 [00:48<00:00,  2.18s/it]
evaluate: 100%|██████████| 22/22 [00:48<00:00,  2.18s/it]
faiss: recall=0.960, duration=176.475ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:47,  2.26s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:45,  2.26s/it]
evaluate:  14%|█▎        | 3/22 [00:06<00:42,  2.26s/it]
evaluate:  18%|█▊        | 4/22 [00:09<00:40,  2.26s/it]
evaluate:  23%|██▎       | 5/22 [00:11<00:38,  2.26s/it]
evaluate:  27%|██▋       | 6/22 [00:13<00:36,  2.26s/it]
evaluate:  32%|███▏      | 7/22 [00:15<00:33,  2.26s/it]
evaluate:  36%|███▋      | 8/22 [00:18<00:31,  2.26s/it]
evaluate:  41%|████      | 9/22 [00:20<00:29,  2.26s/it]
evaluate:  45%|████▌     | 10/22 [00:22<00:27,  2.26s/it]
evaluate:  50%|█████     | 11/22 [00:24<00:24,  2.26s/it]
evaluate:  55%|█████▍    | 12/22 [00:27<00:22,  2.26s/it]
evaluate:  59%|█████▉    | 13/22 [00:29<00:20,  2.26s/it]
evaluate:  64%|██████▎   | 14/22 [00:31<00:18,  2.26s/it]
evaluate:  68%|██████▊   | 15/22 [00:33<00:15,  2.26s/it]
evaluate:  73%|███████▎  | 16/22 [00:36<00:13,  2.26s/it]
evaluate:  77%|███████▋  | 17/22 [00:38<00:11,  2.26s/it]
evaluate:  82%|████████▏ | 18/22 [00:40<00:09,  2.26s/it]
evaluate:  86%|████████▋ | 19/22 [00:42<00:06,  2.26s/it]
evaluate:  91%|█████████ | 20/22 [00:45<00:04,  2.26s/it]
evaluate:  95%|█████████▌| 21/22 [00:47<00:02,  2.26s/it]
evaluate: 100%|██████████| 22/22 [00:49<00:00,  2.26s/it]
evaluate: 100%|██████████| 22/22 [00:49<00:00,  2.26s/it]
faiss: recall=0.967, duration=250.986ms

evaluate:   0%|          | 0/22 [00:00<?, ?it/s]
evaluate:   5%|▍         | 1/22 [00:02<00:49,  2.33s/it]
evaluate:   9%|▉         | 2/22 [00:04<00:46,  2.34s/it]
evaluate:  14%|█▎        | 3/22 [00:07<00:44,  2.34s/it]
evaluate:  18%|█▊        | 4/22 [00:09<00:42,  2.34s/it]
evaluate:  23%|██▎       | 5/22 [00:11<00:39,  2.34s/it]
evaluate:  27%|██▋       | 6/22 [00:14<00:37,  2.34s/it]
evaluate:  32%|███▏      | 7/22 [00:16<00:35,  2.34s/it]
evaluate:  36%|███▋      | 8/22 [00:18<00:32,  2.34s/it]
evaluate:  41%|████      | 9/22 [00:21<00:30,  2.34s/it]
evaluate:  45%|████▌     | 10/22 [00:23<00:28,  2.34s/it]
evaluate:  50%|█████     | 11/22 [00:25<00:25,  2.34s/it]
evaluate:  55%|█████▍    | 12/22 [00:28<00:23,  2.34s/it]
evaluate:  59%|█████▉    | 13/22 [00:30<00:21,  2.33s/it]
evaluate:  64%|██████▎   | 14/22 [00:32<00:18,  2.34s/it]
evaluate:  68%|██████▊   | 15/22 [00:35<00:16,  2.34s/it]
evaluate:  73%|███████▎  | 16/22 [00:37<00:14,  2.34s/it]
evaluate:  77%|███████▋  | 17/22 [00:39<00:11,  2.34s/it]
evaluate:  82%|████████▏ | 18/22 [00:42<00:09,  2.34s/it]
evaluate:  86%|████████▋ | 19/22 [00:44<00:07,  2.34s/it]
evaluate:  91%|█████████ | 20/22 [00:46<00:04,  2.34s/it]
evaluate:  95%|█████████▌| 21/22 [00:49<00:02,  2.34s/it]
evaluate: 100%|██████████| 22/22 [00:51<00:00,  2.34s/it]
evaluate: 100%|██████████| 22/22 [00:51<00:00,  2.34s/it]
faiss: recall=0.973, duration=329.534ms
search-pilot {'recall': 0.9042089843749965, 'timing': 15.62661695000429, 'n_neighbors': 32, 'ef_search': 16}
search-pilot {'recall': 0.9386181640624971, 'timing': 24.763099350008133, 'n_neighbors': 32, 'ef_search': 32}
search-pilot {'recall': 0.9479541015624975, 'timing': 32.53943175001268, 'n_neighbors': 32, 'ef_search': 48}
search-pilot {'recall': 0.9558251953124977, 'timing': 40.75729169999818, 'n_neighbors': 32, 'ef_search': 64}
search-pilot {'recall': 0.9635205078124978, 'timing': 55.373822949991336, 'n_neighbors': 32, 'ef_search': 96}
search-pilot {'recall': 0.9676708984374981, 'timing': 69.94290830000409, 'n_neighbors': 32, 'ef_search': 128}
search-pilot {'recall': 0.9708789062499983, 'timing': 98.98512790001064, 'n_neighbors': 32, 'ef_search': 192}
search-pilot {'recall': 0.9735009765624982, 'timing': 125.94395004999228, 'n_neighbors': 32, 'ef_search': 256}
faiss {'recall': 0.8318408203124973, 'timing': 46.139192449993516, 'n_neighbors': 32, 'ef_search': 16}
faiss {'recall': 0.9028222656249973, 'timing': 65.95181419999108, 'n_neighbors': 32, 'ef_search': 32}
faiss {'recall': 0.9267089843749978, 'timing': 84.05481609999015, 'n_neighbors': 32, 'ef_search': 48}
faiss {'recall': 0.9410009765624979, 'timing': 102.54711749997796, 'n_neighbors': 32, 'ef_search': 64}
faiss {'recall': 0.953413085937498, 'timing': 138.67827450001187, 'n_neighbors': 32, 'ef_search': 96}
faiss {'recall': 0.9601367187499985, 'timing': 176.47515465000652, 'n_neighbors': 32, 'ef_search': 128}
faiss {'recall': 0.9666748046874986, 'timing': 250.98584330000904, 'n_neighbors': 32, 'ef_search': 192}
faiss {'recall': 0.9726123046874987, 'timing': 329.5344065999984, 'n_neighbors': 32, 'ef_search': 256}
```
