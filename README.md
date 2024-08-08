# Bi-DexHands: Bimanual Dexterous Manipulation via Reinforcement Learning

## Environment
- Nvidia RTX4060
- Ubuntu20.04 with python3.8
- cuda11.1
- cudnn8.9.5
- torch1.8.1+cu111
- torchvision0.9.1
- torchaudio0.8.1
- (editor: vscode)

## Work
Create a new task called allegro_hand_catch_point_cloud in task/AllegroHandCatchPointCloud.py.
Pass `python train.py --task=AllegroHandCatchPointCloud --algo=ddpg` in DexterousHands/bidexhands/ to train.

## ps
If raise the error `RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)`, 
remove `@torch.jit.script` in `isaacgym/torch_utils.py` and `task/AllegroHandCatchPointCloud.py`. 
