from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union, Optional

import gym
import torch
import torch as th
from torch import nn

from stable_baselines3.common.type_aliases import TensorDict


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()



def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetImaginationExtractorGP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, pc_key: str, feat_key: Optional[str] = None,
                 out_channel=256, extractor_name="smallpn",
                 gt_key: Optional[str] = None, imagination_keys=("imagination_robot",), state_key="state",
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU, *kwargs):
        self.imagination_key = imagination_keys
        # Init state representation
        self.use_state = state_key is not None
        self.state_key = state_key

        print(f"extractor use state = {self.use_state}")
        if self.use_state:
            if state_key not in observation_space.spaces.keys():
                raise RuntimeError(f"State key {state_key} not in observation space: {observation_space}")
            self.state_space = observation_space[self.state_key]
        if feat_key is not None:
            if feat_key not in list(observation_space.keys()):
                raise RuntimeError(f"Feature key {feat_key} not in observation space.")
        if pc_key not in list(observation_space.keys()):
            raise RuntimeError(f"Point cloud key {pc_key} not in observation space.")

        super().__init__(observation_space, out_channel)
        # Point cloud input should have size (n, 3), spec size (n, 3), feat size (n, m)
        self.pc_key = pc_key
        self.has_feat = feat_key is not None
        self.feat_key = feat_key
        self.gt_key = gt_key

        if extractor_name == "smallpn":
            from stable_baselines3.networks.pretrain_nets import PointNet
            self.extractor = PointNet()
        elif extractor_name == "mediumpn":
            from stable_baselines3.networks.pretrain_nets import PointNetMedium
            self.extractor = PointNetMedium()
        elif extractor_name == "largepn":
            from stable_baselines3.networks.pretrain_nets import PointNetLarge
            self.extractor = PointNetLarge()
        else:
            raise NotImplementedError(f"Extractor {extractor_name} not implemented. Available:\
             smallpn, mediumpn, largepn")

        # self.n_input_channels = n_input_channels
        self.n_output_channels = out_channel
        assert self.n_output_channels == 256

        if self.use_state:
            self.state_dim = self.state_space.shape[0]
            if len(state_mlp_size) == 0:
                raise RuntimeError(f"State mlp size is empty")
            elif len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]

            self.n_output_channels = out_channel + output_dim
            self._features_dim = self.n_output_channels
            self.state_mlp = nn.Sequential(*create_mlp(self.state_dim, output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observations: TensorDict) -> th.Tensor:
        # get raw point cloud segmentation mask
        points = observations[self.pc_key]  # B * N * 3
        b, _, _ = points.shape
        if len(self.imagination_key) > 0:
            for key in self.imagination_key:
                obs = observations[key]
                if len(obs.shape) == 2:
                    obs = obs.unsqueeze(0)
                img_points = obs[:, :, :3]
                points = torch.concat([points, img_points], dim=1)

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * 256
        if self.use_state:
            state_feat = self.state_mlp(observations[self.state_key])
            return torch.cat([pn_feat, state_feat], dim=-1)
        else:
            return pn_feat

