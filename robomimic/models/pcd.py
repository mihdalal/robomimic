from torch import nn
import pytorch_lightning as pl
import torch
import numpy as np
from typing import Tuple
from pointnet2_ops.pointnet2_modules import PointnetSAModule

class MPiNetsPointNet(pl.LightningModule):
    def __init__(self, size='small', output_dim=1024):
        super().__init__()
        self.size = size
        self.output_dim = output_dim
        self._build_model()

    def _build_model(self):
        """
        Assembles the model design into a ModuleList
        """
        self.SA_modules = nn.ModuleList()
        if self.size == 'super_small':
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=128,
                    radius=0.05,
                    nsample=64,
                    mlp=[1, 64, 64, 64],
                    bn=False,
                )
            )
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=64,
                    radius=0.3,
                    nsample=64,
                    mlp=[64, 64, 64],
                    bn=False,
                )
            )
            self.SA_modules.append(PointnetSAModule(mlp=[64, 64, 64], bn=False))

            self.fc_layer = nn.Sequential(
                nn.Linear(64, 64),
                nn.GroupNorm(16, 64),
                nn.LeakyReLU(inplace=True),
                nn.Linear(64, 64),
                nn.GroupNorm(16, 64),
                nn.LeakyReLU(inplace=True),
                nn.Linear(64, 7),
            )
        elif self.size == 'small':
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=128,
                    radius=0.05,
                    nsample=64,
                    mlp=[1, 64, 64, 64],
                    bn=False,
                )
            )
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=64,
                    radius=0.3,
                    nsample=64,
                    mlp=[64, 128, 128, 256],
                    bn=False,
                )
            )
            self.SA_modules.append(PointnetSAModule(nsample=64, mlp=[256, 512, 512], bn=False))

            self.fc_layer = nn.Sequential(
                nn.Linear(512, 2048),
                nn.GroupNorm(16, 2048),
                nn.LeakyReLU(inplace=True),
                nn.Linear(2048, 1024),
                nn.GroupNorm(16, 1024),
                nn.LeakyReLU(inplace=True),
                nn.Linear(1024, self.output_dim),
            )
        elif self.size == 'medium':
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=512,
                    radius=0.05,
                    nsample=128,
                    mlp=[1, 64, 64, 64],
                    bn=False,
                )
            )
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=128,
                    radius=0.3,
                    nsample=128,
                    mlp=[64, 128, 128, 256],
                    bn=False,
                )
            )
            self.SA_modules.append(PointnetSAModule(nsample=128, mlp=[256, 512, 1024], bn=False))

            self.fc_layer = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.GroupNorm(16, 2048),
                nn.LeakyReLU(inplace=True),
                nn.Linear(2048, 1536),
                nn.GroupNorm(16, 1536),
                nn.LeakyReLU(inplace=True),
                nn.Linear(1536, 1536),
            )
        else:
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=512,
                    radius=0.05,
                    nsample=128,
                    mlp=[1, 64, 64, 64],
                    bn=False,
                )
            )
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=128,
                    radius=0.3,
                    nsample=128,
                    mlp=[64, 128, 128, 256],
                    bn=False,
                )
            )
            self.SA_modules.append(PointnetSAModule(nsample=128, mlp=[256, 512, 512, 1024], bn=False))

            self.fc_layer = nn.Sequential(
                nn.Linear(1024, 4096),
                nn.GroupNorm(16, 4096),
                nn.LeakyReLU(inplace=True),
                nn.Linear(4096, 2048),
                nn.GroupNorm(16, 2048),
                nn.LeakyReLU(inplace=True),
                nn.Linear(2048, 2048),
            )

    @staticmethod
    def _break_up_pc(pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Breaks up the point cloud into the xyz coordinates and segmentation mask

        :param pc torch.Tensor: Tensor with shape [B, N, M] where M is larger than 3.
                                The first three dimensions along the last axis will be x, y, z
        :rtype Tuple[torch.Tensor, torch.Tensor]: Two tensors, one with just xyz
            and one with the corresponding features
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous()
        return xyz, features

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass of the network

        :param point_cloud torch.Tensor: Has dimensions (B, N, 4)
                                              B is the batch size
                                              N is the number of points
                                              4 is x, y, z, segmentation_mask
                                              This tensor must be on the GPU (CPU tensors not supported)
        :rtype torch.Tensor: The output from the network
        """
        assert point_cloud.size(-1) == 4
        xyz, features = self._break_up_pc(point_cloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.fc_layer(features.squeeze(-1))