"""
This file contains the gym environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
from collections import OrderedDict
import json
from copy import deepcopy

from omegaconf import DictConfig, OmegaConf
import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils
from neural_mp.envs import *

import torch

# Util function for loading point clouds|
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

from neural_mp.envs.franka_pybullet_env import depth_to_rgb


class EnvMP(EB.EnvBase):
    """Wrapper class for motion planning envs"""
    def __init__(
        self,
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True, 
        pcd_params=None,
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): ignored - gym envs always support on-screen rendering

            render_offscreen (bool): ignored - gym envs always support off-screen rendering

            use_image_obs (bool): ignored - gym envs don't typically use images

            postprocess_visual_obs (bool): ignored - gym envs don't typically use images
        """
        if type(kwargs['cfg']) == dict:
            cfg = OmegaConf.create(kwargs['cfg'])
        elif type(kwargs['cfg']) == DictConfig:
            cfg = kwargs['cfg']
            kwargs['cfg'] = OmegaConf.to_container(kwargs['cfg'], resolve=True)
        self._init_kwargs = deepcopy(kwargs)
        self._env_name = env_name
        self._current_obs = None
        self._current_reward = None
        self._current_done = None
        self._done = None
        self.pcd_params = pcd_params
        self.env = eval(env_name)(cfg)

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, reward, done, info = self.env.step(action.copy())
        self._current_obs = obs
        self._current_reward = reward
        self._current_done = done
        return self.get_observation(obs), reward, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        self._current_obs = self.env.reset()
        self._current_reward = None
        self._current_done = self.is_success()
        return self.get_observation(self._current_obs)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains:
                - states (np.ndarray): initial state of the mujoco environment
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        self.env.set_state(state["states"])
        return self.get_observation(self.env.get_observation())

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
        """
        if mode =="human":
            return self.env.render(mode=mode, **kwargs)
        if mode == "rgb_array":
            return self.env.get_alpha_blended_target_img(
                        self.env.goal_mask, self.env.goal_img
                    )[:, :, ::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.

        Args:
            ob (np.array): current flat observation vector to wrap and provide as a dictionary.
                If not provided, uses self._current_obs.
        """
        if obs is None:
            obs = self.env.get_observation()
        ob_return = OrderedDict()
        for k in obs:
            ob_return[k] = obs[k].copy()
            if k.endswith('image'):
                #TODO: figure out how to do this correctly without hacking
                ob_return[k] = ob_return[k].transpose(2, 0, 1)
            if k.endswith('depth'):
                ob_return[k] = depth_to_rgb(ob_return[k])
                ob_return[k] = ob_return[k].transpose(2, 0, 1)
        return ob_return

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        return dict(states=self.env.get_state())

    def get_reward(self):
        """
        Get current reward.
        """
        assert self._current_reward is not None
        return self._current_reward

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        assert self._current_done is not None
        return self._current_done

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        return dict(task=float(self.env.get_success(self.env.goal_angles)[0]))

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return 7

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.MP_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def create_for_data_processing(cls, env_name, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. For gym environments, input arguments (other than @env_name)
        are ignored, since environments are mostly pre-configured.

        Args:
            env_name (str): name of gym environment to create

        Returns:
            env (EnvGym instance)
        """

        # make sure to initialize obs utils so it knows which modalities are image modalities.
        # For currently supported gym tasks, there are no image observations.
        obs_modality_specs = {
            "obs": {
                "low_dim": ["flat"],
                "rgb": [],
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        return cls(env_name=env_name, **kwargs)

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return ()

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)

def render_pointcloud(pcd):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    verts = torch.Tensor(pcd[:, :3]).to(device)

    rgb_pcd = pcd[:, 3]
    rgb_pcd = rgb_pcd.reshape(-1, 1)
    # convert each rgb value to a different color
    # 0 -> red (255, 0, 0)
    # 1 -> green (0, 255, 0)
    # 2 -> blue (0, 0, 255)
    rgb_pcd = np.concatenate((rgb_pcd, rgb_pcd, rgb_pcd), axis=1)
    rgb_pcd = np.where(rgb_pcd == np.array([0, 0, 0]), np.array([255, 0, 0]), rgb_pcd)
    rgb_pcd = np.where(rgb_pcd == np.array([1, 1, 1]), np.array([0, 255, 0]), rgb_pcd)
    rgb_pcd = np.where(rgb_pcd == np.array([2, 2, 2]), np.array([0, 0, 255]), rgb_pcd)

    rgb_pcd = rgb_pcd * 255
    rgb = torch.Tensor(rgb_pcd).to(device)

    point_cloud = Pointclouds(points=[verts], features=[rgb])

    # Initialize a camera.
    R, T = look_at_view_transform(1, 0, 90, up=((1, 0, 0),))
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
    raster_settings = PointsRasterizationSettings(
        image_size=512, 
        # radius = 0.003,
        points_per_pixel = 10
    )


    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    images = renderer(point_cloud)
    img = images[0, ..., :3].cpu().numpy()
    return img

def render_single_pointcloud(pcd):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    verts = torch.Tensor(pcd[:, :3]).to(device)

    rgb_pcd = torch.ones_like(verts) * torch.tensor([255, 0, 0]).to(device)
    rgb = torch.Tensor(rgb_pcd).to(device)

    point_cloud = Pointclouds(points=[verts], features=[rgb])

    # Initialize a camera.
    R, T = look_at_view_transform(1, 0, 90, up=((1, 0, 0),))
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
    raster_settings = PointsRasterizationSettings(
        image_size=512, 
        # radius = 0.003,
        points_per_pixel = 10
    )


    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    images = renderer(point_cloud)
    img = images[0, ..., :3].cpu().numpy()
    return img

if __name__ == "__main__":
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    initialize(config_path="../../../neural_mp/neural_mp/configs", job_name="")
    cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))
    EnvMP(env_name=cfg.task.env_name, cfg=cfg)