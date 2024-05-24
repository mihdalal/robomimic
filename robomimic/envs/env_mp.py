"""
This file contains the gym environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
from collections import OrderedDict
import traceback
import gym
import json
from copy import deepcopy

from tqdm import tqdm
import gymnasium

from omegaconf import DictConfig, OmegaConf
from neural_mp.utils.franka_utils import normalize_franka_joints
import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils
from neural_mp.envs import *

import os
import torch

# Util function for loading point clouds|
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

from neural_mp.envs.franka_pybullet_env import depth_to_rgb, compute_full_pcd
import h5py

class EnvMP(EB.EnvBase, gymnasium.Env):
    """Wrapper class for motion planning envs"""
    def __init__(
        self,
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True, 
        pcd_params=None,
        mpinets_enabled=False,
        dataset_path=None,
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
            
            pcd_params (dict): parameters for point cloud processing
            
            mpinets_enabled (bool): whether to use mpinets style unprocessing of delta actions
            
            dataset_path (str): path to dataset
            
            split (str): split of dataset to use
            
            env_idx (int): index of parallel env
            
            num_envs (int): number of parallel envs
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
        self.pcd_params = pcd_params if pcd_params is not None else dict()
        self.postprocesss_visual_obs = postprocess_visual_obs
        self.mpinets_enabled = mpinets_enabled
        self.env = eval(env_name)(cfg)
        # build observation space from the observation dictionary
        obs = self.get_observation()
        self.observation_space = gymnasium.spaces.Dict(
            {k: gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=v.dtype) for k, v in obs.items()}
        )
        # build action space from the action dictionary
        self.action_space = gymnasium.spaces.Box(
            low=-1*np.ones(7), high=np.ones(7), dtype=np.float32
        )
        self.dataset_path = dataset_path
        self.split = None
        self.demos = None
        self.num_envs = 1
        self.num_resets = 0
        self.dagger_resets = 0
        self.initial_states = None
        self.ep = 'demo_-1'
    
    def set_env_specific_params(self, split, num_envs, shift, use_env_idx, env_idx):
        """
        Note this sets params for the parallel envs to sample different demonstrations to init from
        This needs to be run before running the env!
        """
        self.split = split
        f = h5py.File(self.dataset_path, "r", libver='latest', swmr=True)
        self.hdf5_file = f
        filter_key = split

        # list of all demonstration episodes (sorted in increasing number order)
        if filter_key is not None:
            print("using filter key: {}".format(filter_key))
            demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
        else:
            demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        self.demos = [demos[i] for i in inds]
        if split == 'valid':
            if num_envs == 1:
                env_idx = 0 # only one env so don't shift
            else:
                env_idx = env_idx - shift
        if use_env_idx:
            self.demos = [demos[i] for i in range(len(demos)) if i % num_envs == env_idx]
        print("env idx: {}, split: {}, demos: {}".format(env_idx, split, len(self.demos)))
        self.num_envs = num_envs

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
        obs, reward, done, info = self.env.step(action.copy(), unnormalize_delta_actions=self.mpinets_enabled)
        self._current_obs = obs
        self._current_reward = reward
        self._current_done = done
        done = self.is_done()
        trunc = done # this is ignored but necessary for gymanasium compatibility
        # add dummy (None) values for all splits:
        new_info = {}
        for split in ["train", "valid", None]:
            for k, v in info.items():
                if split is None:
                    if split == self.split:
                        new_info[k] = v
                    else:
                        new_info[k] = None
                else:
                    if split == self.split:
                        new_info['{}/{}'.format(split, k)] = v
                    else:
                        new_info['{}/{}'.format(split, k)] = None
        # adding action err
        if self.demos is not None:
            # add self.split as a prefix to every key in infos
            info = {f"{self.split}/{k}": v for k, v in info.items()}
            if self.current_step < len(self.plan):
                action_err = np.linalg.norm(action - self.plan[self.current_step])
                action_mse = np.mean((action - self.plan[self.current_step])**2)
                self.total_action_err += action_err.item()
                self.total_action_mse += action_mse.item()
                info[f"{self.split}/action_err"] = self.total_action_err / (self.current_step + 1)
                info[f"{self.split}/action_mse"] = self.total_action_mse / (self.current_step + 1)
                for split in ['train', 'valid']:
                    if split != self.split:
                        info[f"{split}/action_err"] = None
                        info[f"{split}/action_mse"] = None
            else:
                info[f"{self.split}/action_err"] = self.total_action_err / len(self.plan)
                info[f"{self.split}/action_mse"] = self.total_action_mse / len(self.plan)
                for split in ['train', 'valid']:
                    if split != self.split:
                        info[f"{split}/action_err"] = None
                        info[f"{split}/action_mse"] = None
        else:
            for split in ['train', 'valid']:
                info[f"{split}/action_err"] = None
                info[f"{split}/action_mse"] = None
        new_info.update(info)
        info = new_info
        info['ep'] = int(self.ep.split('_')[-1])
        self.current_step += 1
        return self.get_observation(obs), reward, self.is_done(), trunc, info

    def reset(self, seed=None, reset_with_plan=False, reset_with_scene=True):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        # set seed
        if seed is None:
            seed = np.random.randint(0, 10000000)
        np.random.seed(seed)
        self._current_reward = None
        self._current_done = self.is_success()
        reset_infos = {} # this is ignored but necessary for gymanasium compatibility
        self.current_step = 0
        if self.demos is not None:
            # sample a new demonstration
            idx = self.num_resets % len(self.demos)
            ep = self.demos[idx]
            self.states = self.hdf5_file["data/{}/states".format(ep)][()]
            self.plan = self.hdf5_file["data/{}/actions".format(ep)][()]
            self.total_action_err = 0
            self.total_action_mse = 0
            print("resetting to demo: {}, idx: {}".format(ep, idx))
            self.num_resets += 1
            self.ep = ep
            return self.reset_to({"states": self.states[0]}), reset_infos
        elif self.initial_states is not None:
            idx = self.num_resets % len(self.initial_states)
            print("resetting to initial state idx: {}".format(idx))
            self.num_resets += 1
            return self.reset_to({"states": self.initial_states[idx]}), reset_infos
        else:
            print("resetting to random state")
            self._current_obs = self.env.reset(reset_with_plan=reset_with_plan, reset_with_scene=reset_with_scene)
            return self.get_observation(self._current_obs), reset_infos

    def set_to_dagger_sampling(self, num_envs, env_idx):
        """
        Set the environment to sampling train states only.
        """
        self.saved_demos = self.demos.copy() if self.demos is not None else self.demos
        self.saved_split = self.split
        self.saved_initial_states = self.initial_states.copy() if self.initial_states is not None else self.initial_states
        self.dataset_path = self.dataset_path
        self.saved_num_resets = self.num_resets
        self.num_resets = self.dagger_resets    
        f = h5py.File(self.dataset_path, "r", libver='latest', swmr=True)
        self.hdf5_file = f
        filter_key = 'train'

        # list of all demonstration episodes (sorted in increasing number order)
        if filter_key is not None:
            print("using filter key: {}".format(filter_key))
            demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
        else:
            demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        self.demos = [demos[i] for i in inds if i % num_envs == env_idx]
        self.split = 'train'
        print("env idx: {}, split: {}, demos: {}".format(env_idx, 'train', len(self.demos)))
    
    def set_to_env_original(self):
        """
        Set the environment back to the original state (in case it was sampling from dataset states).
        """
        self.demos = self.saved_demos
        self.split = self.saved_split
        self.initial_states = self.saved_initial_states
        self.dagger_resets = self.num_resets
        self.num_resets = self.saved_num_resets
    
    def relabel_traj_with_mp(self, trajs, env_idx):
        """
        Relabel all actions in a trajectory with motion planning.
        This is used for DAgger, so we will re-plan from every successive state in the trajectory.
        The key trick is that we re-use the search tree across planner calls so after the first call,
        the planning is much faster.
        NOTE: we assume all trajs contain a collision state and are of length << 50.
        We also only relabel until the first collision state.
        Args:
            trajs (_type_): _description_
            env_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        actions = []
        states = []
        cfg = self.env.cfg
        if env_idx >= len(trajs):
            return {}
        traj = trajs[env_idx]
        start_config = traj['obs']['current_angles'][0]
        goal_config = traj['obs']['goal_angles'][0] # will be const. throughout traj
        initial_state = traj['initial_state'][0]
        self.env.reset(reset_with_plan=False)
        self.reset_to(initial_state) # sets the obstacles
        self.env.set_robot_joint_state(start_config)
        mp_kwargs = cfg.task.mp_kwargs.copy()
        
        # load planner from data
        ep = 'demo_' + str(traj['info'][0]['ep'])
        dataset_name = os.path.basename(self.dataset_path).replace('.hdf5', '')
        data_path = f'planners/{dataset_name}/{ep}.pkl'
        import neural_mp
        data_path = os.path.join(os.path.dirname(neural_mp.__file__)[:-len('neural_mp')], data_path)
        current_angles = self.hdf5_file["data/{}/obs/current_angles".format(ep)][()][0]
        target_angles = self.hdf5_file["data/{}/obs/goal_angles".format(ep)][()][0]
        planner, pdef = self.env.load_planner_from_data(data_path, current_angles, target_angles)
        
        for step in range(len(traj['obs']['current_angles'])):
            mp_kwargs_ = mp_kwargs.copy()
            self.env.set_robot_joint_state(traj['obs']['current_angles'][step])
            mp_kwargs_['initial_planning_time'] = .1
            mp_kwargs_['maximum_planning_time'] = .1
            mp_kwargs_['execute_plan'] = False # assume we don't need to execute the plan
            mp_kwargs_['force_goal_reaching'] = True # this should already have been true in the dataset though
            try:
                (
                    plan_actions,
                    _,
                    _,
                    planning_states,
                    planner,
                    _,
                    _,
                ) = self.env.mp_to_joint_target(
                    goal_config, planner=planner, pdef=pdef,  **mp_kwargs_
                )
            except:
                print(traceback.format_exc())
                plan_actions = None
            # assumption: if original plan successful, can re-plan easily from other states
            if plan_actions is None:
                print("Failed to re-plan at step: {}".format(step))
                break
            if plan_actions.max() > 1 or plan_actions.min() < -1:
                print("plan actions out of bounds")
                break
            actions.append(plan_actions[0])
            states.append(planning_states[0])
            # check if it has collided, if so break
            if traj['collision_state'][step]:
                print('collision state reached, breaking')
                break        
        if len(actions) == 0:
            return {}
        # shorten obs to same length as actions
        obs = {k:v[:len(actions)] for k, v in traj['obs'].items()}  
        obs["compute_pcd_params"] = initial_state['states'][14:]
        states = np.array(states).astype(np.float32)[0:1]
        actions = np.array(actions).astype(np.float32) 
        output_traj = {
            'actions': actions, 
            'obs': obs, 
            'states': states, 
            'num_samples': len(actions)
        }
        return output_traj
        

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

    def render(self, mode="rgb_array", height=512, width=512, camera_name=None, **kwargs):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
        """
        if mode =="human":
            return self.env.render(mode=mode, **kwargs)
        if mode == "rgb_array" or mode is None:
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
        saved_pcd_params = None
        for k in obs:
            ob_return[k] = obs[k].copy()
            if k.endswith('image'):
                #TODO: figure out how to do this correctly without hacking
                ob_return[k] = ob_return[k].transpose(2, 0, 1)
            if k.endswith('depth'):
                ob_return[k] = depth_to_rgb(ob_return[k])
                ob_return[k] = ob_return[k].transpose(2, 0, 1)
            if 'pcd' in k and self.postprocesss_visual_obs:
                saved_pcd_params = ob_return[k]
                ob_return[k] = compute_full_pcd(
                    pcd_params=np.expand_dims(ob_return[k], axis=0),
                    **self.pcd_params
                )[0]
            if 'angles' in k and self.pcd_params.get('normalize_joint_angles', False):
                ob_return[k] = normalize_franka_joints(ob_return[k])
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
        success = float(self.env.get_success(self.env.goal_angles)[0])
        if self.num_envs > 1:
            return_dict = dict()
            for split in ["train", "valid", None]:
                key = split if split is not None else 'task'
                if self.split == split:
                    return_dict[key] = success
                else:
                    return_dict[key] = None
        else:
            return_dict = {"task": success}
        return return_dict

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
    
    def get_env_cfg(self):
        """
        Get environment configuration.
        """
        return self.env.cfg
    
    def get_valid_scene(self):
        """
        Get a valid scene for the environment.
        """
        return self.env.get_valid_scene()
    
    def seed(self, seed):
        """
        Seed the environment.
        """
        np.random.seed(seed)

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