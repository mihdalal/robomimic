"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import setup

import robomimic
from robomimic.models.base_nets import DDPModelWrapper
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import h5py

class SubprocVecEnvWrapper(SubprocVecEnv):
    def env_method_pass_idx(self, method_name: str, *method_args, indices = None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for idx, remote in zip(indices, target_remotes):
            method_kwargs['env_idx'] = idx
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

class DummyVecEnvWrapper(DummyVecEnv):
    def env_method_pass_idx(self, method_name: str, *method_args, indices = None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        out = []
        for idx, env in zip(indices, target_envs):
            method_kwargs['env_idx'] = idx
            out.append(getattr(env, method_name)(*method_args, **method_kwargs))
        return out

def make_env(env_meta, use_images, render_video, pcd_params, mpinets_enabled, dataset_path, config_path):
    env_meta['env_kwargs']['dataset_path'] = dataset_path
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=render_video,
        use_image_obs=use_images,
        pcd_params=pcd_params,
        mpinets_enabled=mpinets_enabled,
    )
    ext_cfg = json.load(open(config_path, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
    env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment warpper, if applicable
    return env

def train(config, device, ckpt_path=None, ckpt_dict=None, output_dir=None, start_from_checkpoint=False, rank=0, world_size=1, ddp=False, dataset_path=None, additional_datasets=()):
    """
    Train a model using the algorithm.
    """
    os.environ['WANDB_API_KEY'] = "010fcba9b0530d8e86f54a8e7e68725a06be7dba"
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    torch.set_num_threads(1)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    if rank == 0:
        if ckpt_dict and not start_from_checkpoint:
            print("loading dirs from checkpoint")
            log_dir, ckpt_dir, video_dir = ckpt_dict["log_dir"], ckpt_dict["ckpt_dir"], ckpt_dict["video_dir"]
            epoch = ckpt_dict["epoch"]
        elif output_dir is not None and output_dir != 'None':
            print("getting dirs from output_dir")
            log_dir = os.path.join(output_dir, "logs")
            ckpt_dir = os.path.join(output_dir, "models")
            video_dir = os.path.join(output_dir, "videos")
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
            epoch = 1
        else:
            print("getting dirs from config")
            log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config, auto_remove_exp_dir=start_from_checkpoint or ddp)
            epoch = 1
    else:
        if ckpt_dict and not start_from_checkpoint:
            epoch = ckpt_dict["epoch"]
        else:
            epoch = 1

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(dataset_path)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    if rank == 0:
        # create environment
        envs = OrderedDict()
        if config.experiment.rollout.enabled:
            # create environments for validation runs
            env_names = [env_meta["env_name"]]

            if config.experiment.additional_envs is not None:
                for name in config.experiment.additional_envs:
                    env_names.append(name)

            for env_name in env_names:
                pcd_params = config.experiment.pcd_params.to_dict()
                mpinets_enabled = config.algo.mpinets.enabled
                render_video = config.experiment.render_video
                # save config 
                config_path = os.path.join(log_dir, 'config.json')
                json.dump(config, open(config_path, 'w'), indent=4)
                env_meta['env_kwargs']['cfg']['task']['setup_cameras'] = True # set up cameras for rendering
                if config.experiment.num_envs > 1:
                    env_fns = []
                    for env_idx in range(config.experiment.num_envs):
                        env_fn = lambda: make_env(env_meta, shape_meta['use_images'], render_video, pcd_params, mpinets_enabled, dataset_path, config_path)
                        env_fns.append(env_fn)
                    env = SubprocVecEnvWrapper(env_fns, start_method='fork')
                    for env_idx in range(config.experiment.num_envs):
                        if env_idx < 5:
                            split = 'train'
                            num_split_envs = 5 # because each split has 5 envs (for train and val)
                        elif env_idx < 10:
                            split = 'valid'
                            num_split_envs = 5 # because each split has 5 envs (for train and val)
                        else:
                            split = None
                            num_split_envs = config.experiment.num_envs - 10
                        env.env_method_pass_idx("set_env_specific_params", split, num_split_envs, indices=[env_idx])
                else:
                    env = DummyVecEnvWrapper([lambda: make_env(env_meta, shape_meta['use_images'], render_video, pcd_params, mpinets_enabled, dataset_path, config_path)])
                    for env_idx in range(config.experiment.num_envs):
                        split = 'valid'
                        num_split_envs = 1 
                        env.env_method_pass_idx("set_env_specific_params", split, num_split_envs, indices=[env_idx])
                envs[env_name] = env
                print(envs[env_name])

    print("")

    if rank == 0:
        # setup for a new training run
        data_logger = DataLogger(
            log_dir,
            config,
            log_tb=config.experiment.logging.log_tb,
            log_wandb=config.experiment.logging.log_wandb,
        )
    # restore policy
    if ckpt_path is not None and ckpt_path != 'None':
        model, _ = FileUtils.model_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True, ddp=ddp, rank=rank, world_size=world_size, config=config)
    else:
        model = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device=device,
        )
        model.nets['policy'] = DDPModelWrapper(model.nets['policy'])
        if ddp:
            setup(rank, world_size)
            model.nets['policy'] = nn.parallel.DistributedDataParallel(model.nets['policy'], device_ids=[rank])
    
    if rank == 0:
        # save the config as a json file
        with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
            json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    def format_parameters(num):
        if num < 1e6:
            return f"{num / 1e3:.2f}K"  # Thousands
        elif num < 1e9:
            return f"{num / 1e6:.2f}M"  # Millions
        elif num < 1e12:
            return f"{num / 1e9:.2f}G"  # Billions
        else:
            return f"{num / 1e12:.2f}T"  # Trillions
    if ddp:
        num_policy_params =sum(p.numel() for p in model.nets['policy'].module.parameters())
        num_enc_params = sum(p.numel() for p in model.nets['policy'].module.model.nets['encoder'].parameters())
    else:
        num_policy_params =sum(p.numel() for p in model.nets['policy'].parameters())
        try:
            num_enc_params = sum(p.numel() for p in model.nets['policy'].model.nets['encoder'].parameters())
        except:
            # this is for diffusion policy
            num_enc_params = sum(p.numel() for p in model.nets['policy'].model['obs_encoder'].nets['obs'].parameters())
    print("Policy params: ", format_parameters(num_policy_params))
    print("encoder params: ", format_parameters(num_enc_params))

    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    if additional_datasets:
        trainsets = [trainset]
        validsets = [validset]
        for additional_dataset in additional_datasets:
            # temporarily set the dataset path in the config
            config.unlock()
            config.train.data = additional_dataset
            config.lock()
            additional_trainset, additional_validset = TrainUtils.load_data_for_training(
                config, obs_keys=shape_meta["all_obs_keys"])
            trainsets.append(additional_trainset)
            validsets.append(additional_validset)
        config.unlock()
        config.train.data = dataset_path # reset the dataset path
        config.lock()
        trainset = torch.utils.data.ConcatDataset(trainsets)
        validset = torch.utils.data.ConcatDataset(validsets)
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    else:
        train_sampler = None
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True if config.train.num_data_workers > 0 else False,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = config.train.num_data_workers
        if ddp:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(validset,
                                                                        num_replicas=world_size,
                                                                        rank=rank)
        else:
            valid_sampler = None
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if config.train.num_data_workers > 0 else False,
        )
    else:
        valid_loader = None

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    if rank == 0:
        # main training loop
        best_valid_loss = None
        best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
        best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
        last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    def handler(signum, frame):
        print('Signal handler called with signal', signum)
        print("Saving checkpoint before exiting...")
        if rank == 0:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, "model_latest.pth"),
                obs_normalization_stats=obs_normalization_stats,
                log_dir=log_dir,
                ckpt_dir=ckpt_dir,
                video_dir=video_dir,
                epoch=epoch,
            )
        exit()
    import signal 
    signal.signal(signal.SIGUSR1, handler)
    if ddp:
        # this is used for doing all-reduce on logs across ddp processes
        group = dist.new_group(list(range(world_size)))
    for epoch in range(epoch, config.train.num_epochs + 1): # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        if rank == 0:
            should_save_ckpt = False
            if config.experiment.save.enabled:
                time_check = (config.experiment.save.every_n_seconds is not None) and \
                    (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
                epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                    (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
                epoch_list_check = (epoch in config.experiment.save.epochs)
                should_save_ckpt = (time_check or epoch_check or epoch_list_check)
            ckpt_reason = None
            if should_save_ckpt:
                last_ckpt_time = time.time()
                ckpt_reason = "time"

            print("Train Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
                else:
                    if ddp:
                        tensor = torch.tensor([v])
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
                        data_logger.record("Train/{}".format(k), tensor[0].item() / world_size, epoch)
                    else:
                        data_logger.record("Train/{}".format(k), v, epoch)
        else:
            for k, v in step_log.items():
                if not k.startswith("Time_"):
                    tensor = torch.tensor([v])
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                try:
                    low_noise_eval = model.nets['policy'].model.low_noise_eval
                    model.nets['policy'].model.low_noise_eval = False
                except:
                    low_noise_eval = None
                    pass
                step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
                try:
                    model.nets['policy'].model.low_noise_eval = low_noise_eval
                except:
                    pass
            if rank == 0:
                for k, v in step_log.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                    else:
                        if ddp:
                            tensor = torch.tensor([v])
                            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
                            data_logger.record("Valid/{}".format(k), tensor[0].item() / world_size, epoch)
                        else:
                            data_logger.record("Valid/{}".format(k), v, epoch)

                print("Validation Epoch {}".format(epoch))
                print(json.dumps(step_log, sort_keys=True, indent=4))

                # save checkpoint if achieve new best validation loss
                valid_check = "Loss" in step_log
                if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                    best_valid_loss = step_log["Loss"]
                    if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                        epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                        should_save_ckpt = True
                        ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason
            else:
                for k, v in step_log.items():
                    if not k.startswith("Time_"):
                        tensor = torch.tensor([v])
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        if rank == 0:
            if config.experiment.dagger.enabled:
                # wrap model as a RolloutPolicy to prepare for rollouts
                rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
                dagger_data_dir = os.path.join(log_dir, "dagger_data")
                os.makedirs(dagger_data_dir, exist_ok=True)
                dataset_path = os.path.join(dagger_data_dir, f"online_dataset_{epoch}.hdf5")
                data_writer = h5py.File(dataset_path, "w")
                online_epoch = epoch // config.experiment.dagger.online_epoch_rate
                data = TrainUtils.collect_online_dataset(
                    policy=rollout_model,
                    envs=envs,
                    horizon=config.experiment.rollout.horizon,
                    use_goals=config.use_goals,
                    num_episodes=config.experiment.dagger.num_episodes,
                    render=False,
                    terminate_on_success=config.experiment.rollout.terminate_on_success,
                    online_epoch=online_epoch,
                    resampling_strategy=config.experiment.dagger.resampling_strategy,
                    num_trajs_to_relabel=config.experiment.dagger.num_trajs_to_relabel,
                    data_writer=data_writer,
                )
                trainset.update_demo_info(
                    list(data.keys()), online_epoch, data, hdf5_file=data_writer
                )
                # reload the dataloader
                train_loader = DataLoader(
                    dataset=trainset,
                    sampler=train_sampler,
                    batch_size=config.train.batch_size,
                    shuffle=(train_sampler is None),
                    num_workers=config.train.num_data_workers,
                    drop_last=True,
                    pin_memory=True,
                    persistent_workers=True if config.train.num_data_workers > 0 else False,
                )
                data_writer.close()
                
            video_paths = None
            rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
            if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

                # wrap model as a RolloutPolicy to prepare for rollouts
                rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)

                num_episodes = config.experiment.rollout.n
                all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                    policy=rollout_model,
                    envs=envs,
                    horizon=config.experiment.rollout.horizon,
                    use_goals=config.use_goals,
                    num_episodes=num_episodes,
                    render=False,
                    video_dir=video_dir if config.experiment.render_video else None,
                    epoch=epoch,
                    video_skip=config.experiment.get("video_skip", 5),
                    terminate_on_success=config.experiment.rollout.terminate_on_success,
                )

                for k in video_paths.keys():
                    data_logger.record("Video/{}".format(k), video_paths[k], epoch, data_type="video")
                # summarize results from rollouts to tensorboard and terminal
                for env_name in all_rollout_logs:
                    rollout_logs = all_rollout_logs[env_name]
                    for k, v in rollout_logs.items():
                        if k.startswith("Time_"):
                            data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                        else:
                            data_logger.record("Rollout_{}/{}".format(env_name, k), v, epoch, log_stats=True)

                    print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                    print('Env: {}'.format(env_name))
                    print(json.dumps(rollout_logs, sort_keys=True, indent=4))

                # checkpoint and video saving logic
                updated_stats = TrainUtils.should_save_from_rollout_logs(
                    all_rollout_logs=all_rollout_logs,
                    best_return=best_return,
                    best_success_rate=best_success_rate,
                    epoch_ckpt_name=epoch_ckpt_name,
                    save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                    save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
                )
                best_return = updated_stats["best_return"]
                best_success_rate = updated_stats["best_success_rate"]
                epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
                should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
                if updated_stats["ckpt_reason"] is not None:
                    ckpt_reason = updated_stats["ckpt_reason"]

            # Only keep saved videos if the ckpt should be saved (but not because of validation score)
            should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
            if video_paths is not None and not should_save_video:
                for env_name in video_paths:
                    os.remove(video_paths[env_name])

            # Save model checkpoints based on conditions (success rate, validation loss, etc)
            if should_save_ckpt:
                TrainUtils.save_model(
                    model=model,
                    config=config,
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                    obs_normalization_stats=obs_normalization_stats,
                    log_dir=log_dir,
                    ckpt_dir=ckpt_dir,
                    video_dir=video_dir,
                    epoch=epoch,
                )

            # Finally, log memory usage in MB
            process = psutil.Process(os.getpid())
            mem_usage = int(process.memory_info().rss / 1000000)
            data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
            print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))
    if rank == 0:
        # terminate logging
        data_logger.close()


def main(rank, args):

    # set torch backend
    import torch
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    import torch._dynamo                                                    
    torch._dynamo.config.suppress_errors = True
    
    # relative path to agent
    ckpt_path = args.agent
    ckpt_dict = None

    if ckpt_path is not None and ckpt_path != 'None':
        ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=ckpt_path)
        if args.start_from_checkpoint:
            # if starting from checkpoint, use input config to override the checkpoint config
            ext_cfg = json.load(open(args.config, 'r'))
            config = config_factory(ext_cfg["algo_name"])
            # update config with external json - this will throw errors if
            # the external config has keys not present in the base algo config
            with config.values_unlocked():
                config.update(ext_cfg)
        else:
            config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
            config.unlock()
    elif args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.datasets is not None:
        config.train.data = args.datasets[0] # assume this is primary dataset

    if args.name is not None:
        config.experiment.name = args.name
    
    if args.slurm:
        # check if the dataset is in /scratch/mdalal/dataset_name
        dataset_name = config.train.data.split('/')[-1]
        os.makedirs("/scratch/mdalal", exist_ok=True)
        if not os.path.exists(f"/scratch/mdalal/{dataset_name}"):
            # copy the dataset to /scratch/mdalal
            shutil.copyfile(config.train.data, f"/scratch/mdalal/{dataset_name}")
        dataset_path = f"/scratch/mdalal/{dataset_name}"
    else:
        dataset_path = config.train.data

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda, rank=rank)
    torch.cuda.set_device(device)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"

    try:
        train(config, device=device, ckpt_path=ckpt_path, ckpt_dict=ckpt_dict, 
              output_dir=args.output_dir, start_from_checkpoint=args.start_from_checkpoint, 
              rank=rank, world_size=args.num_gpus, ddp=args.ddp, dataset_path=dataset_path, additional_datasets=args.datasets[1:])
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="path to saved checkpoint pth file",
    )

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--datasets",
        nargs='+',
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    # output dir
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="(optional) if provided, override the output directory defined in the config",
    )

    parser.add_argument(
        "--start_from_checkpoint",
        action='store_true',
        help="set this flag to start from checkpoint (not resume)",
    )

    parser.add_argument(
        "--ddp", 
        action='store_true',
        help="set this flag to use distributed data parallel"
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="number of gpus to use for distributed data parallel"
    )
    
    parser.add_argument(
        "--slurm",
        action='store_true',
        help="set this flag to run on slurm"
    )
    args = parser.parse_args()

    if args.ddp:
        mp.spawn(main, nprocs=args.num_gpus, args=(args,))
    else:
        main(0, args)

