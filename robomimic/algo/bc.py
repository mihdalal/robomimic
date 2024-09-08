"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from neural_mp.utils.pcd_utils import decompose_scene_pcd_params_obs_batched
from neural_mp.utils import franka_utils
from neural_mp.utils.constants import FRANKA_LOWER_LIMITS, FRANKA_UPPER_LIMITS

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
import neural_mp.utils.mpinets_loss as loss

# convert limits to torch.Tensor
FRANKA_LOWER_LIMITS = torch.tensor(FRANKA_LOWER_LIMITS, dtype=torch.float32)
FRANKA_UPPER_LIMITS = torch.tensor(FRANKA_UPPER_LIMITS, dtype=torch.float32)

@register_algo_factory_func("bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)

    rnn_enabled = algo_config.rnn.enabled
    # support legacy configs that do not have "transformer" item
    transformer_enabled = ("transformer" in algo_config) and algo_config.transformer.enabled
    
    mpinets_enabled = algo_config.mpinets.enabled if "mpinets" in algo_config else False
    
    act_enabled = algo_config.transformer.act_enabled if "transformer" in algo_config else False
    
    if mpinets_enabled:
        if rnn_enabled:
            algo_class = BC_RNN_MpiNet
            algo_kwargs = {}
        else:
            algo_class = BC_MpiNet
            algo_kwargs = {}
    elif gaussian_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_Gaussian, {}
    elif gmm_enabled:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN_GMM, {}
        elif act_enabled:
            algo_class, algo_kwargs = BC_ACT_GMM, {}
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer_GMM, {}
        else:
            algo_class, algo_kwargs = BC_GMM, {}
    elif vae_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_VAE, {}
    else:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN, {}
        elif act_enabled:
            
            algo_class, algo_kwargs = BC_ACT, {}
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer, {}
        else:
            algo_class, algo_kwargs = BC, {}

    return algo_class, algo_kwargs


class BC(PolicyAlgo):
    """
    Normal BC training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))


    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"]('forward', obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"]('forward', obs_dict, goal_dict=goal_dict)

class BC_MpiNet(BC):
    """
    BC training with MpiNet loss.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fun = loss.CollisionAndBCLossContainer()
    
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            activation=eval(self.algo_config.actor_activation),
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)
        
    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        curr_angles = batch['obs']['current_angles'] # this is normalized
        y_hat = torch.clamp(curr_angles + predictions['actions'], min=-1, max=1) # note this means we need to unnormalize the predictions in the env
        scene_pcd_params = batch["obs"]["saved_params"][:, 14:]
        cuboid_dims, cuboid_centers, cuboid_quats, cylinder_radii, cylinder_heights, cylinder_centers, cylinder_quats, sphere_centers, sphere_radii, M = decompose_scene_pcd_params_obs_batched(scene_pcd_params)
        if cylinder_centers.shape[1] == 0:
            cylinder_centers = torch.zeros_like(cuboid_centers)
            cylinder_radii = torch.zeros_like(cuboid_dims[..., 0:1])
            cylinder_heights = torch.zeros_like(cuboid_dims[..., 0:1])
            cylinder_quats = torch.zeros_like(cuboid_quats)
        if sphere_centers.shape[1] == 0:
            sphere_centers = torch.zeros_like(cuboid_centers)
            sphere_radii = torch.zeros_like(cuboid_dims[..., 0:1])
        collision_loss, point_match_loss = self.loss_fun(
            y_hat,
            cuboid_centers.reshape(-1, M, 3),
            cuboid_dims.reshape(-1, M, 3),
            cuboid_quats.reshape(-1, M, 4),
            cylinder_centers.reshape(-1, M, 3),
            cylinder_radii.reshape(-1, M, 1),
            cylinder_heights.reshape(-1, M, 1),
            cylinder_quats.reshape(-1, M, 4),
            sphere_centers.reshape(-1, M, 3),
            sphere_radii.reshape(-1, M, 1),
            batch['actions'], # assume we are training with absolute actions
        )
        losses = OrderedDict()
        a_target = batch["actions"]
        # convert a_target into delta action
        a_target = a_target - curr_angles
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])
        losses['collision_loss'] = collision_loss
        losses['point_match_loss'] = point_match_loss
        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
            self.algo_config.loss.collision_weight * losses['collision_loss'],
            self.algo_config.loss.point_match_weight * losses['point_match_loss'],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "collision_loss" in info["losses"]:
            log["Collision_Loss"] = info["losses"]["collision_loss"].item()
        if "point_match_loss" in info["losses"]:
            log["Point_Match_Loss"] = info["losses"]["point_match_loss"].item()
        return log

class BC_Gaussian(BC):
    """
    BC training with a Gaussian policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gaussian.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets['policy']('train', 
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
            actions=dists.sample()
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        actions = predictions["actions"]
        a_target = batch["actions"]
        losses = OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        ) 
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])
        losses['log_probs'] = -action_loss
        losses['action_loss'] = action_loss
        return losses 

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        return log


class BC_GMM(BC_Gaussian):
    """
    BC training with a Gaussian Mixture Model policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            mlp_activation=eval(self.algo_config.actor_activation),
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)
    
    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets['policy']('train', 
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])
        
        predictions = OrderedDict(
            log_probs=log_probs,
            actions=dists.sample(),
        )
        
    
    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets['policy']('train', 
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
            actions=dists.sample()
        )
        if self.algo_config.loss.exponential_precision_weight > 0:
            component_means = dists.component_distribution.mean
            component_weights = dists.mixture_distribution.probs
            sq_error = ((component_means - batch['actions'].unsqueeze(1)) ** 2).mean(dim=-1) # shape (B, M)
            additional_loss = torch.exp(-1000*sq_error) + torch.exp(-10000*sq_error) # shape (B, M)
            predictions['exponential_precision_loss'] = -1*(additional_loss * component_weights).sum(dim=-1).mean() # shape (B, M) -> (B) -> scalar
        else:
            predictions['exponential_precision_loss'] = torch.zeros(1, device=self.device)
            
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        exponential_precision_weight = self.algo_config.loss.exponential_precision_weight
        action_loss = -predictions["log_probs"].mean() + exponential_precision_weight * predictions['exponential_precision_loss']
        actions = predictions["actions"]
        a_target = batch["actions"]
        losses = OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        ) 
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])
        losses['log_probs'] = -action_loss
        losses['action_loss'] = action_loss
        if exponential_precision_weight > 0:
            losses['exponential_precision_loss'] = predictions['exponential_precision_loss']
        return losses 

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "exponential_precision_loss" in info["losses"]:
            log["Exponential_Precision_Loss"] = info["losses"]["exponential_precision_loss"].item()
        return log

class BC_VAE(BC):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.VAEActor(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )
        
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
            self.nets["policy"].set_gumbel_temperature(temperature)
        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets['policy']('train', **vae_inputs)
        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],
            kl_loss=vae_outputs["kl_loss"],
            reconstruction_loss=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],
        )
        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if self.algo_config.vae.prior.use_categorical:
            log["Gumbel_Temperature"] = self.nets["policy"].get_gumbel_temperature()
        else:
            log["Encoder_Variance"] = info["predictions"]["encoder_variance"].mean().item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_RNN(BC):
    """
    BC training with an RNN policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        # assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"]('get_rnn_init_state', batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets['policy']('step', 
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0

class BC_RNN_MpiNet(BC_MpiNet):
    """
    BC training with an RNN policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        curr_angles = batch['obs']['current_angles'] # this is normalized
        # reshape seq dim to batch dim
        curr_angles = curr_angles.reshape(-1, curr_angles.shape[-1])
        act_pred = predictions['actions'].reshape(-1, predictions['actions'].shape[-1])
        gt_actions = batch['actions'].reshape(-1, batch['actions'].shape[-1]) # assume we are training with absolute actions
        y_hat = torch.clamp(curr_angles + act_pred, min=-1, max=1) # note this means we need to unnormalize the predictions in the env
        
        scene_pcd_params = batch["obs"]["saved_params"].reshape(-1, batch["obs"]["saved_params"].shape[-1])
        scene_pcd_params = scene_pcd_params[:, 14:]
        cuboid_dims, cuboid_centers, cuboid_quats, cylinder_radii, cylinder_heights, cylinder_centers, cylinder_quats, sphere_centers, sphere_radii, M = decompose_scene_pcd_params_obs_batched(scene_pcd_params)
        if cylinder_centers.shape[1] == 0:
            cylinder_centers = torch.zeros_like(cuboid_centers)
            cylinder_radii = torch.zeros_like(cuboid_dims[..., 0:1])
            cylinder_heights = torch.zeros_like(cuboid_dims[..., 0:1])
            cylinder_quats = torch.zeros_like(cuboid_quats)
        if sphere_centers.shape[1] == 0:
            sphere_centers = torch.zeros_like(cuboid_centers)
            sphere_radii = torch.zeros_like(cuboid_dims[..., 0:1])
        collision_loss, point_match_loss = self.loss_fun(
            y_hat,
            cuboid_centers.reshape(-1, M, 3),
            cuboid_dims.reshape(-1, M, 3),
            cuboid_quats.reshape(-1, M, 4),
            cylinder_centers.reshape(-1, M, 3),
            cylinder_radii.reshape(-1, M, 1),
            cylinder_heights.reshape(-1, M, 1),
            cylinder_quats.reshape(-1, M, 4),
            sphere_centers.reshape(-1, M, 3),
            sphere_radii.reshape(-1, M, 1),
            gt_actions
        )
        losses = OrderedDict()
        a_target = gt_actions
        # convert a_target into delta action
        a_target = a_target - curr_angles
        actions = act_pred
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])
        losses['collision_loss'] = collision_loss
        losses['point_match_loss'] = point_match_loss
        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
            self.algo_config.loss.collision_weight * losses['collision_loss'],
            self.algo_config.loss.point_match_weight * losses['point_match_loss'],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"]('get_rnn_init_state', batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets['policy']('step', 
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0


class BC_RNN_GMM(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)
        self.collision_loss = loss.CollisionLossContainer()

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets['policy']('train', 
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])
        
        predictions = OrderedDict(
            log_probs=log_probs,
            actions=dists.sample(),
        )
        if self.algo_config.loss.exponential_precision_weight > 0:
            component_means = dists.component_distribution.mean
            component_weights = dists.mixture_distribution.probs
            sq_error = ((component_means - batch['actions'].unsqueeze(2)) ** 2).mean(dim=-1) # shape (B, T, M)
            additional_loss = torch.exp(-1000*sq_error) + torch.exp(-10000*sq_error)
            predictions['exponential_precision_loss'] = -1*(additional_loss * component_weights).sum(dim=-1).mean()
        else:
            predictions['exponential_precision_loss'] = torch.zeros(1, device=self.device)
            
        if self.algo_config.loss.collision_weight > 0:
            curr_angles = batch['obs']['current_angles']
            # compute overall mean of gmm:
            means = dists.mean
            curr_angles = curr_angles.reshape(-1, curr_angles.shape[-1])
            act_pred = means.reshape(-1, means.shape[-1])
            y_hat = curr_angles + act_pred
            scene_pcd_params = batch["obs"]["saved_params"].reshape(-1, batch["obs"]["saved_params"].shape[-1])
            scene_pcd_params = scene_pcd_params[:, 14:]
            cuboid_dims, cuboid_centers, cuboid_quats, cylinder_radii, cylinder_heights, cylinder_centers, cylinder_quats, sphere_centers, sphere_radii, M = decompose_scene_pcd_params_obs_batched(scene_pcd_params)
            collision_loss, sdf_collisions = self.collision_loss(
                y_hat,
                cuboid_centers.reshape(-1, M, 3),
                cuboid_dims.reshape(-1, M, 3),
                cuboid_quats.reshape(-1, M, 4),
                cylinder_centers.reshape(-1, M, 3),
                cylinder_radii.reshape(-1, M, 1),
                cylinder_heights.reshape(-1, M, 1),
                cylinder_quats.reshape(-1, M, 4),
                sphere_centers.reshape(-1, M, 3),
                sphere_radii.reshape(-1, M, 1),
                reduction=self.algo_config.loss.collision_loss_params['reduction'],
                hinge_loss=self.algo_config.loss.collision_loss_params['hinge_loss'],
                margin=self.algo_config.loss.collision_loss_params['margin'],
                smooth_sdf_loss=self.algo_config.loss.collision_loss_params['smooth_sdf_loss'],
                compute_loss_on_penetrations_only=self.algo_config.loss.collision_loss_params['compute_loss_on_penetrations_only'],
            ) 
            predictions['collision_loss'] = collision_loss
            predictions['sdf_collision_rate'] = sdf_collisions.mean()
        else:
            predictions['collision_loss'] = torch.zeros(1, device=self.device)
            predictions['sdf_collision_rate'] = torch.zeros(1, device=self.device)
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        # loss is just negative log-likelihood of action targets
        collision_weight = self.algo_config.loss.collision_weight
        exponential_precision_weight = self.algo_config.loss.exponential_precision_weight
        action_loss = -predictions["log_probs"].mean() + collision_weight * predictions['collision_loss'] + exponential_precision_weight * predictions['exponential_precision_loss']
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])
        losses['log_probs'] = -action_loss
        losses['action_loss'] = action_loss
        if self.algo_config.loss.collision_weight > 0:
            losses['collision_loss'] = predictions['collision_loss']
            losses['sdf_collision_rate'] = predictions['sdf_collision_rate']
        if exponential_precision_weight > 0:
            losses['exponential_precision_loss'] = predictions['exponential_precision_loss']
        return losses

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "collision_loss" in info["losses"]:
            log["Collision_Loss"] = info["losses"]["collision_loss"].item()
        if "sdf_collision_rate" in info["losses"]:
            log["SDF_Collision_Rate"] = info["losses"]["sdf_collision_rate"].item()
        if "exponential_precision_loss" in info["losses"]:
            log["Exponential_Precision_Loss"] = info["losses"]["exponential_precision_loss"].item()
        return log


class BC_Transformer(BC):
    """
    BC training with a Transformer policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)
        
    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        h = self.context_length
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        if self.supervise_all_steps:
            # supervision on entire sequence (instead of just current timestep)
            input_batch["actions"] = batch["actions"][:, :h, :]
        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h-1, :]

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch

    def _forward_training(self, batch, epoch=None):
        """
        Internal helper function for BC_Transformer algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        predictions = OrderedDict()
        predictions["actions"] = self.nets["policy"](obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"])
        if not self.supervise_all_steps:
            # only supervise final timestep
            predictions["actions"] = predictions["actions"][:, -1, :]
        return predictions

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        # assert not self.nets.training

        return self.nets["policy"]('forward', obs_dict, actions=None, goal_dict=goal_dict)[:, -1, :]

class BC_ACT(BC):
    """
    BC training with a ACT policy.
    """
    def __init__(self, *args, **kwargs):
        super(BC_ACT, self).__init__(*args, **kwargs)
        self.inference_num_open_loop_steps = 4
        #todo - make part of config instead of hardcoding
        self.inference_action_cache = []

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerActorACTNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)
        
    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        h = self.context_length
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        # supervision on entire sequence (instead of just current timestep)
        input_batch["actions"] = batch["actions"][:, :h, :]

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch

    def _forward_training(self, batch, epoch=None):
        """
        Internal helper function for BC_Transformer algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        predictions = OrderedDict()
        predictions["actions"] = self.nets["policy"]('forward', 
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
            actions=batch["actions"],
        )
        return predictions
    
    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"][:, len(batch['obs'])-1:] # skip first len(batch['obs'])-1 timesteps since they are not used to predict the actions
        mask = (a_target == torch.zeros(a_target.shape[-1], device=self.device)).all(dim=-1) 
        
        # note this is because the actions can be padded to complete zeros in the dataset (if we sample at the end of the sequence)
        # we throw those actions out of the loss
        a_target = a_target[~mask] # B*T x act_dim
        actions = actions[~mask] # B*T x act_dim
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training   
        if len(self.inference_action_cache) == 0:
            # need to iteratively predict the actions
            bs = obs_dict['current_angles'].shape[0]
            assert bs == 1, "Only supports batch size 1 for now"
            obs_dict = {k: obs_dict[k].unsqueeze(1) for k in obs_dict}
            obs_shift = self.algo_config.transformer.context_length - self.global_config.train.seq_length
            act_history = []
            for i in range(self.inference_num_open_loop_steps):
               
                zero_act = torch.zeros((bs, self.global_config.train.seq_length-i, self.ac_dim), device=self.device)
                if len(act_history) == 0:
                    acts = zero_act
                else:
                    _hist = torch.vstack(act_history).unsqueeze(0)
                    acts = torch.cat([_hist, zero_act], axis = 1)
                
                out = self.nets["policy"]('forward', obs_dict, actions=acts, goal_dict=goal_dict)
                act_history.append(out[:, obs_shift+i, :])

            self.inference_action_cache = torch.vstack(act_history) 
        
        cur_act = self.inference_action_cache[0]
        self.inference_action_cache = self.inference_action_cache[1:]
        return cur_act.unsqueeze(0)

class BC_ACT_GMM(BC_ACT):
    """
    BC training with a Transformer GMM ACT policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerGMMACTActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch, epoch=None):
        """
        Modify from super class to support GMM training.
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        # TensorUtils.assert_size_at_dim(
        #     batch["obs"], 
        #     size=(self.context_length), 
        #     dim=1, 
        #     msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        # )
        
        dists = self.nets['policy']('train', 
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
            low_noise_eval=False,
            actions=batch["actions"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        a_target = batch["actions"]
        mask = (a_target == torch.zeros(a_target.shape[-1], device=self.device)).all(dim=-1) 
        log_probs = dists.log_prob(batch["actions"])[~mask]

        predictions = OrderedDict(
            log_probs=log_probs,
            actions=dists.sample(),
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC_Transformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        a_target = batch["actions"]
        actions = predictions["actions"]
        a_target = a_target[:, 0] # just measure loss on the action we would actually predict
        actions = actions[:, 0] # just measure loss on the action we would actually predict
        # loss is just negative log-likelihood of action targets
        l2_loss = nn.MSELoss()(actions, a_target)
        l1_loss = nn.SmoothL1Loss()(actions, a_target)
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        actions = torch.zeros((obs_dict['current_angles'].shape[0], self.global_config.train.seq_length, self.ac_dim), device=self.device)
        # add time dimension to obs_dict
        obs_dict = {k: obs_dict[k].unsqueeze(1) for k in obs_dict}
        return self.nets["policy"]('forward', obs_dict, actions=actions, goal_dict=goal_dict)[:, 0, :] # first timestep prediction


class BC_Transformer_GMM(BC_Transformer):
    """
    BC training with a Transformer GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch, epoch=None):
        """
        Modify from super class to support GMM training.
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        dists = self.nets['policy']('train', 
            obs_dict=batch["obs"], 
            actions=None,
            goal_dict=batch["goal_obs"],
            low_noise_eval=False,
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]

        if not self.supervise_all_steps:
            # only use final timestep prediction by making a new distribution with only final timestep.
            # This essentially does `dists = dists[:, -1]`
            component_distribution = D.Normal(
                loc=dists.component_distribution.base_dist.loc[:, -1],
                scale=dists.component_distribution.base_dist.scale[:, -1],
            )
            component_distribution = D.Independent(component_distribution, 1)
            mixture_distribution = D.Categorical(logits=dists.mixture_distribution.logits[:, -1])
            dists = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )

        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC_Transformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
