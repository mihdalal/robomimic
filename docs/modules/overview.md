# Overview

<p align="center">
  <img width="100.0%" src="../images/module_overview.png">
 </p>

The **manipgen_robomimic** framework consists of several modular components that interact to train and evaluate a policy:
- **Experiment config**: a config object defines all settings for a training run
- **Data**: an hdf5 dataset is loaded into a dataloader, which provides minibatches to the algorithm
- **Training**: an algorithm object trains a set of models (including the policy)
- **Evaluation**: the policy is evaluated in the environment by conducting a set of rollouts
- **Logging**: experiment statistics, model checkpoints, and videos are saved to disk

These modules are encapsulated by the manipgen_robomimic directory structure:

- `examples`: examples to better understand modular components in the codebase
- [`manipgen_robomimic/algo`](./algorithms.html): policy learning algorithm implementations
- [`manipgen_robomimic/config`](./configs.html): default algorithm configs
- [`manipgen_robomimic/envs`](./environments.html): wrappers for environments, used during evaluation rollouts
- `manipgen_robomimic/exps/templates`: config templates for experiments
- [`manipgen_robomimic/models`](./models.html): network implementations
- `manipgen_robomimic/scripts`: main repository scripts 
- `manipgen_robomimic/utils`: a collection of utilities, including the [SequenceDataset](./dataset.html) class to load datasets, and [TensorUtils](../tutorials/tensor_collections.html#tensorutils) to work with nested tensor dictionaries 

