"""
A useful script for generating json files and shell scripts for conducting parameter scans.
The script takes a path to a base json file as an argument and a shell file name.
It generates a set of new json files in the same folder as the base json file, and 
a shell file script that contains commands to run for each experiment.

Instructions:

(1) Start with a base json that specifies a complete set of parameters for a single 
    run. This only needs to include parameters you want to sweep over, and parameters
    that are different from the defaults. You can set this file path by either
    passing it as an argument (e.g. --config /path/to/base.json) or by directly
    setting the config file in @make_generator. The new experiment jsons will be put
    into the same directory as the base json.

(2) Decide on what json parameters you would like to sweep over, and fill those in as 
    keys in @make_generator below, taking note of the hierarchical key
    formatting using "/" or ".". Fill in corresponding values for each - these will
    be used in creating the experiment names, and for determining the range
    of values to sweep. Parameters that should be sweeped together should
    be assigned the same group number.

(3) Set the output script name by either passing it as an argument (e.g. --script /path/to/script.sh)
    or by directly setting the script file in @make_generator. The script to run all experiments
    will be created at the specified path.

Args:
    config (str): path to a base config json file that will be modified to generate config jsons.
        The jsons will be generated in the same folder as this file.

    script (str): path to output script that contains commands to run the generated training runs

Example usage:

    # assumes that /tmp/gen_configs/base.json has already been created (see quickstart section of docs for an example)
    python hyperparam_helper.py --config /tmp/gen_configs/base.json --script /tmp/gen_configs/out.sh
"""
import argparse
import os
import shutil

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils


def make_generator(config_file, script_file, wandb_proj_name, output_dir):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file, wandb_proj_name=wandb_proj_name,
    )

    generator.add_param(
        key='train.output_dir',
        name="",
        group=0,
        values=[output_dir],
    )

    generator.add_param(
        key="train.data",
        name="",
        group=1,
        values=[
            "/home/mdalal/research/neural_mp/neural_mp/datasets/table_bins_10K_pcd_params_obs_delta_True_single_obs_task_oriented_1.0_save_planner.hdf5",
        ],
    )

    generator.add_param(
        key="experiment.dagger.dagger_traj_filter",
        name="dtf",
        group=2,
        values=[
            'all',
            'collide',
            'fail',
            'collide_or_fail'
        ],
    )
    
    generator.add_param(
        key="experiment.dagger.data_mode",
        name="dm",
        group=3,
        values=[
            'all',
            'online_data_only',
            'latest_data_only'
        ],
    )
    
    generator.add_param(
        key="experiment.dagger.resampling_strategy",
        name="rs",
        group=4,
        values=[
            # 'all',
            'collision'
        ],
    )

    generator.add_param(
        key="experiment.dagger.num_steps_to_keep_before_collision",
        name="nstkbc",
        group=5,
        values=[
            1, 5, 10, 25
        ],
    )

    return generator


def main(args):
    dirname = os.path.dirname(args.base_config)
    os.makedirs(args.exp_dir, exist_ok=True)
    config_path = os.path.join(args.exp_dir, os.path.basename(args.base_config))
    shutil.copyfile(args.base_config, config_path)

    exp_dir_suffix = os.path.basename(args.exp_dir)
    wandb_proj_name = exp_dir_suffix
    output_dir = os.path.join("logs/mp/trained_models", exp_dir_suffix)
    # set output dir in config
    config = HyperparamUtils.load_json(config_path)
    config["experiment"]['name'] = wandb_proj_name
    HyperparamUtils.save_json(config, config_path)
    # make config generator
    generator = make_generator(config_file=config_path, script_file=args.script, wandb_proj_name=wandb_proj_name, output_dir=output_dir)

    # generate jsons and script
    import neural_mp
    sif_path = os.path.join(neural_mp.__file__[:-len("neural_mp/__init__.py")], "containers/neural_mp_zsh.sif")

    generator.generate_matrix_commands(sif_path, args.checkpoint_path, args.ddp, args.num_gpus)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to base json config - will override any defaults.
    parser.add_argument(
        "--base_config",
        type=str,
        help="path to base config json that will be modified to generate jsons",
    )

    parser.add_argument(
        "--exp_dir",
        type=str,
        help="path to folder where the jsons will be generated.",
    )

    # Script name to generate - will override any defaults
    parser.add_argument(
        "--script",
        type=str,
        help="path to output script that contains commands to run the generated training runs",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="path to checkpoint to start from",
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

    args = parser.parse_args()
    main(args)
