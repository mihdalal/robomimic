from manipgen_robomimic.algo.algo import register_algo_factory_func, algo_name_to_factory_func, algo_factory, Algo, PolicyAlgo, ValueAlgo, PlannerAlgo, HierarchicalAlgo, RolloutPolicy

# note: these imports are needed to register these classes in the global algo registry
from manipgen_robomimic.algo.bc import BC, BC_Gaussian, BC_GMM, BC_VAE, BC_RNN, BC_RNN_GMM
from manipgen_robomimic.algo.bcq import BCQ, BCQ_GMM, BCQ_Distributional
from manipgen_robomimic.algo.cql import CQL
from manipgen_robomimic.algo.iql import IQL
from manipgen_robomimic.algo.gl import GL, GL_VAE, ValuePlanner
from manipgen_robomimic.algo.hbc import HBC
from manipgen_robomimic.algo.iris import IRIS
from manipgen_robomimic.algo.td3_bc import TD3_BC
