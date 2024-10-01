from sp_robomimic.algo.algo import register_algo_factory_func, algo_name_to_factory_func, algo_factory, Algo, PolicyAlgo, ValueAlgo, PlannerAlgo, HierarchicalAlgo, RolloutPolicy

# note: these imports are needed to register these classes in the global algo registry
from sp_robomimic.algo.bc import BC, BC_Gaussian, BC_GMM, BC_VAE, BC_RNN, BC_RNN_GMM
from sp_robomimic.algo.bcq import BCQ, BCQ_GMM, BCQ_Distributional
from sp_robomimic.algo.cql import CQL
from sp_robomimic.algo.iql import IQL
from sp_robomimic.algo.gl import GL, GL_VAE, ValuePlanner
from sp_robomimic.algo.hbc import HBC
from sp_robomimic.algo.iris import IRIS
from sp_robomimic.algo.td3_bc import TD3_BC
