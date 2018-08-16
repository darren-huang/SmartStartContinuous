from rllab.rllab.algos.npo import NPO
from rllab.rllab.core.serializable import Serializable
from rllab.rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer


class PPO(NPO, Serializable):
    """
    Penalized Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        super(PPO, self).__init__(optimizer=optimizer, **kwargs)
