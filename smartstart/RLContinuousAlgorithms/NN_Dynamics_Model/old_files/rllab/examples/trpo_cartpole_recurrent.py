


from rllab.rllab.algos.trpo import TRPO
from rllab.rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.rllab.envs.normalized_env import normalize
from rllab.rllab.misc.instrument import stub, run_experiment_lite
from rllab.rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

stub(globals())

env = normalize(CartpoleEnv())

policy = GaussianGRUPolicy(
    env_spec=env.spec,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=10,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)
run_experiment_lite(
    algo.train(),
    n_parallel=1,
    seed=1,
)
