from rllab.rllab.algos.trpo import TRPO
from rllab.rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env import PointEnv
from rllab.rllab.envs.normalized_env import normalize
from rllab.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(PointEnv())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()
