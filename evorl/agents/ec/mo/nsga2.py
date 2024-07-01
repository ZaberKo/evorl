import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import evox.algorithms
from evox.operators import non_dominated_sort

from evorl.utils.ec_utils import ParamVectorSpec
from evorl.utils.jax_utils import jit_method
from evorl.workflows import ECWorkflow
from evorl.envs import create_wrapped_brax_env
from evorl.ec import MultiObjectiveBraxProblem
from evorl.metrics import EvaluateMetric
from evorl.distributed import tree_unpmap
from evorl.evaluator import Evaluator
from evorl.types import State
from ..ec import DeterministicECAgent


class NSGA2Workflow(ECWorkflow):
    @classmethod
    def name(cls):
        return "NSGA2"

    def __init__(self, config: DictConfig):
        env = create_wrapped_brax_env(
            config.env.env_name,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset=False,
        )

        agent = DeterministicECAgent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,  # use linear model
            normalize_obs=False
        )

        problem = MultiObjectiveBraxProblem(
            agent=agent,
            env=env,
            num_episodes=config.episodes,
            max_episode_steps=config.env.max_episode_steps,
            discount=config.discount,
            metric_names=config.obj_names,
            flatten_objectives=True
        )

        # dummy agent_state
        agent_key = jax.random.PRNGKey(config.seed)
        agent_state = agent.init(agent_key)
        param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

        algorithm = evox.algorithms.NSGA2(
            lb=jnp.full((param_vec_spec.vec_size,),
                        fill_value=config.agent_network.lb),
            ub=jnp.full((param_vec_spec.vec_size,),
                        fill_value=config.agent_network.ub),
            n_objs=len(config.obj_names),
            pop_size=config.pop_size,
        )

        def _candidate_transform(flat_cand):
            cand = param_vec_spec.to_tree(flat_cand)
            params = agent_state.params.replace(policy_params=cand)
            return agent_state.replace(params=params)

        super().__init__(
            config=config,
            algorithm=algorithm,
            problem=problem,
            opt_direction=config.opt_directions,
            candidate_transforms=[jax.vmap(_candidate_transform)],
        )

    def learn(self, state: State) -> State:
        start_iteration = tree_unpmap(
            state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, self.config.num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            train_metrics = tree_unpmap(
                train_metrics, axis_name=self.pmap_axis_name)
            workflow_metrics = tree_unpmap(
                workflow_metrics, self.pmap_axis_name)
            self.recorder.write(workflow_metrics.to_local_dict(), i)

            cpu_device = jax.devices('cpu')[0]
            with jax.default_device(cpu_device):
                fitnesses = train_metrics.objectives*self._workflow.opt_direction
                pf_rank = non_dominated_sort(fitnesses, 'scan')
                pf_objectives = train_metrics.objectives[pf_rank == 0]
                _train_metrics = train_metrics.to_local_dict()
                _train_metrics['pf_objectives'] = pf_objectives.tolist()
                _train_metrics['num_pf'] = pf_objectives.shape[0]

            self.recorder.write(_train_metrics, i)
