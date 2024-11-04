import logging

import chex
import optax
from optax.schedules import InjectStatefulHyperparamsState

from evorl.types import PyTreeDict, State

from .pbt_base import PBTWorkflowTemplate
from .pbt_utils import deepcopy_opt_state, log_uniform_init


logger = logging.getLogger(__name__)


class PBTWorkflow(PBTWorkflowTemplate):
    """
    A minimal Example of PBT that tunes the lr of PPO.
    """

    @classmethod
    def name(cls):
        return "PBT"

    def _customize_optimizer(self) -> None:
        """
        Customize the target workflow's optimizer
        """
        self.workflow.optimizer = optax.inject_hyperparams(
            optax.adam, static_args=("b1", "b2", "eps", "eps_root")
        )(learning_rate=self.config.search_space.lr.low)

    def _setup_pop(self, key: chex.PRNGKey) -> chex.ArrayTree:
        pop = PyTreeDict(
            lr=log_uniform_init(self.config.search_space.lr, key, self.config.pop_size)
        )

        return pop

    def apply_hyperparams_to_workflow_state(
        self, workflow_state: State, hyperparams: PyTreeDict[str, chex.Numeric]
    ) -> State:
        """
        Note1: InjectStatefulHyperparamsState is NamedTuple, which is not immutable.
        Note2: try to avoid deepcopy unnessary state
        """
        opt_state = workflow_state.opt_state
        assert isinstance(opt_state, InjectStatefulHyperparamsState)

        opt_state = deepcopy_opt_state(opt_state)
        opt_state.hyperparams["learning_rate"] = hyperparams.lr
        return workflow_state.replace(opt_state=opt_state)
