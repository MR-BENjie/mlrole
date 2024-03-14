from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, store=None):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode, store=store)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, store=None):
        if test_mode:
            self.agent.eval()

        agent_inputs = self._build_inputs(ep_batch, t)

        if store is not None:
            agent_inputs.requires_grad_()
            store['store_obs'].append(ep_batch["obs"][:, t])
            store['store_input'].append(agent_inputs.data)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)


        if store is not None:
            agent_pre = th.tensor(agent_outs, dtype=th.int)
            agent_pre = th.tensor(agent_pre, dtype=th.float)

            loss_func = th.nn.MSELoss()
            l = loss_func(agent_outs, agent_pre)

            l.backward(retain_graph=True)
            store['store_input_grad'].append(agent_inputs.grad.data)

        return agent_outs