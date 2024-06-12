import torch
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class BattleEpisodeRunner:

    def __init__(self, args, logger):

        self.store_info = False

        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        if self.store_info:
            self.store_data = dict()
            self.store_data['store_input_grad'] = list()
            self.store_data['store_obs'] = list()
            self.store_data['store_action'] = list()
            self.store_data['store_input'] = list()
            self.store_data['pre_state'] = list()
        else:
            self.store_data = None
        # Log the first run
        self.log_train_stats_t = -1000000

        self.learner = None
        self.judge_model_used = False
    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.macs = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, games=50):

        accuracy_list = list()
        battle_won = list()
        for ep in range(games):

            self.reset()

            terminated = False
            episode_return = 0

            for mac in self.macs:
                mac.init_hidden(batch_size=self.batch_size)

            won = 0
            while not terminated:
                pre_transition_data = {
                    #"state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()],
                    "enemy_obs":[self.env.get_enemy_obs()]
                }

                accurancy = self.env.get_type_judege_accurancy()
                accuracy_list.append(accurancy)


                self.batch.update(pre_transition_data, ts=self.t)
                actions = self.macs[0].select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True,
                                                      store=self.store_data)

                pre_transition_data["obs"] = pre_transition_data["enemy_obs"]
                pre_transition_data['avail_actions'] = [self.env.get_enemy_avail_actions()]
                self.batch.update(pre_transition_data, ts=self.t)

                actions_2 = self.macs[1].select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True, store=self.store_data )

                # Fix memory leak
                cpu_actions = actions.to("cpu").numpy()

                #reward, terminated, env_info = self.env.step_Twoagent([actions[0]])
                reward, terminated, env_info = self.env.step_Twoagent([actions[0],actions_2[0]])


                episode_return += reward

                post_transition_data = {
                    "actions": cpu_actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }

                self.batch.update(post_transition_data, ts=self.t)

                self.t += 1

                if terminated and env_info["battle_won"]:
                    won = 1

            print("%d run terminate;"%ep)
            battle_won.append(won)

        last_data = {
            #"state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            "enemy_obs": [self.env.get_enemy_obs()],
        }
        self.batch.update(last_data, ts=self.t)
        print("battle_won_mean (first model) : %.3f"%np.mean(battle_won))
        if self.judge_model_used:
            self.logger.log_stat("judge_accuracy_mean", np.mean(accuracy_list), self.t_env)
            self.logger.log_stat("judge_accuracy_ste", np.std(accuracy_list), self.t_env)
            print("judge_accuracy_mean : %.3f "%np.mean(accuracy_list))

        return self.batch #, self.store_data

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
