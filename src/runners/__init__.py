REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .battle_episodes_runner import BattleEpisodeRunner
REGISTRY["battle"] = BattleEpisodeRunner