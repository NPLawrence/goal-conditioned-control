from copy import deepcopy
from typing import Any, Generic, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame

import matplotlib.pyplot as plt
import os 


__all__ = [
    "RenderCollection",
    "RecordVideo",
    "HumanRendering",
    "AddWhiteNoise",
    "ObstructView",
]


class TerminalFrameCapture(
    gym.Wrapper[ObsType, ActType, ObsType, ActType],
    Generic[ObsType, ActType, RenderFrame],
    gym.utils.RecordConstructorArgs,
):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        path: str
    ):
        """Initialize a :class:`RenderCollection` instance.

        Args:
            env: The environment that is being wrapped
            pop_frames (bool): If true, clear the collection frames after ``meth:render`` is called. Default value is ``True``.
            reset_clean (bool): If true, clear the collection frames when ``meth:reset`` is called. Default value is ``True``.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self
        )
        gym.Wrapper.__init__(self, env)

        assert env.render_mode is not None
        assert not env.render_mode.endswith("_list")
        self.path = path
        self.frame_list: list[RenderFrame] = []

        os.makedirs(self.path, exist_ok=True)

        
        self.capture_idx = 0

        self.metadata = deepcopy(self.env.metadata)
        if f"{self.env.render_mode}_list" not in self.metadata["render_modes"]:
            self.metadata["render_modes"].append(f"{self.env.render_mode}_list")

    @property
    def render_mode(self):
        """Returns the collection render_mode name."""
        return f"{self.env.render_mode}_list"

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Perform a step in the base environment and collect a frame."""
        output = super().step(action)
        if output[2] or output[3]:
            self.frame_list.append(super().render())
            plt.imsave(os.path.join(self.path, f"capture_{self.capture_idx}.png"), self.frame_list[-1])
            self.capture_idx += 1
        return output

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the base environment, eventually clear the frame_list, and collect a frame."""
        output = super().reset(seed=seed, options=options)

        # if self.reset_clean:
        self.frame_list = []
        # self.frame_list.append(super().render())

        return output

    def render(self) -> list[RenderFrame]:
        """Returns the collection of frames and, if pop_frames = True, clears it."""
        frames = self.frame_list
        self.frame_list = []

        return frames