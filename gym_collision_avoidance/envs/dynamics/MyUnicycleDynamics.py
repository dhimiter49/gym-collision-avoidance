import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics
from gym_collision_avoidance.envs.util import wrap, find_nearest
import math

class MyUnicycleDynamics(Dynamics):
    """ Convert a speed & heading to a new state according to Unicycle Kinematics model.

    """

    def __init__(self, agent):
        Dynamics.__init__(self, agent)

    def step(self, action, dt):
        """

        In the global frame, assume the agent instantaneously turns by :code:`heading`
        and moves forward at :code:`speed` for :code:`dt` seconds.
        Add that offset to the current position. Update the velocity in the
        same way. Also update the agent's turning direction (only used by CADRL).

        Args:
            action (list): [delta heading angle, speed] command for this agent
            dt (float): time in seconds to execute :code:`action`

        """
        selected_speed = action[0]
        selected_heading = wrap(action[1] + self.agent.heading_global_frame)

        # We assume a constant acceleration to this selected speed which means that in
        # order to compute displacement we can average between the old and the new speed
        displacement_speed = np.mean(self.agent.past_actions[:2, 0])
        dx = displacement_speed * np.cos(selected_heading) * dt
        dy = displacement_speed * np.sin(selected_heading) * dt
        self.agent.pos_global_frame += np.array([dx, dy])

        self.agent.vel_global_frame[0] = selected_speed * np.cos(selected_heading)
        self.agent.vel_global_frame[1] = selected_speed * np.sin(selected_heading)
        self.agent.speed_global_frame = selected_speed
        self.agent.delta_heading_global_frame = wrap(selected_heading -
                                               self.agent.heading_global_frame)
        self.agent.heading_global_frame = selected_heading

        # turning dir: needed for cadrl value fn
        if abs(self.agent.turning_dir) < 1e-5:
            self.agent.turning_dir = 0.11 * np.sign(selected_heading)
        elif self.agent.turning_dir * selected_heading < 0:
            self.agent.turning_dir = max(-np.pi, min(np.pi, -self.agent.turning_dir + selected_heading))
        else:
            self.agent.turning_dir = np.sign(self.agent.turning_dir) * max(0.0, abs(self.agent.turning_dir)-0.1)