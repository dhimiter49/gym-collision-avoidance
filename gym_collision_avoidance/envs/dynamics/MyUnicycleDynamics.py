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
        Implemented primerely for MPC, set velocity in x and y axis. During the step,
        assume a contant acceleration from previous velocity to next.

        Args:
            action (list): [vel x, vel y] command for this agent
            dt (float): time in seconds to execute :code:`action`

        """

        # We assume a constant acceleration to this selected speed which means that in
        # order to compute displacement we can average between the old and the new speed
        last_avg_vel = np.mean(self.agent.past_actions[:2], axis=0)
        self.agent.pos_global_frame += last_avg_vel * dt
        self.agent.vel_global_frame = action