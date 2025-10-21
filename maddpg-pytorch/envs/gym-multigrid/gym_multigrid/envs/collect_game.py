from ..multigrid import *
MAX_STEPS = 100

class CollectGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_balls=[],
        agents_index = [],
        balls_index=[],
        balls_reward=[],
        zero_sum = False,
        view_size=7

    ):
        self.num_balls = num_balls
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum

        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            # max_steps= 10000,
            max_steps=MAX_STEPS,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )

        # Track the positions of balls grouped by their color index so we
        # don't need to rescan the grid every step.
        self._ball_positions_by_color = {}



    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Reset tracked ball positions whenever a new grid is generated.
        self._ball_positions_by_color = {}

        for number, index, reward in zip(self.num_balls, self.balls_index, self.balls_reward):
            for i in range(number):
                pos = self.place_obj(Ball(self.world, index, reward))
                self._ball_positions_by_color.setdefault(index, set()).add(tuple(pos))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)


    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        for j,a in enumerate(self.agents):
            if a.index==i or a.index==0:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, self.agents[i].index]:
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self._remove_ball_position(fwd_cell.index, tuple(fwd_pos))
                    self._reward(i, rewards, fwd_cell.reward)

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        """
        Mild penalty to discourage worthless move forward actions
        """
        for j,a in enumerate(self.agents):
            # agent with index 0 gets shared penalty for all agents
            if a.index==i or a.index==0:
                rewards[j]-=0.01

    def _remove_ball_position(self, color_index, position):
        """Remove a ball from the cached positions when it is picked up."""
        positions = self._ball_positions_by_color.get(color_index)
        if not positions:
            return

        if position in positions:
            positions.remove(position)
            if not positions:
                del self._ball_positions_by_color[color_index]

    def _compute_shared_distance_rewards(self):
        """Compute shared rewards based on Manhattan distance to matching balls."""

        # Group agents by their color index
        color_to_agent_indices = {}
        for agent_idx, agent in enumerate(self.agents):
            color_to_agent_indices.setdefault(agent.index, []).append(agent_idx)

        shared_rewards = np.zeros(len(self.agents), dtype=float)

        for color, agent_indices in color_to_agent_indices.items():
            ball_positions = self._ball_positions_by_color.get(color)
            if not ball_positions:
                continue

            total_distance = 0.0

            for ball_pos in ball_positions:
                nearest_distance = None

                for agent_idx in agent_indices:
                    agent_pos = self.agents[agent_idx].pos
                    if agent_pos is None:
                        continue

                    distance = abs(agent_pos[0] - ball_pos[0]) + abs(agent_pos[1] - ball_pos[1])
                    if nearest_distance is None or distance < nearest_distance:
                        nearest_distance = distance

                if nearest_distance is not None:
                    total_distance += nearest_distance

            reward_value = -float(total_distance)

            for agent_idx in agent_indices:
                shared_rewards[agent_idx] = reward_value

        return shared_rewards

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        distance_rewards = self._compute_shared_distance_rewards()
        rewards += distance_rewards   # element-wise (index-wise) sum
        return obs, rewards, done, info


# class CollectGame4HEnv10x10N2(CollectGameEnv):
#     def __init__(self):
#         super().__init__(size=10,
#         num_balls=[5],
#         agents_index = [0,1,2],
#         balls_index=[0],
#         balls_reward=[1],
#         zero_sum=True)

class CollectGame4HEnv10x10N2(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[2],
        agents_index = [0, 0, 0],   # 3 agents with index 0
        balls_index=[0, 0],     # 2 balls with index 0
        balls_reward=[5, 5],    # rewards to pick balls
        zero_sum=False)
