import numpy as np
import random
import copy
from MDP.mdp import MDP


class Simulator:

    def __init__(self, num_games=0, alpha_value=0, gamma_value=0, epsilon_value=0):
        '''
        Setup the Simulator with the provided values.
        :param num_games - number of games to be trained on.
        :param alpha_value - 1/alpha_value is the decay constant.
        :param gamma_value - Discount Factor.
        :param epsilon_value - Probability value for the epsilon-greedy approach.
        '''
        self.num_games = num_games
        self.epsilon_value = epsilon_value
        self.alpha_value = alpha_value
        self.gamma_val = gamma_value
        self.q_table = np.zeros((10369, 3))
        self.state_ref = {}
        self.used_states = 0

        self.train_agent()

        # Agent Evaluation
        # Should not explore anymore, fully utilizes Q-Table
        self.epsilon_value = 0
        self.run_aggregate_games()

    def run_aggregate_games(self):
        '''
        Play series of games with final Q-Table.
        '''
        total_b = 0
        limit = 5000
        for x in range(0, limit):
            total_b += self.play_game()

        print(total_b / float(limit))

    def choose_action(self, s):
        '''
        Choose action based on an epsilon greedy approach
        :param s tuple for current discretized state
        :return action selected
        '''
        ac_s = None

        x = random.random()
        if x <= self.epsilon_value:
            ac_s = self.random_action()
        else:
            # Simulate all possible next steps, pick optimal
            # If all are equal, randomize action
            _, ac_s = self.get_best_next_s(s)

        return ac_s

    def random_action(self):
        '''
        Randomizes action selected
        '''
        return random.randint(0, 2)

    def train_agent(self):
        '''
        Train the agent over a certain number of games.
        '''
        for i in range(0, self.num_games):
            self.simulate_game()

    def play_game(self):
        '''
        Simulate an actual game till the agent loses.
        '''
        b = self.simulate_game()
        return b

    def simulate_game(self):
        '''
        Simulate a full pong game given an state
        :param state MDP State representing the initial state
        :returns score total number of bounces from this run
        '''
        state = MDP()
        r = 0
        b = 0
        while r != -1:
            ac_s = self.choose_action(state)
            r = state.simulate_one_time_step(ac_s)
            if r == 1:
                b += r
            self.update_q_table(r, ac_s, state)

        return b

    def update_q_table(self, r, a, s):
        '''
        Updates Q-Table based off Q-Update Iterative Equation
        '''
        q = self.q_v(s, a)
        b_s_i, b_a = self.get_best_next_s(s)
        new_q = q + self.alpha_value * (r + self.gamma_val * self.q_table[b_s_i][b_a] - q)
        self.q_table[self.index(s)][a] = new_q

    def index(self, state):
        '''
        Given a discretized state, get the reference index for q-table
        if new state, establish new index in reference dictionary
        :returns index index of state in q-table
        '''
        d_state = state.discretize_state()
        if d_state not in self.state_ref:
            self.state_ref[d_state] = self.used_states
            self.used_states += 1

        return self.state_ref[d_state]

    def q_v(self, state, action):
        '''
        Given a state and action, get the q-value of that from the q-table
        '''
        if state.ball_x > 1:
            if not state.hit_paddle():
                return -1

        return self.q_table[self.index(state)][action]

    def get_best_next_s(self, state):
        '''
        Uses Q-Table to find optimal next step
        '''
        ns0 = copy.deepcopy(state)
        ns0.simulate_one_time_step(0)
        ns1 = copy.deepcopy(state)
        ns1.simulate_one_time_step(1)
        ns2 = copy.deepcopy(state)
        ns2.simulate_one_time_step(2)

        q0 = self.q_v(ns0, 0)
        q1 = self.q_v(ns1, 1)
        q2 = self.q_v(ns2, 2)

        if q0 > q1 and q0 > q2:
            return self.index(ns0), 0
        elif q1 > q0 and q1 > q2:
            return self.index(ns1), 1
        elif q2 > q0 and q2 > q1:
            return self.index(ns2), 2
        else:
            ac_s = self.random_action()
            if ac_s == 0:
                return self.index(ns0), 0
            elif ac_s == 1:
                return self.index(ns1), 1
            elif ac_s == 2:
                return self.index(ns2), 2
