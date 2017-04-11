import random
import numpy as np


class MDP:

    def __init__(self,
                 ball_x=None,
                 ball_y=None,
                 velocity_x=None,
                 velocity_y=None,
                 paddle_y=None):
        '''
        Setup MDP with the initial values provided.
        '''
        self.create_state(
            ball_x=ball_x,
            ball_y=ball_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            paddle_y=paddle_y
        )
        
        # the agent can choose between 3 actions - stay, up or down respectively.
        self.actions = [0, 0.04, -0.04]
        
    
    def create_state(self,
              ball_x=None,
              ball_y=None,
              velocity_x=None,
              velocity_y=None,
              paddle_y=None):
        '''
        Helper function for the initializer. Initialize member variables with provided or default values.
        '''
        self.paddle_height = 0.2
        self.ball_x = ball_x if ball_x != None else 0.5
        self.ball_y = ball_y if ball_y != None else 0.5
        self.velocity_x = velocity_x if velocity_x != None else 0.03
        self.velocity_y = velocity_y if velocity_y != None else 0.01
        self.paddle_y = 0.5

    def is_same_state(self, alt_state):
        '''
        Checks if the compared discretized state is the same as this
        If ball is in the same cell, then it is the same state
        '''
        bx = self.discrete_bx(self.ball_x)
        by = self.discrete_by(self.ball_y)

        if bx == alt_state.ball_x and by == alt_state.ball_y:
            return True
        else:
            return False

    def simulate_one_time_step(self, action_selected):
        '''
        :param action_selected - Current action to execute.
        Perform the action on the current continuous state.
        '''
        # Update Paddle with action
        self.update_position(self.actions[action_selected])
        self.ball_x += self.velocity_x
        self.ball_y += self.velocity_y

        return self.check_ball_pos()

    def update_position(self, dy):
        '''
        Updates paddle position with dy
        Param: dy float represents amount to move the paddle
        '''
        self.paddle_y += dy
        if self.paddle_y < 0:
            self.paddle_y = 0
        elif self.paddle_y > 0.8:
            self.paddle_y = 0.8

    def check_ball_pos(self):
        '''
        Checks if ball should bounce against anything

        :returns current state reward
            1: Hit Paddle
            0: Hit nothing
            -1: Paddle Missed, GAME-OVER
        '''
        if self.ball_y < 0:
            self.ball_y = -self.ball_y
            self.flip_vy()
        elif self.ball_y > 1:
            self.ball_y = 2 - self.ball_y
            self.flip_vy()

        if self.ball_x < 0:
            self.ball_x = -self.ball_x
            self.flip_vx()
        elif self.ball_x >= 1:
            return self.check_agent_loss()

        return 0

    def check_agent_loss(self):
        '''
        Checks if ball goes beyond the paddle
        Game-Ending Condition
        '''
        if not self.hit_paddle():
            return -1

        self.update_velocity()
        return 1

    def update_velocity(self):
        '''
        Update velocity by a random value
        Called every time the ball hits the paddle
        '''
        self.velocity_x = -self.velocity_x + random.uniform(-0.015, 0.015)
        self.velocity_y = self.velocity_y + random.uniform(-0.03, 0.03)

        if abs(self.velocity_x) < 0.03:
            self.velocity_x = 0.03 * (self.velocity_x / self.velocity_x)

        if abs(self.velocity_x) > 1:
            self.velocity_x = 1 * (self.velocity_x / self.velocity_x)

        if abs(self.velocity_y) > 1:
            self.velocity_y = 1 * (self.velocity_y / self.velocity_y)

    def flip_vx(self):
        '''
        Flips the balls x velocity
        Called when ball hits paddle
        '''
        self.velocity_x = -self.velocity_x

    def flip_vy(self):
        '''
        Flips the balls y velocity
        Called when ball hits ceiling or floor
        '''
        self.velocity_y = -self.velocity_y

    def hit_paddle(self):
        '''
        Checks if ball's position will hit paddle
        '''
        return self.paddle_y > self.ball_y and self.ball_y > (self.paddle_y - 0.2)

    def discretize_state(self):
        '''
        Convert the current continuous state to a discrete state.
        '''
        bx = self.d_pos(self.ball_x)
        by = self.d_pos(self.ball_y)
        vx = self.d_vx(self.velocity_x)
        vy = self.d_vy(self.velocity_y)
        py = self.d_py(self.paddle_y)
        return (bx, by, vx, vy, py)

    def d_pos(self, pos):
        '''
        Board is divided into 12x12 grid
        Returns int of appropriate cell
        '''
        if pos == 0:
            return 1
        if pos > 1:
            return 12
        cell_width = 1.0 / 12.0
        return int(np.ceil(pos / cell_width))

    def d_vx(self, vx):
        '''
        Discretize x velocity per documentation
        '''
        if vx >= 0:
            return 1
        else:
            return -1

    def d_vy(self, vy):
        '''
        Discretize y velocity per documentation
        '''
        if vy > 0.015:
            return 1
        elif vy < -0.015:
            return -1
        else:
            return 0

    def d_py(self, py):
        '''
        Discretize paddle y per documentation
        '''
        if py >= 0.8:
            return 11
        else:
            return np.floor(12 * py / 0.8)
