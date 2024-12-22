import torch

from dataclasses import dataclass


@dataclass
class GameState:
    """
    body: grid_size * grid_size x 2
    fruit: 2
    length: 0
    direction: 2
    """
    grid_size: int
    body: torch.LongTensor
    fruit: torch.LongTensor
    length: torch.LongTensor
    direction: torch.LongTensor

    @property
    def tensor(self):
        grid = torch.zeros(3, self.grid_size, self.grid_size)
        grid[0, self.body[0, 0], self.body[0, 1]] = 1.0
        grid[1, self.body[1:self.length], self.body[1:self.length]] = 1.0
        grid[2, self.fruit[0], self.fruit[1]] = 1.0

        return grid

    def step(self, action):
        if action == 1:  # Turn left
            self.direction = self.direction[[1, 0]]
            self.direction[1] *= -1
        elif action == 2:  # Turn right
            self.direction = self.direction[[1, 0]]
            self.direction[0] *= -1

        next_head = self.body[0] + self.direction
        if (next_head in self.body[1:self.length] or
                next_head[0] < 0 or
                next_head[1] < 0 or
                next_head[0] > self.grid_size - 1 or
                next_head[1] > self.grid_size - 1):
            return -1.0

        tail = self.body[self.length - 1].clone()

        self.body[1:self.length] = self.body[:self.length - 1]
        self.body[0] += self.direction

        if self.body[0] == self.fruit:
            self.body[self.length] = tail
            self.length += 1
            self.fruit = torch.randint(0, self.grid_size, (2,))  # TODO. What if the snake is there?

            return 1.0

        return 0.0
