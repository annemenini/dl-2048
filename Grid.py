import random

import matplotlib.pyplot as plt
import numpy as np


class Grid:

    def __init__(self, show_flag=False, dim_x=4, dim_y=4):
        self.show_flag = show_flag
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.flat_grid_size = dim_x * dim_y
        self.grid = np.zeros((dim_x, dim_y), dtype=int)
        self.init_grid()

    def init_grid(self):
        flat_index1 = random.randrange(0, self.flat_grid_size)
        flat_index2 = random.randrange(0, self.flat_grid_size)
        while flat_index2 == flat_index1:
            flat_index2 = random.randrange(0, self.flat_grid_size)
        index1 = np.unravel_index(flat_index1, (self.dim_x, self.dim_y))
        index2 = np.unravel_index(flat_index2, (self.dim_x, self.dim_y))
        self.grid[index1] = 4 if random.uniform(0, 1) < 0 else 2
        self.grid[index2] = 4 if random.uniform(0, 1) < 0.25 else 2

    def project(self, direction):
        # Rotate to be come to the "up case"
        current_grid = np.rot90(self.grid, k=direction)

        # # Check that the projection is valid
        # condition = (current_grid[0:(current_grid.shape[0]-1), :] - current_grid[1:current_grid.shape[0], :]) == 0
        # similar_indexes = np.extract(condition, current_grid)
        # if similar_indexes.size > 0 and np.max(similar_indexes) == 0:
        #     return False

        # Apply projection
        new_grid = np.zeros_like(current_grid)
        for y in range(0, current_grid.shape[1]):
            previous = None
            previous_index = 0
            for x in range(0, current_grid.shape[0]):
                if current_grid[x, y] > 0:
                    if previous is None:
                        previous = current_grid[x, y]
                    else:
                        if current_grid[x, y] == previous:
                            new_grid[previous_index, y] = 2 * previous
                            previous = None
                        else:
                            new_grid[previous_index, y] = previous
                            previous = current_grid[x, y]
                        previous_index += 1
            if previous is not None:
                new_grid[previous_index, y] = previous

        # Check that the projection is valid
        if np.array_equal(current_grid, new_grid):
            return False

        # Rotate back to the initial orientation
        self.grid = np.rot90(new_grid, k=4 - direction)
        return True

    def add(self):
        # Define the value to add
        value = 4 if random.uniform(0, 1) < 0.125 else 2

        # Fine a location to add it
        condition = self.grid == 0
        index_array = np.arange(0, self.flat_grid_size).reshape((self.dim_x, self.dim_y))
        zeros_indexes = np.extract(condition, index_array)
        chosen_flat_index = zeros_indexes[random.randrange(0, zeros_indexes.size)]
        index = np.unravel_index(chosen_flat_index, (self.dim_x, self.dim_y))

        # Add the value
        self.grid[index] = value
        return True

    def show(self):
        plt.imshow(np.log(1 + self.grid), vmin=0, vmax=11)
        plt.pause(0.01)

    def random_step(self):
        is_projected = False
        direction = np.random.permutation(np.arange(4))
        direction_index = 0
        while not is_projected:
            is_projected = self.project(direction[direction_index])
            direction_index += 1
        is_success = self.add()
        if self.show_flag:
            self.show()
        return is_success

    def fcnn(self, weights):
        ns = [4 * 4, 64, 32, 16, 4]

        layer = np.reshape(np.log2(self.grid + 1), ns[0])
        layer = layer.astype(float)
        i2 = 0

        for i in range(1, len(ns)):
            i0 = i2
            i1 = i0 + ns[i] * ns[i - 1]
            i2 = i1 + ns[i]
            w = np.reshape(weights[i0:i1], (ns[i], ns[i - 1]))
            b = np.reshape(weights[i1:i2], ns[i])
            layer = np.matmul(w, layer) + b
            layer = np.maximum(layer, 0)

        direction = np.argmax(layer)
        return direction

    def step(self, weights):
        direction = self.fcnn(weights)
        is_projected = self.project(direction)
        if not is_projected:
            return False
        is_success = self.add()
        if self.show_flag:
            self.show()
        return is_success

    def play_until_end(self, weights):
        grid0 = self.grid
        do_continue = True
        step = 0
        while do_continue:
            do_continue = self.step(weights)
            step += 1
        self.grid = grid0
        return step


if __name__ == "__main__":
    # Add -m cProfile -s cumtime to profile
    grid = Grid(show_flag=True)
    do_continue = True
    step = 0
    while do_continue:
        do_continue = grid.random_step()
        step += 1
    print(step)
