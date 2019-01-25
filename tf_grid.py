import tensorflow as tf


def rotation(grid, direction):
    def f0(): return grid

    def f1(): return tf.transpose(grid, perm=[0, 2, 1])

    def f2(): return tf.image.flip_up_down(grid)

    def f3(): return tf.image.flip_up_down(tf.transpose(grid, perm=[0, 2, 1]))

    def fn1(): return tf.cond(tf.equal(direction, 2), f2, f3)

    def fn0(): return tf.cond(tf.equal(direction, 1), f1, fn1)

    grid = tf.cond(tf.equal(direction, 0), f0, fn0)

    return grid


def reverse_rotation(grid, direction):
    def f0(): return grid

    def f1(): return tf.transpose(grid, perm=[0, 2, 1])

    def f2(): return tf.image.flip_up_down(grid)

    def f3(): return tf.transpose(tf.image.flip_up_down(grid), perm=[0, 2, 1])

    def fn1(): return tf.cond(tf.equal(direction, 2), f2, f3)

    def fn0(): return tf.cond(tf.equal(direction, 1), f1, fn1)

    grid = tf.cond(tf.equal(direction, 0), f0, fn0)

    return grid


def is_previous_zero(previous_index, elem):
    return elem, previous_index


def is_previous_equal_elem(previous, previous_index, grid):
    grid = tf.get_variable("grid_column", [4], tf.int64, trainable=False)
    tf.scatter_update(grid, previous_index, 2 * previous)
    return tf.cast(0, dtype=tf.int64), previous_index + 1


def is_previous_not_elem(previous, previous_index, elem, grid):
    grid = tf.get_variable("grid_column", [4], tf.int64, trainable=False)
    tf.scatter_update(tf.Variable(grid), previous_index, previous)
    return elem, previous_index + 1


def is_previous_non_zero(previous, previous_index, elem, grid):
    return tf.cond(tf.equal(previous, elem),
                   lambda: is_previous_equal_elem(previous, previous_index, grid),
                   lambda: is_previous_not_elem(previous, previous_index, elem, grid))


def is_elem_equal_zero(previous, previous_index):
    return previous, previous_index


def is_elem_non_zero(previous, previous_index, elem, grid):
    return tf.cond(tf.equal(previous, 0),
                   lambda: is_previous_zero(previous_index, elem),
                   lambda: is_previous_non_zero(previous, previous_index, elem, grid))


def final_previous_equal_zero():
    return True


def final_previous_non_zero(previous, previous_index, grid):
    grid = tf.get_variable("grid_column", [4], tf.int64, trainable=False)
    tf.scatter_update(grid, previous_index, previous)
    return True


def project(grid, direction):
    # Rotate to be come to the "up case"
    current_grid = rotation(grid, direction)

    # Apply projection
    new_grid = tf.zeros([1, 4, 0], dtype=tf.int64)
    for y in range(0, 4):
        previous = tf.cast(0, dtype=tf.int64)
        previous_index = 0
        new_column = tf.get_variable("grid_column", [4], tf.int64, trainable=False)
        tf.assign(new_column, tf.zeros([4], dtype=tf.int64))
        for x in range(0, 4):
            elem = current_grid[0, x, y]
            previous, previous_index = tf.cond(tf.equal(elem, 0),
                                               lambda: is_elem_equal_zero(previous, previous_index),
                                               lambda: is_elem_non_zero(previous, previous_index, elem, new_column))
        tf.cond(tf.equal(previous, 0),
                lambda: final_previous_equal_zero(),
                lambda: final_previous_non_zero(previous, previous_index, new_column))
        new_grid = tf.concat([new_grid, tf.reshape(new_column, [1, 4, 1])], axis=2)

    # Check if the project actually did something
    is_successful = tf.cond(tf.reduce_all(tf.equal(current_grid, new_grid)), lambda: False, lambda: True)

    # Rotate back
    new_grid = reverse_rotation(new_grid, direction)

    return is_successful, new_grid


def fcnn(grid):
    """
    Fully Connected Neural Network
    Assumes an input image of shape [N, 4, 4]
    """
    shape = tf.shape(grid)
    units0 = 4 * 4
    grid = tf.reshape(grid, [shape[0], units0])
    grid = tf.cast(grid, tf.float32)
    layer = tf.layers.dense(inputs=grid, units=32, activation=tf.nn.relu, name="dense0")
    layer = tf.layers.dense(inputs=layer, units=64, activation=tf.nn.relu, name="dense1")
    layer = tf.layers.dense(inputs=layer, units=128, activation=tf.nn.relu, name="dense2")
    layer = tf.layers.dense(inputs=layer, units=64, activation=tf.nn.relu, name="dense3")
    layer = tf.layers.dense(inputs=layer, units=32, activation=tf.nn.relu, name="dense4")
    logits = tf.layers.dense(inputs=layer, units=4, activation=None, name="dense5")

    direction = tf.argmax(logits, axis=1)

    return direction


def add_elem(grid):
    # Define the value to add
    value = tf.cond(tf.less(tf.random_uniform([]), 0.125), lambda: tf.cast(4, tf.int64), lambda: tf.cast(2, tf.int64))

    # Fine a location to add the value
    grid = tf.get_variable("grid_column", [4], tf.int64, trainable=False)
    condition = grid == 0
    zero_indexes = tf.where(condition)
    chosen_index = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int64)
    index = zero_indexes[chosen_index, :]

    # Add the value
    tf.scatter_update(grid, index, value)

    return grid


def step_network(input_grid):
    direction = fcnn(input_grid)
    direction = tf.squeeze(direction, axis=0)
    is_successful, projected_grid = project(input_grid, direction)
    output_grid = tf.cond(tf.equal(is_successful, True), lambda: add_elem(projected_grid), lambda: input_grid)
    return is_successful, output_grid


def game_while_body(is_successful, grid, step):
    tf.add(step, 1)
    new_is_successful, new_grid = step_network(grid)
    return new_is_successful, new_grid, step


def game_while_condition(is_successful, grid, step):
    return tf.equal(is_successful, True)


def game_network(grid):
    step = 0
    with tf.variable_scope("game", reuse=tf.AUTO_REUSE):
        is_successful, grid = step_network(grid)
        is_successful, grid, step = tf.while_loop(game_while_condition,
                                                  game_while_body,
                                                  [is_successful, grid, step])
    return step
