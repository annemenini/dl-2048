import os

import tensorflow as tf

import configuration
from Grid import Grid
import tf_grid


FLAGS = tf.flags.FLAGS


def generate_grid(dummy=None):
    grid = tf.convert_to_tensor(Grid().grid)
    feature = {"grid": grid, "step": 0}
    return feature


def get_dataset(mode='train'):
    if mode == 'train':
        repeat = FLAGS.num_train_cases_per_epoch
    else:
        repeat = FLAGS.num_eval_cases_per_epoch
    dummy_dataset = tf.data.Dataset.from_tensors(0).repeat(repeat)
    dataset = dummy_dataset.map(generate_grid)
    dataset.batch(FLAGS.batchsize)
    if FLAGS.mode == 'train':
        dataset = dataset.repeat(FLAGS.num_epochs)
    else:
        dataset = dataset.repeat(1)
    return dataset


def input_fn(dataset):
    iterator = dataset.make_one_shot_iterator()
    feature = iterator.get_next()
    return feature


def model_fn(features, labels=None, mode=tf.estimator.ModeKeys.TRAIN, params=None, config=None):
    # TODO: Support batch size > 1
    grid = features['grid']
    grid = tf.expand_dims(grid, axis=0)
    step = tf_grid.game_network(grid)

    # Compute predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'step': step}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss
    loss = tf.cast(-step, dtype=tf.float32)

    # Compute evaluation metrics
    tf.summary.scalar('step', step)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    # Create training op
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    export_outputs = {'step': step}
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, export_outputs=export_outputs)


def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=None,
                                           name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    if FLAGS.use_tfrecords:
        features = generate_grid()
    else:
        features = generate_grid()
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def train_and_evaluate(enhancer):
    train_dataset = get_dataset(mode='train')
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_dataset))
    validation_dataset = get_dataset(mode='validation')
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(validation_dataset), steps=10)

    eval_result = tf.estimator.train_and_evaluate(enhancer, train_spec, eval_spec)

    saved_model_path = os.path.join(FLAGS.output_path, 'saved_model')
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)
    enhancer.export_saved_model(saved_model_path, tf.estimator.export.build_raw_serving_input_receiver_fn(
        features={'grid': tf.placeholder(dtype=tf.int64, shape=[4, 4], name='grid'),
                  'step': tf.placeholder(dtype=tf.int64, shape=[], name='step')}))

    return eval_result


def train(enhancer):
    # Train
    train_dataset = get_dataset(mode='train')
    enhancer.train(input_fn=lambda: input_fn(train_dataset))

    # Evaluate the model.
    validation_dataset = get_dataset(mode='validation')
    eval_result = enhancer.evaluate(input_fn=lambda: input_fn(validation_dataset))

    saved_model_path = os.path.join(FLAGS.output_path, 'saved_model')
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)
    enhancer.export_saved_model(saved_model_path, serving_input_receiver_fn)

    return eval_result


def test(enhancer):
    # Test
    test_dataset = get_dataset(mode='test')
    test_result = enhancer.evaluate(input_fn=lambda: input_fn(test_dataset))

    return test_result


def predict(enhancer):
    # Test
    prediction_dataset = get_dataset(mode='prediction')
    output = enhancer.predict(input_fn=lambda: input_fn(prediction_dataset))
    # TODO save output


def main(argv=None):
    configuration.customize_configuration()
    my_feature_columns = [tf.feature_column.numeric_column(key='input')]

    # Build model
    enhancer = tf.estimator.Estimator(
        model_fn=model_fn,
        params={'feature_columns': my_feature_columns},
        model_dir=FLAGS.output_path
    )

    if FLAGS.mode == 'train':
        train_and_evaluate(enhancer)
    elif FLAGS.mode == 'test':
        test(enhancer)
    elif FLAGS.mode == 'predict':
        predict(enhancer)
    else:
        raise ValueError('Unrecognized mode: ' + FLAGS.mode)


if __name__ == '__main__':

    tf.app.run()
