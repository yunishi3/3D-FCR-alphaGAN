import os
import numpy as np

from train import train
from config import cfg
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", cfg.TRAIN.NUM_EPOCH, "Epoch to train [15]") #n_epochs = cfg.TRAIN.NUM_EPOCH
flags.DEFINE_float("learning_rate_G", cfg.LEARNING_RATE_G, "Learning rate for Generator of adam [0.0001]") #learning_rate_G = cfg.LEARNING_RATE_G
flags.DEFINE_float("learning_rate_D", cfg.LEARNING_RATE_D, "Learning rate for Discriminator of adam [0.0001]") #learning_rate_D = cfg.LEARNING_RATE_D
flags.DEFINE_integer("batch_size", cfg.CONST.BATCH_SIZE, "The size of batch voxels [100]") #batch_size = cfg.CONST.BATCH_SIZE

flags.DEFINE_boolean("middle_start", False, "True for starting from the middle [False]")
flags.DEFINE_integer("ini_epoch", 0, "The number of initial epoch --if middle_start: False -> 0, True -> must assign the number [0]")
flags.DEFINE_string("mode", 'train', "Execute mode: train/evaluate_recons/evaluate_interpolate/evaluate_noise")
flags.DEFINE_integer("conf_epoch", 10000, "The number of confirmation epoch to evaluate interpolate, reconstruction etc [100]")

FLAGS = flags.FLAGS

def main():
    if not os.path.exists(cfg.DIR.CHECK_POINT_PATH):
        os.makedirs(cfg.DIR.CHECK_POINT_PATH)
    if not os.path.exists(cfg.DIR.TRAIN_OBJ_PATH):
        os.makedirs(cfg.DIR.TRAIN_OBJ_PATH)
    if not os.path.exists(cfg.DIR.EVAL_PATH):
        os.makedirs(cfg.DIR.EVAL_PATH)
    if FLAGS.middle_start:
        print 'middle_start'

    if FLAGS.mode == 'train':
        train(FLAGS.epoch, FLAGS.learning_rate_G, FLAGS.learning_rate_D, FLAGS.batch_size, FLAGS.middle_start, FLAGS.ini_epoch)
    elif FLAGS.mode == 'evaluate_recons' or 'evaluate_interpolate' or 'evaluate_noise':
        from evaluate import evaluate
        if FLAGS.mode == 'evaluate_recons':
            mode = 'recons'
        elif FLAGS.mode == 'evaluate_interpolate':
            mode = 'interpolate'
        else:
            mode = 'noise'
        evaluate(FLAGS.batch_size, FLAGS.conf_epoch, mode)


if __name__ == '__main__':
    #tf.app.run()
    main()
