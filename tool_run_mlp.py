"""This runs an MLP on feature vectors"""
from __future__ import print_function
import argparse
from os.path import join as join_path

import tensorflow as tf

from data_utils.loader import load
from prediction.predictor import MlpPredictor
from prediction.trainer import Trainer
from prediction.simple_batch_generator import SimpleBatchGenerator

from eval.eval import evaluate_ranking

DEGREE = 8  # a constant to define the number of hidden units
NUM_CLASSES = 2


def run(args):
    """ This method runs the training. """
    input_dir = args.data_dir
    output_dir = args.output
    if args.verbose:
        print('Loading the data...')
    train_feat, train_labels, train_ids = load(join_path(input_dir, 'train'), True)
    dev_feat, dev_labels, dev_ids = load(join_path(input_dir, 'dev'), True)
    test_feat, test_labels, test_ids = load(join_path(input_dir, 'test'), True)
    if args.verbose:
        print('Done.')
    input_dim = int(len(train_feat[0]))
    units = [input_dim]
    if args.units is not None:
        units.extend(parse_units(args.units))
    else:
        deg = max(DEGREE, args.num_layers)
        units.extend([pow(2, deg - i) for i in xrange(args.num_layers)])
    units.append(NUM_CLASSES)

    if args.verbose:
        print('The MLP has: {} units'.format(' '.join(map(str, units))))

    train_batch = SimpleBatchGenerator(train_feat, train_labels, train_ids)
    dev_batch = SimpleBatchGenerator(dev_feat, dev_labels, dev_ids)
    test_batch = SimpleBatchGenerator(test_feat, test_labels, test_ids)

    saved_session = None

    best_dev_p1 = .0
    best_dev_mrr = .0

    with tf.Session(graph=tf.get_default_graph()) as session:
        mlp_model = MlpPredictor(units, args.keep_prob)
        trainer = Trainer(mlp_model, optimizer=args.optimizer, loss=args.loss,
                          learning_rate=args.learning_rate, reg_rate=1e-5,
                          keep_prob=args.keep_prob)
        tf.set_random_seed(8)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        no_improvement = 0
        step = 0

        while no_improvement < args.wait_epochs:
            step += 1
            batch_x, batch_labels, _ = train_batch.next_batch(step, args.batch_size)
            loss, _ = trainer.training_step(session, batch_x, batch_labels)

            if step % args.eval_every == 0:
                if args.verbose:
                    print('Training loss: {} at step {}'.format(loss, step))
                dev_batch_x, dev_batch_labels, dev_batch_ids = \
                    dev_batch.batch_by_offset(0, len(dev_ids))
                dev_loss, dev_predictions = \
                    trainer.eval_step(session, dev_batch_x, dev_batch_labels)
                dev_p1, dev_mrr = evaluate_ranking(dev_predictions, dev_labels, dev_batch_ids)

                if best_dev_p1 < dev_p1:
                    no_improvement = 0
                    best_dev_p1 = dev_p1
                    best_dev_mrr = dev_mrr
                    saved_session = saver.save(session, join_path(output_dir, 'model.ckpt'))
                    print('New best dev loss: {}, dev P@1: {}, dev MRR: {}, at step {}'.format(
                        dev_loss, dev_p1,
                        dev_mrr, step))
                    fout_train = open(join_path(output_dir, 'dev_predictions.txt'), 'w')
                    for k in xrange(len(dev_ids)):
                        fout_train.write(dev_ids[k] + '\t' + str(dev_predictions[k][1]) + '\n')
                    fout_train.close()
                else:
                    if args.verbose:
                        print('Dev loss: {}, dev P@1: {}, dev MRR: {}'.format(dev_loss, dev_p1,
                                                                              dev_mrr))
                    no_improvement += 1
                train_batch.randomize_data()
        if args.verbose:
            print('Restoring session from', saved_session)
        saver.restore(session, saved_session)

        test_batch, test_batch_labels, test_batch_ids = test_batch.batch_by_offset(0,
                                                                                   len(test_ids))
        _, test_predictions = trainer.eval_step(session, test_batch, test_batch_labels)

        test_p1, test_mrr = evaluate_ranking(test_predictions, test_labels, test_batch_ids)

        print('Test P1 and MRR: ', test_p1, test_mrr)
        print(
            'RES\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(input_dir, args.units,
                                                         args.learning_rate, 100*best_dev_p1,
                                                         100*best_dev_mrr, 100*test_p1, 100*test_mrr))

        fout_test = open(join_path(output_dir, 'test_predictions.txt'), 'w')
        for k in xrange(len(test_ids)):
            fout_test.write(test_ids[k] + '\t' + str(test_predictions[k][1]) + '\n')
        fout_test.close()


def parse_units(units_str):
    """Converts the string into an array of integers"""
    try:
        units = map(int, units_str.split())
        return units
    except ValueError:
        print("could not pass the hidden units")


def main():
    """ Main method.

    Runs training of a tf model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='folder containing "train", "dev" and "test" files, '
                                           'each file should have tab separated: label, question_id '
                                           'and a feature vector', default='fake_data')
    parser.add_argument('--num_layers', help='number of hidden layers, either this or number of '
                                             'units should be specified', type=int, default=2)
    parser.add_argument('--units',
                        help='number of hidden units, i.e. "512 128 32"', type=str, default=None)
    # training parameters
    parser.add_argument('--optimizer',
                        help='optimizer to be used (SGD, Adagrad, Adam, Momentum), default: SGD',
                        default='SGD')
    parser.add_argument('--loss',
                        help='loss function to be used (cross-entropy, hinge, log), '
                             'default: cross-entropy', default='cross-entropy')
    parser.add_argument('--momentum',
                        help='momentum for momentum optimizer, only needed if momentum '
                             'optimizer is chosen', type=float, default=0.9)
    parser.add_argument('--batch_size', help='minibatch size, default: 100', type=int,
                        default=100)

    parser.add_argument('--keep_prob', help='dropout keep probability', type=float, default=1.0)
    parser.add_argument('--reg_rate', help='L2 regularization rate', type=float, default=1e-06)
    parser.add_argument('--learning_rate', help='initial learning rate', type=float, default=0.001)
    parser.add_argument('--wait_epochs', help='wait that many epochs for improvement, default: 15',
                        type=int, default=15)
    parser.add_argument('--eval_every',
                        help='how often (in steps) the evaluation on the dev set is '
                             'performed, default: 1000', type=int, default=400)

    parser.add_argument('--output', type=str, default='output folder where the model and the '
                                                      'predictions are saved')

    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    run(args)


main()
