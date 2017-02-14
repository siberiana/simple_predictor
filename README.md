# Simple MLP predictor

This is an implementation of a multilayer perceptron.

## Requires
- Tensorflow 0.12


## How to use
```bash
python tool_run_mlp.py --data_dir [path_to_your_data] --units "256 128 64 32 16" --learning_rate 0.01 --output [path_to_output_dir]
```

Print parameters:

```bash
python tool_run_mlp.py --help
```

```
arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   folder containing "train", "dev" and "test" files, each file
                        should have tab separated label, question_id and a feature vector
  --num_layers NUM_LAYERS
                        number of hidden layers, either this or number of
                        units should be specified
  --units UNITS         number of hidden units in each layer, as a string, i.e. "512 128 32"
  --optimizer OPTIMIZER
                        optimizer to be used (SGD, Adagrad, Adam, Momentum),
                        default: SGD
  --loss LOSS           loss function to be used (cross-entropy, hinge, log),
                        default: cross-entropy
  --momentum MOMENTUM   momentum for momentum optimizer, only needed if
                        momentum optimizer is chosen
  --batch_size BATCH_SIZE
                        minibatch size, default: 100
  --keep_prob KEEP_PROB
                        dropout keep probability
  --reg_rate REG_RATE   L2 regularization rate
  --learning_rate LEARNING_RATE
                        initial learning rate
  --wait_epochs WAIT_EPOCHS
                        wait that many epochs for improvement, default: 15
  --eval_every EVAL_EVERY
                        how often (in steps) the evaluation on the dev set is
                        performed, default: 1000
  --output OUTPUT output folder where the model and the predictions are saved
  --verbose
```