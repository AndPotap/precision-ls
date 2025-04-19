```shell
py src/train.py --config src/conf/least_squares/least_squares.yaml --autoname False --model.n_layer 1 --model.seq_op "base_conv" --training.train_steps 1000000 --training.scheduler_steprate 10000 --training.learning_rate 1e-4 --wandb.name task="test" --test_run=True
```
