# Testing
For testing::
```console
python test.py --config config_test.ini
```

In config_test.ini, you configure the testing and where the results are saved (Also contained in the output of the program). <br />

Structure of config_test.ini
```
[DataExchange]
basedir = [Absolute path to the directory where the snapshot is saved on which the model was last trained.]
basedirTest =  [Absolute path to the directory where the snapshot is on which the model is tested.]


[WorkParameter]
guard = [Maximum number of vertices in the batch.]
epochs = [Number of epochs.]
cuda_core = [cuda:0 or cpu (for torch)]
summary = [1: 1-Hop AC (only for MLP), 2: 2-Hop AC, 3: 2-Hop AC and Edges.]
val_step = [Number of epochs between validations and the start of the first validation.]
test = [0;1]
validation = [0;1]
training = [0;1]
# test + validation + training = 1
[GNN]
h_layers = [Number of hidden layers.]
weight_decay = 0.0005
model_name = [graphmlp, grapsaint, mlp, grapsaintN]
learning_rate = [Size of the hidden layers.]
dropout = [Value for the dropout.]
#GraphMLP
tau = [Value for tau (only for GraphMLP).]
alpha = [Value for alpha (only for GraphMLP).]
k_hop=1

```