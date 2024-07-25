# Lifelong Graph Summarization with Neural Networks



## Downloading the DyLDO dataset

The snapshots from the DyLDO dataset can be downloaded from http://km.aifb.kit.edu/projects/dyldo/data. You can choose a snapshot and download its data.nq.gz. We propose that you put the file in a directory with the date of the snapshot, like 2012-05-06. The directory is used to save all results, models, and measures for the snapshot.


## Required Packages

A list of required packages to be imported is provided in the **requirements.txt** file.

## Executing the Code
All code used for the experiments is in the directory 'src'.

# Preprocessing

## Filter
Execute the code in  **src/filter** first to filter the downloaded snapshot. <br /> 
For filtering a snapshot run:
```console
python filterRDF.py --config config_summaryRDF.ini
```
In config_summaryRDF.ini, you configure which snapshot you want to filter and how. <br /> <br />

Structure of config_summaryRDF.ini:
```
[DataExchange]
basedir = [Absolute path to the directory of the snaphsot]

[Dyldo]
raw_datafile = [The snapshot name like data2022-09-25.nq.gz]
filtered_datafile = [The name of the filtered snapshot like data-filtered-no-duplicates2022-09-25.nq (no longer gzipped)]
trashed_datafile = [The name of the file that contains all fault quads data-trash2022-09-25.nq]
num_lines = [The number of quads in the snapshot. It does not have to be known in advance. The code sets it if nun_lines_counted = False.]
nun_lines_counted = [True: restart the previous run. False: Start filtering and set num_lines.]
begin_line = [1 for starting the filtering in the beginning. The code saves the last read line in 100,000 steps.]
read_lines = [The number of lines that should be filtered in the snapshot. The maximum is always the number of lines.]
finished =  [True: If every line was filtered (Set by program). False: Not every line is filtered (The program can filter the rest).]
pre_skolemize = True
```

To eliminate duplicates run, e.g., in Ubuntu:
```console
awk ’!seen[$0]++’ data-filtered.nq > data-filtered-no-duplicates.nq
```

## Creating the Summaries
Then, run the code in **src/summary** to create the summaries and meta-information of the snapshots.<br />
For creating the summaries of a snapshot run:
```console
python summaryRDF.py --config config_summaryRDF.ini
```
In config_summaryRDF.ini, you configure for which snapshot you want the summaries. <br /> <br />

Structure of config_summaryRDF.ini:
```
[DataExchange]
basedirP = [Absolute path to the directory of the predecessor snapshot.]
basedirC = [Absolute path to the directory of the current snapshot.]

[Dyldo]
filtered_datafileP =  [The name of the filtered predecessor snapshot.]
filtered_datafileC = [The name of the filtered current snapshot for which the summaries are created.]

[GraphSummary]
Prev = [True if the predecessor  is used; otherwise False.]
save_fileP = [The name of the file in which the information of the summaries of the previous snapshot was saved. (e.g., graph_data_gs2012-05-06)]
save_fileC = [The name of the file in which the information of the summaries of the current snapshot is saved. (e.g., graph_data_gs2012-05-13)]
save_fileT = [The name of the file in which the data for training and testing is saved. (e.g., t2012-05-13)]

[WorkParameter]
load_data = False
```
 
## Creating the subgraphs
Execute the code in **src/createSubgraphs** to create the subgraphs later used for validation, training, and testing.<br />

For creating the subgraphs of a snapshot run:
```console
python createSubgraphs.py --config config_summaryRDF.ini
```
In config_summaryRDF.ini, you configure for which snapshot you want the subgraphs. <br /> <br />

Structure of config_summaryRDF.ini:
```
[DataExchange]
basedirC = [Absolute path to the directory of the snapshot.]

[WorkParameter]
maxDegree = [Maximum number of degrees a vertex can have to be sampled.]

[GraphSummary]
save_fileT = [The name of the file in which the information for training and testing is saved.]
```

# Measuring
## Measures
For the measures, first, execute the code in **src/measures**.  <br />
For calculating the unary measures of a snapshot run:
```console
python unary_measures.py --config config_summaryRDF.ini
```
For calculating the unary measures of a snapshot run:
```console
python binary_measures.py --config config_summaryRDF.ini
```
In config_summaryRDF.ini, you configure for which snapshot you want to calculate the unary measures. <br />
All plots and values are saved in the directory of basedirP.  <br /> <br />

Structure of config_summaryRDF.ini:
```
[DataExchange]
basedirP = [Absolute path to the directory of the predecessor snapshot.]
based = [Absolute path to the directory of the current snapshot.]

[Dyldo]
filtered_datafileP =  [The name of the filtered predecessor snapshot.]
filtered_datafileC = [The name of the filtered current snapshot for which the summaries are created.]

[GraphSummary]
Prev = [True if predecessor  is used (binary measures); otherwise False.]
save_fileP = [The name of the file in which the information of the summaries of the previous snapshot was saved. (e.g., graph_data_gs2012-05-06)]
save_fileC = [The name of the file in which the information of the summaries of the current snapshot is saved. (e.g., graph_data_gs2012-05-13)]
save_fileT = [The name of the file in which the data for training and testing is saved. (e.g., t2012-05-13)]

[WorkParameter]
load_data = False
plot = [True: Create distribution plots.]
```

## Meta-Plots
Run the code in **Meta-plots**.<br />
For calculating the plots for the changes in the snapshots:
```console
python plot.py --config config_plot.ini
```
```console
python plot2.py --config config_plot.ini
```

In config config_plot.ini, you configure which snapshots are used from 1 to filesN and their order. <br />

Structure of config config_plot.ini:
```
[DataExchange]
basedir =  [Absolute path to the directory where the measures are saved.]
basedir1 =   [Absolute path to the directory of snapshot 1.]
basedir2 =  [Absolute path to the directory of snapshot 2.]
basedir3 =  [Absolute path to the directory of snapshot 3.]
basedir4 =  [Absolute path to the directory of snapshot 4.]
basedir5=   [Absolute path to the directory of snapshot 5.]
basedir6 =  [Absolute path to the directory of snapshot 6.]
basedir7 =  [Absolute path to the directory of snapshot 7.]
basedir8 =  [Absolute path to the directory of snapshot 8.]
basedir9 =  [Absolute path to the directory of snapshot 9.]
basedir10 =  [Absolute path to the directory of snapshot 10.]

[GraphSummary]
filesN = [Number of snapshots. Here 10.]
save_file1 = [The name of snapshot 1.]
save_file2 =  [The name of snapshot 2.]
save_file3 =  [The name of snapshot 3.]
save_file4 =  [The name of snapshot 4.]
save_file5 =  [The name of snapshot 5.]
save_file6 =  [The name of snapshot 6.]
save_file7 =  [The name of snapshot 7.]
save_file8 =  [The name of snapshot 8.]
save_file9 =  [The name of snapshot 9.]
save_file10 =  [The name of snapshot 10.]

[WorkParameter]
load_data = [True: Data are already calculated and saved. Otherwise, it must be False.]
name = [Name of the folder where the plots are saved.]
```


## Validation
The code for the validation is in **src/gnn_validating**.<br />
For validation:
```console
python trainValidate.py --config config_trainValidate.ini
```

In config_trainValidate.ini, you configure the validation and where the results are saved (Also contained in the program output). <br />

Structure of config_trainValidate.ini
```
[DataExchange]
basedirC = [Absolute path to the directory where the snapshot is saved.]


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
## Training
The code for the training is in **src/gnn-training**.<br />
For training:
```console
python train.py --config config_train.ini
```

In config_train.ini, you configure the training and where the trained model is saved. <br />

Structure of config_train.ini
```
[DataExchange]
basedirC = [Absolute path to the directory where the snapshot is saved on which the model is trained.]
basediP =  [Absolute path to the directory where the snapshot is saved on which the model was last trained.]


[WorkParameter]
loadM = [False: Model is trained from scratch.]
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

## Testing
The code for the testing is in **src/gnn-testing**.<br />
For testing:
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

## Evaluataion
The code for the evaluation is in **src/evaluation**.<br />

For creating the LaTex Tables  in **accuracy** and **measures**:
```console
python evaluation.py --config config_eval.ini
```

For creating the heatmaps in **heatmaps**:
```console
python evaluationC.py --config config_eval.ini
```

For creating the LaTex Tables  in **F1**:

```console
python evaluationF1.py --config config_eval.ini
```

For creating the plots for the distribution of the F1 values  in **F1**:

```console
python evaluationF1D.py --config config_eval.ini
```

In config_eval.ini, you config which snapshots are used from 1 to filesN and also their order. <br />

Structure of config_eval.ini:
```
[GraphSummary]
basedir = basedir =  [Absolute path to the directory where the snapshots are saved.]
Summaries =  [Number of snapshots. Here 10.]
summary1 = [Directory of snapshot 1.]
summary2 = [Directory of snapshot 2.]
summary3 = [Directory of snapshot 3.]
summary4 = [Directory of snapshot 4.]
summary5 = [Directory of snapshot 5.]
summary6 = [Directory of snapshot 6.]
summary7 = [Directory of snapshot 7.]
summary8 = [Directory of snapshot 8.]
summary9 = [Directory of snapshot 9.]
summary10 = [Directory of snapshot 10.]

[WorkParameter]
summary = [1: 1-Hop AC (only for MLP), 2: 2-Hop AC, 3: 2-Hop AC and Edges.]
model_name = [graphmlp, grapsaint, mlp, grapsaintN]

```

For creating the plots and LaTex tables for forgetting in **forgetting**:

```console
python evaluation.py --config config_evalF.ini
```
In config_evalF.ini, you config which snapshots are used from 1 to filesN and also their order. <br />

Structure of config_eval.ini:
```
[GraphSummary]
basedir = /home/ul/ul_student/ul_bkj10/TemporalGraphSummaries/data
Summaries =  [Number of snapshots. Here 10.]
summary1 = [Directory of snapshot 1.]
summary2 = [Directory of snapshot 2.]
summary3 = [Directory of snapshot 3.]
summary4 = [Directory of snapshot 4.]
summary5 = [Directory of snapshot 5.]
summary6 = [Directory of snapshot 6.]
summary7 = [Directory of snapshot 7.]
summary8 = [Directory of snapshot 8.]
summary9 = [Directory of snapshot 9.]
summary10 = [Directory of snapshot 10.]

[WorkParameter]
names = [Number of pairs of summary and model. Here 5.]
summary1 = 1
model_name1 = mlp
summary2 = 2
model_name2 = mlp
summary3 = 2
model_name3 = graphmlp
summary4 = 2
model_name4 = graphsaint
summary5 = 3
model_name5 = graphsaint
```

## Authors and Acknowledgment
We thank  Maximilian Blasi, Manuel Freudenreich, and Johannes Horvath for providing their code to us. 
Their code is the foundation of our code, which we adapted and expanded for our project.

