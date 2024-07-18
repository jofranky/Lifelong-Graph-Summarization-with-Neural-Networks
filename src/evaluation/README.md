# Evaluation
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