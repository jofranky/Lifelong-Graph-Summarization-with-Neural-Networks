# Creating the subgraphs
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