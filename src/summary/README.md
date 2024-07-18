# Creating the summaries
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