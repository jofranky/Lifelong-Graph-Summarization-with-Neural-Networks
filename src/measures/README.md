# Measures
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
