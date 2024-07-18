# Measures
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
