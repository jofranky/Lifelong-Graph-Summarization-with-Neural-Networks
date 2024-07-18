# Filter
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