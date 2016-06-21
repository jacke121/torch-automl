import pandas as pd
import h5py
import sys

df = pd.read_csv(sys.argv[1])

with h5py.File(sys.argv[2], 'w') as hf:
	  hf.create_dataset(sys.argv[3], data=df)

columnNames = str(df.columns.tolist()).replace('[','').replace(']','')
print(columnNames)