import json
import pandas as pd


df = pd.read_json('avg.json',orient='records')

df.to_csv('avg.csv',index=True)

