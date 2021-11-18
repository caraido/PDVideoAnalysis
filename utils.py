import pandas as pd

def emit_nan(x,y):
    x['labels']=y.tolist()

    new_df= x.dropna()
    new_y=new_df['labels']
    new_x=new_df.drop(columns=['labels'])
    return new_x, new_y