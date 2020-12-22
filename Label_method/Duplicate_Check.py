import pandas as pd
import numpy as np

df=pd.DataFrame(np.load('dribble_data.npy').reshape(192,40*25*3))
new_df =df.drop_duplicates()
print(new_df)