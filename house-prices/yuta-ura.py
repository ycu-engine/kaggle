# %%
from pathlib import Path

import pandas as pd

# %%
data_dir = (Path().parent / 'data').absolute()


# %%

df = pd.read_csv(data_dir / 'train.csv')

df.head()

# %%

df.describe()
# %%
