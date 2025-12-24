import pandas as pd
import numpy as np

import glob

files=glob.glob(r'K:\项目\Transunet改进\Unet\\\/*/*/evaluation_results.csv')

for file in files:
    da=pd.read_csv(file)
    print('#'*100)
    print(file.split('\\')[4])
    print(da)



