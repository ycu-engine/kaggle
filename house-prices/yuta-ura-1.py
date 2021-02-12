"""
以下のチュートリアルを参考にサブミッションまでやってみる
https://www.kaggle.com/katotaka/kaggle-prediction-house-prices
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# %%
parent_path = Path().parent.absolute()
data_dir = parent_path/'data'
submission_dir = parent_path/'submission'


# %%
df = pd.read_csv(data_dir / 'train.csv')


# %%

df
# %%
df.head()
# %%

X = df[["OverallQual"]].values
y = df["SalePrice"].values

# アルゴリズムに線形回帰(Linear Regression)を採用
slr = LinearRegression()

# fit関数でモデル作成
slr.fit(X, y)

# 偏回帰係数(回帰分析において得られる回帰方程式の各説明変数の係数)を出力
# 偏回帰係数はscikit-learnのcoefで取得
print('傾き：{0}'.format(slr.coef_[0]))

# y切片(直線とy軸との交点)を出力
# 余談：x切片もあり、それは直線とx軸との交点を指す
print('y切片: {0}'.format(slr.intercept_))
# %%
plt.scatter(X, y)

# 折れ線グラフを描画
plt.plot(X, slr.predict(X), color='red')

# 表示
plt.show()
# %%

df_test = pd.read_csv(data_dir / 'test.csv')

# %%
df_test.head()
# %%
# No.9
# テストデータの OverallQual の値をセット
X_test = df_test[["OverallQual"]].values

# 学習済みのモデルから予測した結果をセット
y_test_pred = slr.predict(X_test)
# %%
y_test_pred
# %%
df_test["SalePrice"] = y_test_pred
# %%
df_test.head()
# %%
df_test[["Id", "SalePrice"]].head()
# %%
