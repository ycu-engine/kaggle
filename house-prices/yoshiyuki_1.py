# %%
# %%
import matplotlib.pyplot as plt
import pandas as pd
# %%
import seaborn as sns

# %%
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
df = pd.concat([df_train, df_test])
# %%
train_ID = df_train['Id']
test_ID = df_test['Id']
 
df_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)
# %%
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
# %%
# データのうちの欠損値の割合を求める
all_data_na_ratio = (all_data.isnull().sum() / len(all_data)) * 100
# 欠損値がない列を削除し、降順に並び替える
all_data_na_ratio = all_data_na_ratio.drop(all_data_na_ratio[all_data_na_ratio == 0].index).sort_values(ascending=False)[:30]

all_data_na_count = all_data.isnull().sum()
all_data_na_count = all_data_na_count.drop(all_data_na_count[all_data_na_count == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na_ratio, 'Missing Counts':all_data_na_count})
print(missing_data.shape)
missing_data
# %%
# PoolQC,Alley, MiscFeature,Fence, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, BsmtFinType1, BsmtFinType2, 'BsmtCond','BsmtQual'
# これらのNAはそれがないことを示すので、Noneで補完する
cols = ['Alley', "PoolQC", "MiscFeature" ,"Fence", "FireplaceQu", 'GarageType','GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtFinType1', 'BsmtFinType2', 'BsmtCond','BsmtQual']
for col in cols:
     all_data[col] = all_data[col].fillna('None')

# Utilitiesの中身は”NoSeWa”1つとNA2つで構成されており、”NoSeWa”は学習データにしか存在しない。
# よって、このカラムはテストデータの予測には役に立たないので、カラムごと削除する。
# all_data = all_data.drop(['Utilities'], axis=1)


# Functional。データの説明によると、NAは一般的であることを示している。よって、欠損値は文字列”Typ”で補完する。
all_data["Functional"] = all_data["Functional"].fillna("Typ")

# Electrical。欠損値は1つだけ。最頻値(SBrkr)で補完する。
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# KitchenQual。欠損値は1つだけ。最頻値(TA)で補完する。
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

# Exterior1st, Exterior2nd。いずれも欠損値は1つだけ。最頻値で補完する。
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# SaleType。欠損値は1つだけ。最頻値で補完する。
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# GarageArea欠損値は1つだけ。最頻値で補完する。
all_data['GarageArea'] = all_data['GarageArea'].fillna(all_data['GarageArea'].mode()[0])

# GarageCars。欠損値は1つだけ。最頻値で補完する。
all_data['GarageCars'] = all_data['GarageCars'].fillna(all_data['GarageCars'].mode()[0])

# BsmtFinSF2。欠損値は1つだけ。最頻値で補完する。
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(all_data['BsmtFinSF2'].mode()[0])

# BsmtUnfSF。欠損値は1つだけ。最頻値で補完する。
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(all_data['BsmtUnfSF'].mode()[0])

# TotalBsmtSF。欠損値は1つだけ。最頻値で補完する。
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(all_data['TotalBsmtSF'].mode()[0])

# BsmtFinSF1。欠損値は1つだけ。最頻値で補完する。
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(all_data['BsmtFinSF1'].mode()[0])

# MSZoning。欠損値は4つだけ。最頻値で補完する。
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# BsmtHalfBath。欠損値は2つだけ。最頻値で補完する。
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(all_data['BsmtHalfBath'].mode()[0])

# BsmtFullnath。欠損値は2つだけ。最頻値で補完する。
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(all_data['BsmtFullBath'].mode()[0])

# MasVnrType。最頻値で補完する。
all_data['MasVnrType'] = all_data['MasVnrType'].fillna(all_data['MasVnrType'].mode()[0])

# MasVnrArea。中央値で補完する。
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].median())

# GarageYrBlt。中央値で補完する。
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(all_data['GarageYrBlt'].median())

# LotFrontage	。平均値で補完する。
all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].mean())

# BsmtExposure。最頻値で補完する。
all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna(all_data['BsmtExposure'].mode()[0])


# MSSubClass。欠損値は建物クラスなしに近い。よって、欠損値をNoneで補完する。
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

# Utilities。欠損値は2つ。よって、最頻値で補完する。
all_data['Utilities'] = all_data['Utilities'].fillna(all_data["Utilities"].mode()[0])
# %%
all_data.isnull().sum()[all_data.isnull().sum() > 0]
# %%
import numpy as np

# %%
corrmat = df_train.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# %%
# 比例尺度でない数値データの型を変換する
all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)
all_data['OverallCond'] = all_data["OverallCond"].astype(str)
all_data['OverallQual'] = all_data["OverallQual"].astype(str)
all_data["YrSold"] = all_data["YrSold"].astype(str)
all_data["MoSold"] = all_data["MoSold"].astype(str)
# %%
# 量的変数の標準化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 数値データ
all_data_num = all_data.select_dtypes('number')
# カテゴリカルデータ
all_data_str = all_data.select_dtypes("object")

scaler.fit(all_data_num)
scaler.transform(all_data_num)
all_data_num_std = pd.DataFrame(scaler.transform(all_data_num), columns=all_data_num.columns)

# %%
# ダミー変数化
all_data_std = pd.concat([all_data_num_std,all_data_str],axis = 1)
all_data_std = pd.get_dummies(all_data_std)

# %%
# 多重共線性のある説明変数を消去
all_data_std = all_data_std.drop(["TotRmsAbvGrd", "GarageArea", "TotalBsmtSF"],axis=1) 

# %%
# trainとtestに分割
train_x = all_data_std.iloc[:1460,:]
test_x = all_data_std.iloc[1460:,:]
# %%
# 目的変数の分布を確認
sns.distplot(df_train['SalePrice'])
# %%
# 対数変換して可視化
import numpy as np

sns.distplot(np.log(df_train['SalePrice']))

# %%
# 目的変数を対数変換
train_y = pd.DataFrame(np.log1p((df_train['SalePrice'])))

# %%
display(train_x)
display(test_x)
# %%
# 予測
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_x, train_y)

# %%
y_pred = pd.DataFrame(lr.predict(test_x), columns = ["SalePrice" ])
y_pred
# %%
saleprice_pred = np.exp(y_pred)
pred1 = pd.concat([test_ID,saleprice_pred],axis = 1)
pred1 = pred1.set_index("Id")
pred1
# %%
# csvを出力
# pred1.to_csv("./house-prices-submission/yoshiyuki1_submission.csv")
# %%
# import torch
# %%
