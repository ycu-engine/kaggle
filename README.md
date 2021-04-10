# kaggle の分析ソースコード共有

## Colab テンプレート

1. 以下のリンクからファイルを開く
    - https://colab.research.google.com/github/ycu-engine/kaggle/blob/main/kaggle_template.ipynb
2. 「ファイル」メニューから「ドライブにコピー保存」を選択する

## クローン

`Github Desktop`でこのレポジトリの URL を指定するか、`git clone <このURL>`とする

## python 3.8 系のインストール

- ターミナルで`pyenv --version`として数字が表示される場合

  `pyenv install 3.8.6`として最新の 3.8 系をインストールする

- ターミナルで`pyenv --version`として数字が表示されないが、`python --version`とした時に 3.8 系以外が表示された場合

  - 表示されたバージョンが 3 系の場合(2 系の場合は飛ばして良い)

    今インストールされている Python をアンインストールする

  - pyenv をインストールする。

    やり方は各自でググってください（「Windows pyenv インストール」のように OS も含めて検索してください）

  pyenv をインストールできたら`ターミナルで pyenv --version として数字が表示される場合`の項目を実行してください

## pipenv をインストール

ターミナルで`pipenv --version`とした時にバージョンが表示されない場合は

`pip install pipenv`としてインストールしてください

## 依存モジュールなどのインストール

`pipenv install`として依存モジュールのインストールをしてください

# Tips

## ライブラリを追加する場合

`pipenv install numpy`のようにしてください。`pip install numpy`のようにはしないでください。

# ファイルの書き方

py ファイルの方はそのままでもソースコードを読むこと書き、管理するのに最適なのでできれば py ファイルで作成して欲しいです。

## py ファイルでの書き方（推奨）

以下のように書くと py ファイルでも Jupyter のようにセルごとに実行することができる

```python
# %%
import pandas as pd

# %%

df = pd.read_csv('./data/train.csv')

df.head()

# %%

df.describe()
# %%

```

## jupyter での書き方

`pipenv run jupyter-lab`と実行すると jupyter が立ち上がります。
そこでファイルを作成し実行してみてください。
