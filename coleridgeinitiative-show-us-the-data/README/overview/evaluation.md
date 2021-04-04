このコンテストの目的は、科学出版物の中でデータセットについて言及しているかどうかを特定することです。あなたの予測は、データセットに言及していると思われる出版物からの短い抜粋です。

投稿作品は、予測テキストとグランドトゥルーステキストの間の[ジャカードベースの FBeta](https://en.wikipedia.org/wiki/Jaccard_index)スコアで評価され、`Beta = 0.5`（`マイクロ F0.5` スコア）となります。複数の予測結果がある場合は、投稿ファイルにパイプ(`|`)文字で区切られています。

以下は、1 つのグランドトゥルース文字列に対する 1 つの予測文字列の Jaccard スコアを計算する Python コードです。なお、サンプルの総合スコアは、パイプで区切られた複数のグランドトゥルース文字列と予測文字列を比較するために Jaccard を使用していますが、このコードではその処理や最終的な`マイクロF-β`の計算は行いません。

```python
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

なお、すべてのグランドトゥルースのテキストは、以下のコードを用いてマッチングのためにクリーニングされています。

```python
def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())
```

各出版物の予測セットについて，予測の可能性があるものとグランドトゥルースの組み合わせごとに，トークンベースのヤカードスコアが計算されます。あるグランドトゥルースに対して最も高いスコアを持つ予測が、そのグランドトゥルースと**照合されます**。

- 各出版物の予測された文字列は、アルファベット順にソートされ、その順に処理されます。スコアが同点の場合は、そのソートに基づいて解決されます。
- 一致した予測のうち、Jaccard スコアが閾値である 0.5 以上を満たすものを真陽性（`TP`）、残りを偽陽性（`FP`）としてカウントする。
- 一致しない予測は、偽陽性（`FP`）としてカウントされます。
- 最寄りの予測値がないグランドトゥルースは、偽陰性（`FN`）としてカウントされます。

すべてのサンプルの`TP`、`FP`、`FN`を使用して、最終的な`マイクロ F0.5` スコアを算出する。(`マイクロ` F スコアは正確には、`TP`、`FP`、`FN` の 1 つのプールを作成し、それを予測セット全体のスコアを計算するために使用することに注意してください)

### 提出ファイル

テストセットの各出版物 ID に対して，`PredictionString` 変数の抜粋（複数の抜粋をパイプ文字で区切ったもの）を予測する必要があります。ファイルにはヘッダーが含まれ、以下の形式になっています。

```csv
Id,PredictionString
000e04d6-d6ef-442f-b070-4309493221ba,space objects dataset|small objects data
0176e38e-2286-4ea2-914f-0583808a98aa,small objects dataset
01860fa5-2c39-4ea2-9124-74458ae4a4b4,large objects
01e4e08c-ffea-45a7-adde-6a0c0ad755fc,space location data|national space objects|national space dataset
01fea149-a6b8-4b01-8af9-51e02f46f03f,a dataset of large objects
etc.
```