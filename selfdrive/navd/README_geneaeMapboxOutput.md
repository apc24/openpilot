# generateMapboxOutput.py
ナビのルートを強制的に設定します。

## 使い方
* google map(マイマップ)でlineツールでルートを描いて、Exportする
* Exportしたファイルを元にroute.csvを編集する。曲がり角はtypeをleft or rightに、到着点はarriveにして、それ以外はstにする。
* set_destination.pyの到着点も同じ座標に設定する
* PCからcomma.3Xに直接ログインし、環境変数を以下のように設定する。
```bash
export LOADCSVMAP=TRUE
```
* set_destination.pyを実行する

