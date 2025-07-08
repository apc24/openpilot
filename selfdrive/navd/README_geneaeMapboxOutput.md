# generateMapboxOutput.py
ナビのルートを強制的に設定します。

## 使い方
* google map(マイマップ)でlineツールでルートを描いて、Exportする
* Exportしたファイルを元にroute.csvを編集する。曲がり角はtypeをleft or rightに、到着点はarriveにして、それ以外はstにする。
* set_destination.pyの到着点も同じ座標に設定してset_destination.pyを実行する。(座標が違う場合は通常通りMAPBOXのdirection APIでルートが設定される。)

