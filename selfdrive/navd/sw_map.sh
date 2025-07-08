echo APC_openpilotフォルダで以下のコマンドで実行することを想定しています。
echo source selfdrive/navd/sw_map.sh


SCRIPT_DIR="selfdrive/navd"


echo -n "Which map do you use? [1:MapBox / 2:OpenStreetMap / 3:Mapbox Routing Only]: "
read res

case $res in
  "" | 1 | MapBox )
    export MAPS_HOST='https://api.mapbox.com'
    cp $SCRIPT_DIR/style_mb.json $SCRIPT_DIR/style.json
    echo "MapBox (マップ表示 + ルーティング) を選択しました"
    ;;
  2 | OpenStreetMap )
    export MAPS_HOST='https://tile.openstreetmap.jp'
    cp $SCRIPT_DIR/style_osm.json $SCRIPT_DIR/style.json
    echo "OpenStreetMap を選択しました"
    ;;
  3 )
    # マップ表示はMapLibre、ルーティングはMapbox
    echo "Mapbox ルーティングのみを選択しました（マップ表示は無料タイル使用）"
    echo "MAPBOX_TOKEN環境変数が設定されていることを確認してください"
    ;;
  * )
    export MAPS_HOST='https://tile.openstreetmap.jp'
    cp $SCRIPT_DIR/style_osm.json $SCRIPT_DIR/style.json
    echo "デフォルト: OpenStreetMap を選択しました"
    ;;
esac

