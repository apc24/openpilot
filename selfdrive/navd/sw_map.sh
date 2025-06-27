echo APC_openpilotフォルダで以下のコマンドで実行することを想定しています。
echo source selfdrive/navd/sw_map.sh


SCRIPT_DIR="/home/kou/openpilot/APC_openpilot/selfdrive/navd"


echo -n "Which map do you use? [1:MapBox / 2:OpenStreetMap]: "
read res

case $res in
  "" | 1 | MapBox )
    export MAPS_HOST='https://api.mapbox.com'

    cp $SCRIPT_DIR/style_mb.json $SCRIPT_DIR/style.json
    ;;
  * )
    export MAPS_HOST='https://tile.openstreetmap.jp'
    cp $SCRIPT_DIR/style_osm.json $SCRIPT_DIR/style.json
    ;;
esac

