import os
import csv
import json
import math
from typing import List, Dict, Tuple
from openpilot.common.swaglog import cloudlog
import numpy as np
from openpilot.selfdrive.navd.nearestLink import nearestLink


SPD = 30 #[km/h] vehicle speed
#CSVFILE = 'selfdrive/navd/route.txt'
CSVFILE = 'route.txt'

def load_csv(file_path: str) -> List[Tuple[float, float, str, str, float]]:
    """
    Load points from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        List[Tuple[float, float, str, str]]: A list of tuples containing
        longitude, latitude, type, and modifier.
        ファイルが見つからなかった場合やエラー時は空リストを返す。
    """
    points = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
#            points = []
            for row in reader:
                lon = float(row['lon'])
                lat = float(row['lat'])
                type_ = row['type']
                modifier = row['modifier']
                maxspeed = float(row['maxspeed'])
                points.append((lon, lat, type_, modifier, maxspeed))
            return points
    except Exception as e:
        cloudlog.error(f"load_csv error: {e}")
        return points

def generate_mapbox_output(points: List[Tuple[float, float, str, str, float]], start_index: int = 0) -> dict:
    """
    Generate a Mapbox JSON output from a list of points.
    Args:
        points (list): List of tuples containing longitude, latitude, type, modifier and maxspeed.
        start_index (int): Index to start processing points from.
    Returns:
        dict: A dictionary representing the Mapbox JSON output.
    """

    steps: List[Dict] = []
    maxspeed: List[Dict] = []
    coordinates: List[List[float]] = []

    for i in range(start_index, len(points)):
        current = points[i]
        coordinates.append([current[0], current[1]])
        t_mxspd = current[4]
        if t_mxspd < 0:
            maxspeed.append({"unknown": True})
        else:
            maxspeed.append({"speed": t_mxspd, "unit": "km/h"})

        if current[3] != "st":
            distance = calculate_distance(coordinates)
            spd = max(SPD, 1)
            duration = distance / (spd * 1000 / 3600)
            type_ = current[2]
            modifier = current[3]
            if type_ == "arrive":
                text = f"Your destination is one the {modifier}"
            else:
                text = f"Turn {modifier}"
            step = {
                "bannerInstructions": [{
                    "primary": {
                        "type": type_,
                        "modifier": modifier,
                        "text": text
                    },
                    "distanceAlongGeometry": distance
                }],
                "speedLimitSign": "vienna",
                "duration_typical": duration,
                "duration": duration,
                "distance": distance,
                "geometry": {"coordinates": coordinates}
            }
            steps.append(step)
            coordinates = [[current[0], current[1]]]

    routes = {
        "routes": [{
            "legs": [{
                "annotation": {
                    "maxspeed": maxspeed,  # MAPBOXのmaxpeedは最初の8個しかないが、困らないはずなので全部入れる
                },
                "steps": steps
            }]
        }]
    }

    return routes


EARTH_RAD = 6378137 # [m] 地球の半径
def cal_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the distance between two geographical points using the Haversine formula.
    Args:
        lon1 (float): Longitude of the first point.
        lat1 (float): Latitude of the first point.
        lon2 (float): Longitude of the second point.
        lat2 (float): Latitude of the second point.
    Returns:
        float: Distance in meters between the two points.
    参考: https://qiita.com/Yuzu2yan/items/0f312954feeb3c83c70e
    """
    th = math.sin(math.radians(lat1))*math.sin(math.radians(lat2)) + \
        math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.cos(math.radians(lon2-lon1))
    th = min(1.0,max(-1.0,th))

    return EARTH_RAD * math.acos(th)

def calculate_distance(coordinates: List[List[float]]) -> float:
    """
    coordinates: 各要素が [lon, lat] の2つのfloat値からなるリストのリスト
    戻り値: 総距離（メートル）
    """
    distance = 0.0  # 修正: float型で初期化
    for i in range(len(coordinates)-1 ):
        distance += cal_distance(coordinates[i][0],coordinates[i][1],coordinates[i+1][0],coordinates[i+1][1])

    return distance


def genMapboxJson(pos: dict[str, float]) -> dict:
    """
    Generate Mapbox JSON output based on the current position and route points.
    Args:
        pos (dict): A dictionary containing 'lon' and 'lat' keys for the current position.
    Returns:
        dict: A dictionary representing the Mapbox JSON output.

    """
    if 'lon' not in pos or 'lat' not in pos:
        cloudlog.error("genMapboxJson error: 'lon'または'lat'がposに存在しません")
        return {}

    # スクリプトの絶対パスを取得
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # 読み込みたいファイルの相対パスを指定
    file_path = os.path.join(script_dir, CSVFILE)

    points = load_csv(file_path)
    mat = np.zeros([len(points),2])
    for idx in range(len(points)):
        mat[idx,:]=np.array([points[idx][0],points[idx][1]])

    p0 = np.array([[pos['lon'], pos['lat']]])

    [p,idx]=nearestLink(mat,p0)

    maxspeed0 = points[idx][4]
    points = points[(idx+1):]
    point0 = (float(p[0]),float(p[1]),'st','st',maxspeed0)
    points.insert(0,point0)
    output = generate_mapbox_output(points)
    return output

def checkDestination(pos: dict[str, float]) -> bool:
    """
    Check if the destination is equal to the list.
    Args:
        pos (dict): A dictionary containing 'lon' and 'lat' keys for the current position.
    Returns:
        bool: True if the position is close to the destination, False otherwise.
    """
    if 'lon' not in pos or 'lat' not in pos:
        cloudlog.error("checkDestination error: 'lon'または'lat'がposに存在しません")
        return False

    # スクリプトの絶対パスを取得
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # 読み込みたいファイルの相対パスを指定
    file_path = os.path.join(script_dir, CSVFILE)

    points = load_csv(file_path)

    # pointsが空でないか確認
    if not points:
        cloudlog.error("checkDestination error: points is empty")
        return False

    # 最後のポイントが目的地と一致するか確認
    if points[-1][0] == pos['lon'] and points[-1][1] == pos['lat']:
        return True
    else:
        return False


if __name__ == "__main__":
    # スクリプトの絶対パスを取得
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # 読み込みたいファイルの相対パスを指定
    file_path = os.path.join(script_dir, CSVFILE)
    points = load_csv(file_path)
    pos = {'lon': points[0][0], 'lat': points[0][1]}
    output = genMapboxJson(pos)
    print(json.dumps(output, ensure_ascii=False, indent=2))

    # 目的地チェック
    pos = {'lon': points[-1][0], 'lat': points[-1][1]}
    if checkDestination(pos):
        print("OK")
    else:
        print("NG")
