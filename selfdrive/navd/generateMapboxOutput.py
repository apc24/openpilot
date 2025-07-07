import os
import csv
import json
import math
SPD = 30 #[km/h] vehicle speed
CSVFILE = 'selfdrive/navd/route.txt'

def load_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        points = []
        for row in reader:
            lon = float(row['lon'])
            lat = float(row['lat'])
            type_ = row['type']
            modifier = row['modifier']
            points.append((lon, lat, type_, modifier))
        return points
def generate_mapbox_output(points, start_index=0):
    routes = {
        "routes": [{
            "legs": [{
                "annotation": {
                    "maxspeed": [{"unknown": True}] * len(points[start_index:])
                },
                "steps": []
            }]
        }]
    }

    coordinates =[]

    for i in range(start_index, len(points) ):
        current = points[i]
        coordinates.append([current[0],current[1]])

        if current[3] != "st":
#            print("coordinates")
#            print(coordinates)
            distance = calculate_distance(coordinates)
            spd = SPD
            duration = distance / (spd * 1000 / 3600)
            type_ = current[2]
            modifier = current[3]
            if type_ == "arrive":
                text = f"Your destination is one the {modifier}"
            else:
                text = f"Turn {type_}"
            step = {
                "bannerInstructions": [{
                    "primary": {
                        "type": type_,
                        "modifier": modifier,  # modifier
                        "text": text
                    },
                    "distanceAlongGeometry": distance
                }],
                "speedLimitSign": "vienna",
                "duration_typical": duration,  # placeholder
                "duration": duration,  # placeholder
                "distance": distance,
                "geometry": { "coordinates": coordinates}
            }
            routes["routes"][0]["legs"][0]["steps"].append(step)
            coordinates = [[current[0],current[1]]]
#    return json.dumps(routes, ensure_ascii=False, indent=2)
    return routes

#        next_point = points[i + 1]
'''
    # Add the last step for arrival
    arrival_step = {
        "bannerInstructions": [{
            "primary": {
                "type": "arrive",
                "modifier": "right",
                "text": "Your destination will be on the right"
            },
            "distanceAlongGeometry": 0  # placeholder
        }],
        "speedLimitSign": "vienna",
        "duration_typical": 0,  # placeholder
        "duration": 0,  # placeholder
        "distance": 0,  # placeholder
        "geometry": {
            "coordinates": [
                [points[-1][0], points[-1][1]],
                [points[-1][0], points[-1][1]]
            ]
        }
    }

    routes["routes"][0]["legs"][0]["steps"].append(arrival_step)
'''

EARTH_RAD = 6378137
def cal_distance(lon1, lat1, lon2, lat2):
   th = math.sin(math.radians(lat1))*math.sin(math.radians(lat2)) + \
        math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.cos(math.radians(lon2-lon1))
   th = min(1.0,max(-1.0,th))

   return EARTH_RAD * math.acos(th)

# reference: https://qiita.com/Yuzu2yan/items/0f312954feeb3c83c70e
#    return json.dumps(routes, ensure_ascii=False, indent=2)
def calculate_distance(coordinates):
    distance = 0
    for i in range(len(coordinates)-1 ):
        distance += cal_distance(coordinates[i][0],coordinates[i][1],coordinates[i+1][0],coordinates[i+1][1])

    return distance  # Replace with actual distance calculation logic

import numpy as np
from openpilot.selfdrive.navd.nearestLink import nearestLink

def genMapboxJson(pos):
    points = load_csv(CSVFILE)
#    print("points:")
#    print(points)
    mat = np.zeros([len(points),2])
    for idx in range(len(points)):
        mat[idx,:]=np.array([points[idx][0],points[idx][1]])

    p0 = np.array([[pos['lon'], pos['lat']]])
#    print(f"p0:{p0}")
#    print("mat")
#    print(mat)

    [p,idx]=nearestLink(mat,p0)
#    print(f"p:{p}")
#    print(f"idx:{idx}")

    # 自車位置以降の出力を生成
    points = points[(idx+1):]
    point0 = (float(p[0]),float(p[1]),'st','st')
    points.insert(0,point0)
#    print("new points")
#    print(points)
    output = generate_mapbox_output(points)
    return output

if __name__ == "__main__":
    if os.getenv('LOADCSVMAP') == 'TRUE':
        csv_file_path = CSVFILE  # CSVファイルのパスを指定してください
        points = load_csv(csv_file_path)
#        mat = np.zeros([len(points),2])
#        for idx in range(len(points)):
#            mat[idx,:]=np.array([points[idx][0],points[idx][1]])
#        pos = {'lon': mat[3,0], 'lat': mat[3,1]}
        pos = {'lon': points[0][0], 'lat': points[0][1]}
#        print("自車位置")
#        print(pos)
        output = genMapboxJson(pos)

#        print(mat)
#        print("自車位置以降のリストに変更")
#        p0 = mat[3,:]+0.00001
#        [p,idx]=nearestLink(mat,p0)
        # 自車位置以降の出力を生成
#        print("idx=%d",idx)
#        points = points[(idx+1):]
#        point0 = (p[0],p[1],'st','st')
#        points.insert(0,point0)
#        output = generate_mapbox_output(points)
        print(json.dumps(output, ensure_ascii=False, indent=2))

