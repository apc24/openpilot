from metadrive.component.map.pg_map import MapGenerateMethod
#from metadrive.utils.space import S, C

def create_map():
    return dict(
        type=MapGenerateMethod.PG_MAP_FILE,
        lane_num=1,
        lane_width=4,
        config=
        [
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 5
              },
              {
                "id": "O",
                "pre_block_socket_index": 0,
              #  "radius": 30,
	      #  "angle": 110
              },
              #{
              #  "id": "C",
              #  "pre_block_socket_index": 0,
              #  "length": 0.01,
              #  "radius": 5,
              #  "angle": 85,
              #  "dir": 1
              #},
              #{
              #  "id": "C",
              #  "pre_block_socket_index": 0,
              #  "length": 0.01,
              #  "radius": 30,
              #  "angle": 110,
              #  "dir": 0
              #},
              #{
              #  "id": "C",
              #  "pre_block_socket_index": 0,
              #  "length": 0.01,
              #  "radius": 5,
              #  "angle": 85,
              #  "dir": 1
              #},
              #{
              #  "id": "T",
              #  "pre_block_socket_index": 0,
              #},
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 30
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 25,
                "angle": 70,
                "dir": 1
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 130,
                "angle": 55,
                "dir": 1
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 95,
                "angle": 90,
                "dir": 0
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 130,
                "angle": 60,
                "dir": 1
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 5
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 63.375,
                "angle": 40,
                "dir": 0
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 172.53125,
                "angle": 9.012838881997936,
                "dir": 0
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 67.2734375,
                "angle": 25.326170297138617,
                "dir": 0
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 188.125,
                "angle": 9.49016490572501,
                "dir": 0
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 82.8671875,
                "angle": 17.202457582582017,
                "dir": 1
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 63.375,
                "angle": 22.700302377302712,
                "dir": 1
              },
              #{
              #  "id": "S",
              #  "pre_block_socket_index": 0,
              #  "length": 5
              #},
              #{
              #  "id": "S",
              #  "pre_block_socket_index": 0,
              #  "length": 16
              #}
              #{
              #  "id": "S",
              #  "pre_block_socket_index": 0,
              #  "length": 8
              #},
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 30,
                "angle": 7.926414,
                "dir": 1
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 60,
                "angle": 40,
                "dir": 1
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 100,
                "angle": 40,
                "dir": 1
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 78.7
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 60,
                "angle": 55,
                "dir": 0
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 60,
                "angle": 40,
                "dir": 1
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 60
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 80,
                "angle": 75,
                "dir": 1
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 90,
                "angle": 90,
                "dir": 1
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 25
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 100,
                "angle": 60,
                "dir": 0
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 50
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 40,
                "angle": 30,
                "dir": 1
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 20
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 70,
                "angle": 60,
                "dir": 1
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 25
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 80,
                "angle": 60,
                "dir": 1
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 80,
                "angle": 60,
                "dir": 1
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 160,
                "angle": 40,
                "dir": 1
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 90
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 50,
                "angle": 110,
                "dir": 0
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 65,
                "angle": 120,
                "dir": 0
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 10
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 200,
                "angle": 40,
                "dir": 1
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 60,
                "angle": 44,
                "dir": 0
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 60,
                "angle": 30,
                "dir": 0
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 80
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 40,
                "angle": 95,
                "dir": 1
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 8
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 40,
                "angle": 70,
                "dir": 0
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 85
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 25,
                "angle": 25,
                "dir": 1
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 25
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 25,
                "angle": 25,
                "dir": 0
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 38
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 25,
                "angle": 25,
                "dir": 1
              },
              {
                "id": "S",
                "pre_block_socket_index": 0,
                "length": 25
              },
              {
                "id": "C",
                "pre_block_socket_index": 0,
                "length": 0.01,
                "radius": 25,
                "angle": 15,
                "dir": 0
              },
              #{
              #  "id": "S",
              #  "pre_block_socket_index": 0,
              #  "length": 43
              #},
              #{
              #  "id": "C",
              #  "pre_block_socket_index": 0,
              #  "length": 0.01,
              #  "radius": 30,
              #  "angle": 50,
              #  "dir": 1
              #},
            ]
    )












