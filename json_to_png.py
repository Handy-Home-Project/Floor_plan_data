'''

code to compare model inference result between data labels



'''
import cv2
import numpy as np
import json
import os
import random

def draw_polygon(dir, file):
    json_file_path  = f'{dir}{file[:-4]}'
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # 빈 캔버스 생성 (예제 해상도)
    canvas = np.zeros((3000, 4000, 3), dtype=np.uint8)

    # 색상 매핑 (픽셀 값별 랜덤 색 지정, 색 차이 강조)
    color_map = {}
    legend_entries = []
    start_x, start_y = 100, 100  # 범례 시작 위치

    for region, points in data.items():
        pixel_value = int(region.split("_")[1])  # region_x_y 에서 x 추출
        if pixel_value not in color_map:
            color_map[pixel_value] = (
                (pixel_value * 3) % 256, (pixel_value * 5) % 256, (pixel_value * 7) % 256
            )

        # 다각형 내부 색 채우기
        polygon = np.array(points, dtype=np.int32)
        cv2.fillPoly(canvas, [polygon], color_map[pixel_value])
        cv2.polylines(canvas, [polygon], isClosed=True, color=(0, 0, 0), thickness=2)

        # 범례(legend) 추가 (각 region 이름 표시)
        legend_entries.append((color_map[pixel_value], f"Region {pixel_value}"))



    # 범례(legend) 크게 표시
    for idx, (color, text) in enumerate(legend_entries):
        cv2.rectangle(canvas, (start_x, start_y + idx * 50), (start_x + 40, start_y + idx * 50 + 40), color, -1)
        cv2.putText(canvas, text, (start_x + 55, start_y + idx * 50 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 최종 이미지 저장
    cv2.imwrite(f"test_result/{file}", canvas)
    print(f"이미지가 생성되었습니다: {file}")




def draw_from_label(dir, file):
    json_file_path  = f'{dir}{file[:-4]}.json'
    with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

    annotations = data['annotations']

    # 빈 캔버스 생성 (예제 해상도)
    canvas = np.zeros((3000, 4000, 3), dtype=np.uint8)

    category_colors = {}
    legend_entries = []
    legend_pos = (100, 100)
    legend_gap = 60
    font_scale = 1.8

    for ann in annotations:
        cat_id = ann['category_id']
        seg_list = ann['segmentation']

        # category_id 마다 색 고정
        if cat_id not in category_colors:
            category_colors[cat_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            legend_entries.append((cat_id, category_colors[cat_id]))
            # 범례 그리기 전에 정렬
            legend_entries = sorted(legend_entries, key=lambda x: x[0])

        for seg in seg_list:
            pts = np.array([[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], category_colors[cat_id])
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 0), thickness=3)

    # 범례 추가
    for i, (cat_id, color) in enumerate(legend_entries):
        y = legend_pos[1] + i * legend_gap
        cv2.rectangle(canvas, (legend_pos[0], y), (legend_pos[0] + 30, y + 30), color, -1)
        cv2.putText(canvas, f'Category {cat_id}', (legend_pos[0] + 40, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 이미지 저장
    cv2.imwrite(f'origin_label/{file}', canvas)
    #cv2.imwrite(f'origin_test', canvas)











dir = '/c/Users/USER/handy_home/dataset/type/select/'
json_dir = '/c/Users/USER/handy_home/model_custom/floor_plan_1/test/'
origin_dir = '/c/Users/USER/handy_home/model_custom/floor_plan_1/test_origin/'
label_dir = '/c/Users/USER/handy_home/dataset/type/labels/'


for file in os.listdir(dir)[40:50]:
    cmd = f'cp {dir}{file} {origin_dir}'
    os.system(cmd)
    cmd = f'python spa_prediction_3.py -rt test -dt {dir}{file}'
    os.system(cmd)
    
    draw_polygon(json_dir, file)
    
    draw_from_label(label_dir, file)