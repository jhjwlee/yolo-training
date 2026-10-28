"""04. 모델 내보내기와 CPU 속도 비교 — PyTorch(.pt) vs ONNX vs OpenVINO

현장 장비(엣지 장치, 공장 PC)에서는 GPU 없이 CPU로 돌리는 경우가 많습니다.
같은 모델을 실행 엔진에 맞는 형식으로 바꾸면(내보내기, export) 더 빨라질 수 있습니다.

실행 예:
  python 04_export_benchmark.py                        (yolo26n.pt, 입력 640)
  python 04_export_benchmark.py --model my_best.pt --imgsz 320
결과: benchmark.csv (열: format, file, size_mb, imgsz, median_ms, fps)
      내보낸 모델: *.onnx, *_openvino_model/ 폴더 → 02·03 스크립트의 --model에 그대로 넣어 쓸 수 있음
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ASSETS

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolo26n.pt')
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--runs', type=int, default=30, help='속도 측정 반복 횟수')
args = parser.parse_args()

img = str(ASSETS / 'bus.jpg')  # 측정용 예시 사진 (ultralytics에 포함)
base = YOLO(args.model)

# 비교할 형식: (이름, 모델 경로를 만드는 방법)
targets = [('pytorch', args.model)]
for fmt in ['onnx', 'openvino']:
    try:
        # export: 지정 형식으로 변환한 파일(또는 폴더) 경로를 돌려줌. 실패하면(패키지 없음 등) 건너뜀
        path = base.export(format=fmt, imgsz=args.imgsz, verbose=False)
        targets.append((fmt, path))
    except Exception as e:
        print(f'[{fmt}] 내보내기 건너뜀: {type(e).__name__} — {str(e)[:80]}')

# 폴더(OpenVINO)는 안의 파일 크기를 모두 더함
def size_mb(p):
    p = Path(p)
    total = sum(f.stat().st_size for f in p.rglob('*')) if p.is_dir() else p.stat().st_size
    return round(total / 1e6, 1)

rows = []
for name, path in targets:
    m = YOLO(path, task='detect')
    for _ in range(3):  # 워밍업: 처음 몇 번은 준비 작업 때문에 느려서 측정에서 뺌
        m(img, imgsz=args.imgsz, device='cpu', verbose=False)
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        m(img, imgsz=args.imgsz, device='cpu', verbose=False)
        times.append(time.perf_counter() - t0)
    ms = float(np.median(times) * 1000)  # 중앙값: 가끔 튀는 값의 영향을 줄임
    rows.append({'format': name, 'file': str(path), 'size_mb': size_mb(path), 'imgsz': args.imgsz,
                 'median_ms': round(ms, 1), 'fps': round(1000 / ms, 1)})
    print(f'{name:<9} {rows[-1]["size_mb"]:>6} MB  {ms:7.1f} ms  {1000 / ms:6.1f} FPS')

with open('benchmark.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('→ benchmark.csv 저장')
