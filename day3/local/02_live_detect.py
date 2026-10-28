"""02. 실시간 탐지 — 웹캠(또는 영상 파일)에 YOLO를 돌리고 FPS를 화면에 표시

실행 예:
  python 02_live_detect.py                          (웹캠 0번, 기본 모델)
  python 02_live_detect.py --model my_best.pt       (Colab에서 학습해 내려받은 모델)
  python 02_live_detect.py --source sample.mp4      (영상 파일)
  python 02_live_detect.py --imgsz 320              (느리면 입력 크기를 줄이기)
  python 02_live_detect.py --source sample.mp4 --headless   (창 없이 결과 영상만 저장)
키:  q 종료,  s 현재 화면 저장(snapshot_N.jpg)
결과: --headless일 때 live_detect.mp4
"""
import argparse
import time

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolo26n.pt', help='모델 파일 (.pt / .onnx / _openvino_model 폴더)')
parser.add_argument('--source', default='0', help='카메라 번호(0, 1 …) 또는 영상 파일 경로')
parser.add_argument('--conf', type=float, default=0.4, help='신뢰도 기준 (이보다 낮은 예측은 버림)')
parser.add_argument('--imgsz', type=int, default=640, help='모델 입력 한 변 크기 (작을수록 빠르고 덜 정확)')
parser.add_argument('--classes', type=int, nargs='*', default=None, help='남길 클래스 번호 (예: 0 = 사람)')
parser.add_argument('--headless', action='store_true', help='창을 띄우지 않고 live_detect.mp4로 저장')
parser.add_argument('--max-frames', type=int, default=0, help='이 프레임 수만큼만 처리 (0 = 끝까지)')
args = parser.parse_args()

model = YOLO(args.model)
source = int(args.source) if args.source.isdigit() else args.source  # 숫자면 카메라 번호
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise SystemExit(f'영상을 열 수 없습니다: {args.source}  (카메라 권한·번호를 확인하세요)')

writer, fps_ema, n, shots = None, None, 0, 0  # fps_ema: 화면 FPS를 부드럽게 보여 주는 이동 평균
while True:
    ok, frame = cap.read()
    if not ok:  # 영상 끝 또는 카메라 끊김
        break
    t0 = time.perf_counter()
    r = model(frame, conf=args.conf, imgsz=args.imgsz, classes=args.classes, verbose=False)[0]
    dt = time.perf_counter() - t0  # 한 프레임 처리 시간(초)
    fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt  # 지수 이동 평균

    out = r.plot()  # 박스를 그린 그림
    cv2.putText(out, f'{fps_ema:.1f} FPS | {len(r.boxes)} obj | imgsz {args.imgsz}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    n += 1
    if args.headless:
        if writer is None:  # 첫 프레임에서 저장 파일을 엶 (mp4v 코덱)
            h, w = out.shape[:2]
            writer = cv2.VideoWriter('live_detect.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
        writer.write(out)
    else:
        cv2.imshow('YOLO live (q: quit, s: snapshot)', out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            shots += 1
            cv2.imwrite(f'snapshot_{shots}.jpg', out)
            print('저장:', f'snapshot_{shots}.jpg')
    if args.max_frames and n >= args.max_frames:
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print(f'{n}프레임 처리, 평균 약 {fps_ema:.1f} FPS' if fps_ema else '처리한 프레임이 없습니다.')
