"""01. 로컬 환경 점검 — 패키지 버전, 가속 장치, 카메라, YOLO 첫 실행

실행:  python 01_check_env.py            (카메라 0번 확인)
       python 01_check_env.py --camera 1 (외장 카메라 등 다른 번호)
       python 01_check_env.py --no-camera (카메라 없이 점검)
결과:  check_result.jpg (카메라 사진 또는 예시 사진에 YOLO 결과를 그린 그림)
"""
import argparse
import platform
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, default=0, help='카메라 번호 (노트북 내장 카메라는 보통 0)')
parser.add_argument('--no-camera', action='store_true', help='카메라 점검 건너뛰기')
args = parser.parse_args()

print('[1] 파이썬:', sys.version.split()[0], '|', platform.system(), platform.machine())
if sys.version_info < (3, 9):
    print('    ! 파이썬 3.9 이상이 필요합니다. 3.11 또는 3.12를 권장합니다.')

# [2] 패키지 버전 — 하나라도 없으면 설치 안내 후 종료
try:
    import cv2
    import torch
    import ultralytics
except ImportError as e:
    print('    ! 패키지가 없습니다:', e.name, '→ pip install -r requirements.txt')
    sys.exit(1)
print('[2] ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| opencv', cv2.__version__)

# [3] 가속 장치: CUDA(엔비디아 GPU), MPS(애플 실리콘 GPU), 없으면 CPU
if torch.cuda.is_available():
    device = 'cuda'
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print('[3] 사용할 장치:', device, '(cpu여도 추론 실습은 충분히 됩니다)')

# [4] 카메라 사진 한 장 찍기 (실패하면 예시 사진으로 대신)
frame = None
if args.no_camera:
    print('[4] 카메라 점검 건너뜀 (--no-camera) → 예시 사진으로 계속합니다.')
else:
    cap = cv2.VideoCapture(args.camera)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    if ok:
        print(f'[4] 카메라 {args.camera}번: 열림, 크기 {frame.shape[1]}x{frame.shape[0]}')
    else:
        frame = None
        print(f'[4] 카메라 {args.camera}번: 열리지 않음 → 가이드의 "카메라가 안 열릴 때"를 보세요. 예시 사진으로 계속합니다.')

# [5] YOLO 첫 실행 — 모델 파일(yolo26n.pt)은 처음 한 번 자동 다운로드
from ultralytics import YOLO
from ultralytics.utils import ASSETS  # ultralytics 패키지 안에 들어 있는 예시 사진 폴더

model = YOLO('yolo26n.pt')
src = frame if frame is not None else str(ASSETS / 'bus.jpg')
r = model(src, device=device, verbose=False)[0]
names = [model.names[int(c)] for c in r.boxes.cls]
print(f'[5] YOLO 실행 성공: {len(names)}개 탐지 {sorted(set(names))} | 추론 {r.speed["inference"]:.0f} ms')
cv2.imwrite('check_result.jpg', r.plot())
print('    → check_result.jpg 저장. 모두 통과했으면 02_live_detect.py로 넘어가세요.')
