"""03. 실시간 위험구역 감시 — 2회차 02 실습의 규칙을 웹캠에서 그대로 실행

순서:
  1) 첫 화면에서 마우스로 금지 구역을 그립니다.
     왼쪽 클릭 = 꼭짓점 추가, 오른쪽 클릭 = 마지막 점 취소, Enter = 완료 (3점 이상)
     그린 구역은 zone.json에 저장되어 다음 실행 때 그대로 씁니다 (--redraw로 다시 그리기)
  2) 사람을 추적하며 발 위치가 구역 안에 들어가면 진입, 체류가 길면 경고합니다.

실행 예:
  python 03_live_zone.py                       (웹캠 0번)
  python 03_live_zone.py --source vtest.avi    (영상 파일)
  python 03_live_zone.py --source vtest.avi --headless --zone "0.3,0.5;0.7,0.5;0.7,0.95;0.3,0.95"
     --zone: 꼭짓점을 화면 비율(0~1)로 '가로,세로;가로,세로;…' 형식으로 지정
키:  q 종료
결과: zone_events.csv (진입·이탈 기록, 열 이름 영문), --headless일 때 live_zone.mp4
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolo26n.pt')
parser.add_argument('--source', default='0', help='카메라 번호 또는 영상 파일 경로')
parser.add_argument('--conf', type=float, default=0.4)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--zone', default=None, help='구역 꼭짓점(화면 비율) "x,y;x,y;…"')
parser.add_argument('--redraw', action='store_true', help='zone.json을 무시하고 구역을 다시 그리기')
parser.add_argument('--headless', action='store_true', help='창 없이 live_zone.mp4로 저장')
parser.add_argument('--max-frames', type=int, default=0)
args = parser.parse_args()

# 판정 규칙 (2회차 02 실습과 같은 의미)
ENTER_FRAMES = 3     # 연속 몇 프레임 구역 안이면 '진입'으로 확정
EXIT_FRAMES = 5      # 연속 몇 프레임 구역 밖이면 '이탈'로 확정
DWELL_WARN_S = 3.0   # 이 시간(초) 넘게 머물면 경고

source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)
ok, first = cap.read()
if not ok:
    raise SystemExit(f'영상을 열 수 없습니다: {args.source}')
H, W = first.shape[:2]

# ---- 1) 금지 구역 정하기: --zone 인자 > zone.json > 마우스로 그리기 ----
zone_file = Path('zone.json')
if args.zone:
    zone = [[float(x) * W, float(y) * H] for x, y in (p.split(',') for p in args.zone.split(';'))]
elif zone_file.exists() and not args.redraw:
    zone = json.loads(zone_file.read_text())['points']
    print('zone.json의 구역을 씁니다. 다시 그리려면 --redraw')
elif args.headless:
    raise SystemExit('--headless에서는 --zone으로 구역을 지정하세요.')
else:
    pts = []
    def on_mouse(event, x, y, flags, param):  # 마우스 콜백: 클릭할 때마다 호출됨
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
    win = 'Draw zone: L-click add, R-click undo, Enter done'
    cv2.namedWindow(win); cv2.setMouseCallback(win, on_mouse)
    while True:
        view = first.copy()
        if pts:
            cv2.polylines(view, [np.array(pts, np.int32)], len(pts) > 2, (0, 0, 255), 2)
            for p in pts:
                cv2.circle(view, tuple(p), 4, (0, 0, 255), -1)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(pts) >= 3:  # Enter
            break
        if key == ord('q'):
            raise SystemExit('구역 그리기를 취소했습니다.')
    cv2.destroyWindow(win)
    zone = pts
    zone_file.write_text(json.dumps({'width': W, 'height': H, 'points': zone}))
    print('zone.json 저장:', zone)
ZONE = np.array(zone, np.int32)

# ---- 2) 추적 + 규칙 ----
model = YOLO(args.model)
fps_src = cap.get(cv2.CAP_PROP_FPS) or 0  # 영상 파일의 초당 프레임 수 (카메라는 0일 수 있음)
use_video_clock = isinstance(source, str) and fps_src > 0  # 영상 파일이면 프레임 번호로 시간 계산
state = {}   # track_id → {'in': 구역 안 확정 여부, 'cnt': 연속 프레임 수, 'enter_t': 진입 시각}
events = []  # 확정된 진입·이탈 기록
writer, n, t_start = None, 0, time.time()
frame = first
while True:
    now = n / fps_src if use_video_clock else time.time() - t_start  # 현재 시각(초)
    # persist=True: 이전 프레임의 추적 정보를 이어서 씀 → 같은 사람에게 같은 번호(track_id)
    r = model.track(frame, persist=True, classes=[0], conf=args.conf, imgsz=args.imgsz,
                    tracker='bytetrack.yaml', verbose=False)[0]
    out = frame.copy()
    cv2.polylines(out, [ZONE], True, (0, 0, 255), 2)
    alarm = False
    if r.boxes.id is not None:  # 추적 번호가 붙은 사람이 있을 때만
        for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
            foot = (float((x1 + x2) / 2), float(y2))  # 발 위치 = 박스 아래 가운데
            inside = cv2.pointPolygonTest(ZONE, foot, False) >= 0  # 구역 안(+1)·경계(0)면 True
            s = state.setdefault(tid, {'in': False, 'cnt': 0, 'enter_t': None})
            # 연속 프레임 조건: 지금 상태와 다른 판정이 연속으로 쌓여야 상태를 바꿈 (깜빡임 방지)
            s['cnt'] = s['cnt'] + 1 if inside != s['in'] else 0
            if not s['in'] and s['cnt'] >= ENTER_FRAMES:
                s.update({'in': True, 'cnt': 0, 'enter_t': now})
            elif s['in'] and s['cnt'] >= EXIT_FRAMES:
                dwell = now - s['enter_t']
                events.append({'event_id': len(events) + 1, 'track_id': tid, 'enter_s': round(s['enter_t'], 2),
                               'exit_s': round(now, 2), 'dwell_s': round(dwell, 2), 'warning': dwell >= DWELL_WARN_S})
                s.update({'in': False, 'cnt': 0, 'enter_t': None})
            dwell_now = now - s['enter_t'] if s['in'] else 0
            warn = s['in'] and dwell_now >= DWELL_WARN_S
            alarm |= warn
            color = (0, 0, 255) if warn else (0, 165, 255) if s['in'] else (0, 200, 0)  # 경고 빨강, 구역 안 주황, 밖 초록
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f'#{tid}' + (f' {dwell_now:.1f}s' if s['in'] else '')
            cv2.putText(out, label, (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(out, (int(foot[0]), int(foot[1])), 4, color, -1)
    in_zone = sum(s['in'] for s in state.values())
    banner = f'IN ZONE {in_zone} | events {len(events)}' + (' | WARNING' if alarm else '')
    cv2.rectangle(out, (0, 0), (W, 36), (0, 0, 180) if alarm else (40, 40, 40), -1)
    cv2.putText(out, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    n += 1
    if args.headless:
        if writer is None:
            writer = cv2.VideoWriter('live_zone.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_src or 15, (W, H))
        writer.write(out)
    else:
        cv2.imshow('Zone monitor (q: quit)', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if args.max_frames and n >= args.max_frames:
        break
    ok, frame = cap.read()
    if not ok:
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
with open('zone_events.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['event_id', 'track_id', 'enter_s', 'exit_s', 'dwell_s', 'warning'])
    w.writeheader(); w.writerows(events)
print(f'{n}프레임 처리 | 확정 이벤트 {len(events)}건 (경고 {sum(e["warning"] for e in events)}건) → zone_events.csv')
