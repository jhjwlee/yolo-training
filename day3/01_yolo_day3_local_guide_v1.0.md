# 3회차 로컬 실습 가이드 — 내 PC에서 YOLO 실시간으로 돌리기

지금까지는 Colab에서 학습하고 결과를 그림·영상 파일로 확인했습니다. 실제 현장에서는 학습된 모델을 **카메라 옆 컴퓨터(공장 PC, 매장 PC, 엣지 장치)** 에서 실시간으로 돌립니다.
이 실습에서는 내 노트북을 그 현장 컴퓨터로 보고 다음을 해 봅니다.

| 순서 | 스크립트 | 하는 일 | 결과 |
|---|---|---|---|
| 0 | — | 파이썬 가상환경 만들고 패키지 설치 | `yolo-env` 폴더 |
| 1 | `01_check_env.py` | 버전·장치·카메라 점검, YOLO 첫 실행 | `check_result.jpg` |
| 2 | `02_live_detect.py` | 웹캠 실시간 탐지와 FPS 확인 | 화면 창, `snapshot_N.jpg` |
| 3 | `03_live_zone.py` | 2회차 위험구역 규칙을 웹캠에서 실행 | `zone.json`, `zone_events.csv` |
| 4 | `04_export_benchmark.py` | ONNX·OpenVINO로 내보내 CPU 속도 비교 | `benchmark.csv` |

학습은 계속 Colab(GPU)에서 하고, 내 PC는 **추론(학습된 모델로 예측만 하는 것)** 만 합니다. GPU가 없는 노트북도 괜찮습니다.

> 설치나 카메라가 잘 안 되면 시간을 오래 쓰지 말고 [Colab 대체 노트북 02](02_yolo_day3_edge_colab_v1.0.ipynb)로 같은 내용을 진행하세요. 같은 스크립트를 Colab에서 영상 파일로 돌립니다.

---

## 0. 준비 — 파이썬과 가상환경

### 0-1. 파이썬 확인

터미널(Windows는 **명령 프롬프트** 또는 **PowerShell**, Mac은 **터미널** 앱)을 열고 버전을 확인합니다.

```bash
python --version      # Windows
python3 --version     # Mac
```

**3.10, 3.11, 3.12** 중 하나면 됩니다. 없거나 3.9 이하라면 [python.org](https://www.python.org/downloads/)에서 3.12를 설치합니다.
Windows 설치 화면에서는 맨 아래 **"Add python.exe to PATH"** 를 반드시 체크하세요(체크하지 않으면 터미널에서 `python`을 찾지 못합니다).

### 0-2. 실습 파일 받기

GitHub 리포의 `day3/local` 폴더를 내려받습니다. 둘 중 편한 방법을 쓰세요.

- **zip으로**: 리포 첫 화면의 초록색 **Code → Download ZIP** → 압축을 풀고 `day3/local` 폴더로 이동
- **git으로**: `git clone https://github.com/jhjwlee/yolo-training.git` 후 `cd yolo-training/day3/local`

### 0-3. 가상환경 만들고 패키지 설치

**가상환경**(virtual environment)은 이 실습용 패키지만 따로 담는 폴더입니다. 다른 프로젝트의 패키지와 버전이 섞이지 않습니다.

Windows:
```bat
cd 경로\yolo-training\day3\local
python -m venv yolo-env
yolo-env\Scripts\activate
pip install -r requirements.txt
```

Mac:
```bash
cd 경로/yolo-training/day3/local
python3 -m venv yolo-env
source yolo-env/bin/activate
pip install -r requirements.txt
```

- 활성화되면 프롬프트 앞에 `(yolo-env)`가 붙습니다. 터미널을 새로 열 때마다 `activate` 줄을 다시 실행합니다.
- 설치는 PyTorch를 포함해 수백 MB를 받으므로 5~10분 걸릴 수 있습니다.
- PowerShell에서 `activate`가 "스크립트를 실행할 수 없습니다"로 막히면 한 번만 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 실행하거나 명령 프롬프트(cmd)를 쓰세요.

---

## 1. 환경 점검 — `01_check_env.py`

```bash
python 01_check_env.py
```

5단계를 차례로 검사합니다. 끝까지 통과하면 `check_result.jpg`에 웹캠 사진과 탐지 결과가 저장됩니다.

| 단계 | 확인하는 것 | 실패하면 |
|---|---|---|
| [1] | 파이썬 버전 | 0-1로 돌아가 3.10~3.12 설치 |
| [2] | ultralytics·torch·opencv 설치 | 가상환경 활성화 후 `pip install -r requirements.txt` |
| [3] | 가속 장치 (cuda / mps / cpu) | cpu여도 됩니다 |
| [4] | 카메라 열기 | 아래 "막혔을 때" 참고. 예시 사진으로 계속 진행됨 |
| [5] | YOLO 첫 실행 | 인터넷 연결 확인 (모델 파일 자동 다운로드) |

**장치 이름 풀이**: `cuda`는 NVIDIA 그래픽카드, `mps`는 애플 실리콘(M1~M4) 맥의 GPU, `cpu`는 그래픽카드 없이 중앙처리장치만 쓰는 것입니다.

---

## 2. 실시간 탐지 — `02_live_detect.py`

```bash
python 02_live_detect.py                     # 웹캠 0번
python 02_live_detect.py --imgsz 320         # 더 빠르게
python 02_live_detect.py --classes 0         # 사람만
```

창이 뜨면 카메라 앞에서 움직여 보고, 휴대폰·컵·가방 같은 물건도 비춰 보세요. **q** 로 끝내고 **s** 로 현재 화면을 저장합니다.
화면 왼쪽 위의 **FPS**(초당 처리한 프레임 수)를 기록해 두세요. 4단계에서 비교합니다.

| 옵션 | 뜻 |
|---|---|
| `--model` | 쓸 모델 파일. 기본 `yolo26n.pt`(COCO 80종) |
| `--source` | 카메라 번호(0, 1 …) 또는 영상 파일 경로 |
| `--conf` | 신뢰도 기준. 이보다 낮은 예측은 버림 (기본 0.4) |
| `--imgsz` | 모델 입력 크기(한 변 픽셀). 작게 하면 빠르지만 작은 물체를 놓침 |
| `--classes` | 남길 클래스 번호. 0 = 사람, 2 = 승용차, 67 = 휴대폰 |
| `--headless` | 창을 띄우지 않고 `live_detect.mp4`로 저장 |

### Colab에서 학습한 내 모델 쓰기

워크숍에서 학습한 모델은 Drive `MyDrive/yolo_workshop/<프로젝트 이름>/`에 `*_best.pt`로 저장되어 있습니다. 이 파일을 내려받아 `local` 폴더에 넣고 실행합니다.

```bash
python 02_live_detect.py --model baseline_best.pt --conf 0.3
```

카메라로 모니터 화면이나 출력한 사진을 비추면 내 모델이 실시간으로 동작하는 것을 볼 수 있습니다.

---

## 3. 실시간 위험구역 감시 — `03_live_zone.py`

2회차 02 실습(위험구역 침입 감시)에서 만든 규칙을 그대로 웹캠에 붙입니다.

```bash
python 03_live_zone.py
```

1. 첫 화면이 뜨면 마우스로 **금지 구역**을 그립니다. 왼쪽 클릭 = 꼭짓점 추가, 오른쪽 클릭 = 마지막 점 취소, **Enter** = 완료(3점 이상).
2. 구역 안으로 걸어 들어가 보세요. 박스가 **주황색**(구역 안)으로 바뀌고, 3초 넘게 머물면 **빨간색 경고**와 상단 `WARNING`이 뜹니다.
3. **q** 로 끝내면 진입·이탈 기록이 `zone_events.csv`에 저장됩니다.

그린 구역은 `zone.json`에 저장되어 다음 실행 때 그대로 쓰입니다. 다시 그리려면 `--redraw`를 붙입니다.

| 규칙 값 (스크립트 위쪽) | 뜻 | 2회차 실습과 같은 이름 |
|---|---|---|
| `ENTER_FRAMES = 3` | 연속 3프레임 구역 안이면 진입 확정 | ✔ |
| `EXIT_FRAMES = 5` | 연속 5프레임 구역 밖이면 이탈 확정 | ✔ |
| `DWELL_WARN_S = 3.0` | 3초 넘게 머물면 경고 | `DWELL_WARN_S` |

**`zone_events.csv` 열**: `event_id, track_id, enter_s, exit_s, dwell_s, warning`

**해 보기** — 카메라 앞을 빠르게 지나가기만 하면 진입으로 잡히나요? `ENTER_FRAMES`를 1로 바꾸면 어떻게 달라지나요? 영상 파일과 달리 웹캠은 FPS가 일정하지 않아서, 이 스크립트는 웹캠일 때 프레임 번호 대신 **실제 시계 시간**으로 체류 시간을 잽니다.

---

## 4. 내보내기와 CPU 속도 비교 — `04_export_benchmark.py`

```bash
python 04_export_benchmark.py
python 04_export_benchmark.py --imgsz 320
```

같은 모델을 세 가지 형식으로 실행해 CPU 속도를 비교하고 `benchmark.csv`에 저장합니다(열: `format, file, size_mb, imgsz, median_ms, fps`).

| 형식 | 뜻 | 주로 쓰는 곳 |
|---|---|---|
| PyTorch `.pt` | 학습에 쓰는 원래 형식 | 학습·연구 |
| ONNX `.onnx` | 여러 실행 엔진이 함께 읽는 표준 모델 형식 | 서버, 다양한 언어(C++, C#, Java) |
| OpenVINO | 인텔 CPU에 맞춰 최적화하는 형식 | 인텔 CPU 공장 PC·엣지 장치 |

- OpenVINO는 `requirements.txt`에서 주석 처리되어 있습니다. 비교하려면 `pip install openvino` 후 다시 실행하세요. 설치되지 않았으면 그 형식은 건너뜁니다.
- 내보낸 모델은 2·3번 스크립트에 그대로 넣을 수 있습니다: `python 02_live_detect.py --model yolo26n.onnx`

**해 보기** — 결과 표로 "이 노트북에서 실시간(15 FPS 이상)이 되려면 어떤 형식·입력 크기를 써야 하는가"를 정해 보세요. 현장 장비를 고를 때 하는 계산과 같습니다.

---

## 막혔을 때

| 증상 | 원인과 해결 |
|---|---|
| `python`을 찾을 수 없음 (Windows) | 설치 때 PATH 체크를 안 함 → 파이썬 재설치 시 "Add python.exe to PATH" 체크, 또는 `py` 명령으로 실행 |
| `pip install` 중 시간 초과·SSL 오류 | 학원·회사 네트워크 보안 정책일 수 있음 → 휴대폰 핫스팟으로 설치하거나 Colab 대체 노트북 사용 |
| torch 설치 후 `DLL load failed` (Windows) | Microsoft Visual C++ 재배포 패키지가 없음 → "Visual C++ Redistributable x64" 설치 후 재시도 |
| [4] 카메라가 안 열림 (Mac) | 시스템 설정 → 개인정보 보호 및 보안 → **카메라**에서 터미널(또는 VS Code) 허용 후 터미널 재시작 |
| [4] 카메라가 안 열림 (Windows) | 설정 → 개인 정보 → 카메라 → "데스크톱 앱이 카메라에 액세스하도록 허용" 켜기. Zoom·Teams 등 카메라를 쓰는 앱 종료 |
| 외장 웹캠을 쓰고 싶음 | `--source 1` (또는 2)로 번호를 바꿔 보기 |
| 창이 안 뜨고 `cv2.imshow` 오류 | `opencv-python-headless`가 설치된 경우 → `pip uninstall -y opencv-python-headless` 후 `pip install opencv-python` |
| FPS가 2~3밖에 안 나옴 | `--imgsz 320`, `--classes 0`으로 줄이기, 4단계에서 ONNX·OpenVINO 형식 써 보기 |
| `lap` 설치 실패 | 추적 기능에 필요 → `pip install lap` 다시 시도. 안 되면 2번 스크립트(추적 없음)까지만 진행 |

## 용어 정리

| 용어 | 뜻 |
|---|---|
| 추론 (inference) | 학습된 모델로 새 입력에 대해 예측만 하는 것입니다. 학습보다 계산이 훨씬 적습니다. |
| 엣지 장치 | 카메라·센서 옆에서 직접 계산하는 컴퓨터입니다. 클라우드로 영상을 보내지 않아 빠르고 개인정보 부담이 적습니다. |
| 가상환경 (venv) | 프로젝트마다 패키지를 따로 담는 파이썬 폴더입니다. `activate`로 켭니다. |
| requirements.txt | 설치할 패키지 목록 파일입니다. `pip install -r`로 한 번에 설치합니다. |
| FPS | 초당 처리한 프레임 수입니다. 실시간 감시는 보통 10~15 FPS 이상이면 충분합니다. |
| imgsz | 모델에 넣는 이미지 한 변 크기(픽셀)입니다. 640 → 320으로 줄이면 계산량이 약 4분의 1이 됩니다. |
| CUDA / MPS / CPU | NVIDIA GPU / 애플 실리콘 GPU / 중앙처리장치. ultralytics가 있는 장치를 자동으로 고릅니다. |
| 내보내기 (export) | 학습된 모델을 다른 실행 엔진용 형식으로 바꾸는 것입니다. |
| ONNX / ONNX Runtime | 표준 모델 형식 / 그 형식을 실행하는 엔진입니다. |
| OpenVINO | 인텔이 만든 추론 최적화 도구입니다. 인텔 CPU에서 빠른 경우가 많습니다. |
| 워밍업 | 처음 몇 번의 예측은 준비 작업 때문에 느려서 속도 측정에서 빼는 것입니다. |
| 중앙값 (median) | 값을 크기순으로 줄 세웠을 때 가운데 값입니다. 가끔 튀는 측정값의 영향을 줄입니다. |
| 마우스 콜백 | 마우스를 클릭할 때마다 자동으로 불리도록 등록한 함수입니다. 구역 그리기에 씁니다. |
| track_id | 추적 기능이 같은 사람에게 계속 붙이는 번호입니다. |
