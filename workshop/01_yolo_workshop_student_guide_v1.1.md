# YOLO 프로젝트 워크숍 안내서

_주제를 골라 개인 또는 2인 1조로, Google Colab에서 데이터 확보부터 결과 정리까지_

앞에서 함께 해 본 두 실습(컨베이어 상자 크기별 집계, 위험구역 침입 감시)의 흐름을 이제 **여러분이 고른 주제와 데이터**에 적용합니다. 모든 작업은 Google Colab에서 하고, 결과는 Google Drive에 저장되어 다음 시간에 이어서 할 수 있습니다.

| 항목 | 내용 |
|---|---|
| 진행 방식 | 개인 또는 2인 1조. 주제 8개 중 하나를 고름 (같은 주제를 여러 팀이 골라도 됨) |
| 기간 | 2회차 후반에 시작 → 3회차에 이어서 완성 |
| 도구 | Google Colab (T4 GPU), Google Drive, 시작 노트북 `02_yolo_workshop_starter_v1.1.ipynb` |
| 멘토링 | 체크포인트(CP1~CP4)마다 멘토가 팀을 돌며 진행 상황을 함께 점검 |
| 최종 결과물 | `project_report.md`, `report.png`, 최종 모델 `*_best.pt`, 실험 기록 `experiments.csv` |

## 1. 진행 흐름과 체크포인트

| 체크포인트 | 언제 | 해야 할 일 | 확인 기준 (멘토와 함께 봄) |
|---|---|---|---|
| CP1 주제·데이터 확정 | 2회차 끝 | 프로젝트 카드 작성 → 데이터 받기 → `data.yaml` 만들기 → 데이터 살펴보기 | 샘플 그림에 정답 박스가 제자리에 그려짐, 클래스 분포를 설명할 수 있음 |
| CP2 베이스라인 | 3회차 초반 | 본 학습(`QUICK_RUN = False`) → 클래스별 지표 | 베이스라인 mAP50과 중요 클래스 재현율·정밀도를 숫자로 말할 수 있음 |
| CP3 개선 | 3회차 중반 | 임계값 정하기 → 오류 분류표 → 개선 실험 1개 이상 | 실험마다 '왜 바꿨는지'가 오류 분류표와 연결됨 |
| CP4 정리 | 3회차 끝 | 데모 → 요약 리포트 → Notes 작성 | 성공 기준 달성 여부와 다음 할 일을 말할 수 있음 |

> **막히면** 데이터 받기나 형식 변환에서 20분 넘게 막히면 혼자 붙잡지 말고 멘토를 부르거나, 주제 표의 **대체 데이터**로 바꿉니다. 워크숍의 목표는 특정 데이터가 아니라 '문제 정의 → 학습 → 평가 → 개선' 흐름을 끝까지 경험하는 것입니다.

## 2. 주제 고르기

주제마다 **1순위 데이터**와 **대체 데이터**를 준비했습니다. 난이도는 데이터 크기, 클래스 수, 물체 크기를 기준으로 매겼습니다. 처음이라면 ★ 주제를, YOLO 경험이 있다면 ★★ 이상을 권합니다.

| 번호 | 분야 | 주제 | 난이도 | 1순위 데이터 |
|---|---|---|---|---|
| T1 | 산업 안전 | 공사 현장 안전장비 미착용 감지 | ★☆☆ 쉬움 | 캐글 Construction Site Safety |
| T2 | 화재·연기 | 실내 화재·연기 조기 감지 | ★☆☆ 쉬움 | 캐글 Indoor Fire & Smoke |
| T3 | 환경 | 재활용 쓰레기 자동 분류 | ★☆☆ 쉬움 | 캐글 Garbage Detection – 6 Waste Categories |
| T4 | 교통·도로 | 도로 파손(포트홀·균열) 탐지와 심각도 | ★★☆ 보통 | 캐글 Road Damage: Potholes, Cracks, Manholes |
| T5 | 물류·유통 | 매장 선반 진열 상품 인식 | ★★☆ 보통 | 캐글 Shelf Detection |
| T6 | 사람 행동 | 낙상 감지 — 넘어진 사람 찾기 | ★★☆ 보통 | 캐글 Fall Detection Dataset |
| T7 | 농업 | 작물 해충 탐지 | ★★☆ 보통 | 캐글 AgroPest-12 |
| T8 | 제조 품질 | PCB 표면 결함 검사 | ★★★ 도전 | 캐글 DsPCBSD+ |

**나만의 주제**도 가능합니다. 캐글이나 공공데이터에서 라벨(박스)이 있는 이미지 데이터를 찾았다면 CP1 전에 멘토와 상의하세요. 라벨 형식이 YOLO·Pascal VOC·COCO 중 하나이고 2GB 이하면 시작 노트북으로 바로 쓸 수 있습니다.

### T1. 공사 현장 안전장비 미착용 감지  (산업 안전 · ★☆☆ 쉬움)

| 항목 | 내용 |
|---|---|
| 풀어 볼 질문 | 현장 사진에서 안전모·조끼를 착용하지 않은 작업자를 찾아 경고할 수 있는가? |
| 1순위 데이터 | 캐글 Construction Site Safety<br>`snehilsanyal/construction-site-safety-image-dataset-roboflow`<br>YOLO txt (train/valid/test 분할 포함), 10클래스: Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person 등 · 약 228MB · CC BY 4.0 |
| 대체 데이터 | Ultralytics construction-ppe<br>`construction-ppe.yaml`<br>YOLO, 1,416장, 11클래스(helmet, no_helmet, vest 등) · 약 178MB · `DATA_SOURCE='ultralytics'` |
| 성공 기준 예시 | `NO-Hardhat` 클래스 재현율 0.7 이상 |
| 더 해 보기 | 사람 박스 안에 안전모가 있는지 규칙으로 판정해 '미착용자 수'를 사진마다 세기 (추가 실습 '말로 찾는 탐지기' Step 5 참고) |
| 주의할 점 | 미착용(NO-*) 클래스는 착용 클래스보다 박스 수가 적습니다. 클래스별 지표를 꼭 따로 보세요. |

### T2. 실내 화재·연기 조기 감지  (화재·연기 · ★☆☆ 쉬움)

| 항목 | 내용 |
|---|---|
| 풀어 볼 질문 | CCTV 화면에서 불꽃과 연기를 얼마나 빨리, 헛경보 없이 찾을 수 있는가? |
| 1순위 데이터 | 캐글 Indoor Fire & Smoke<br>`sinchanashivanand/indoor-fire-and-smoke-detection-with-yolov8`<br>YOLO txt + data.yaml, 2클래스(Fire, Smoke) · 약 253MB · CC0 |
| 대체 데이터 | 캐글 Fire/Smoke Detection YOLO v9<br>`roscoekerby/firesmoke-detection-yolo-v9`<br>YOLO, 35,000장 이상, 2클래스 · 약 965MB · CC BY-SA 4.0 · `FRACTION=0.2` 권장 |
| 성공 기준 예시 | Fire 클래스 정밀도 0.8 이상에서 재현율 최대화 |
| 더 해 보기 | 연속 프레임 조건(위험구역 실습의 `ENTER_FRAMES` 방식)으로 '0.5초 이상 지속될 때만 경보' 규칙 만들기 |
| 주의할 점 | 1순위 데이터에는 증강 사본(파일명 `_aug1~3`)이 섞여 있습니다. 같은 원본의 사본이 학습과 평가에 나뉘어 들어가면 점수가 부풀려집니다. 파일명으로 확인해 보세요. |

### T3. 재활용 쓰레기 자동 분류  (환경 · ★☆☆ 쉬움)

| 항목 | 내용 |
|---|---|
| 풀어 볼 질문 | 선별장 사진에서 종이·플라스틱·유리·금속을 구분해 위치를 찾을 수 있는가? |
| 1순위 데이터 | 캐글 Garbage Detection – 6 Waste Categories<br>`viswaprakash1990/garbage-detection`<br>YOLO + data.yaml (train/valid/test), 6클래스(Biodegradable, cardboard, glass, metal, paper, plastic) · 약 206MB · CC BY 4.0 |
| 대체 데이터 | 캐글 WaRP 재활용 선별장<br>`parohod/warp-waste-recycling-plant-dataset`<br>약 3,000장(1920×1080), 28클래스 · 약 843MB · CC0 · 라벨 좌표 형식 확인 필요 |
| 성공 기준 예시 | 6클래스 평균 mAP50 0.6 이상, 가장 헷갈리는 두 클래스 찾기 |
| 더 해 보기 | YOLOE에 'plastic bottle' 같은 단어로 찾게 해 학습 모델과 비교 (추가 실습 '말로 찾는 탐지기') |
| 주의할 점 | 혼동 행렬에서 어떤 재질끼리 헷갈리는지 보는 것이 핵심입니다. |

### T4. 도로 파손(포트홀·균열) 탐지와 심각도  (교통·도로 · ★★☆ 보통)

| 항목 | 내용 |
|---|---|
| 풀어 볼 질문 | 도로 사진에서 포트홀·균열·맨홀을 구분하고, 보수가 급한 곳을 고를 수 있는가? |
| 1순위 데이터 | 캐글 Road Damage: Potholes, Cracks, Manholes<br>`lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`<br>YOLO, 2,000장 이상(640×360), 3클래스(Pothole, Crack, Manhole) · 약 195MB · MIT |
| 대체 데이터 | 캐글 Pothole Detection<br>`andrewmvd/pothole-detection`<br>Pascal VOC xml, 665장, 1클래스 · 약 351MB · 시작 노트북이 자동으로 YOLO 변환 |
| 성공 기준 예시 | Pothole 클래스 mAP50 0.6 이상 |
| 더 해 보기 | 박스 면적으로 포트홀 크기를 S/M/L로 나눠 '보수 우선순위' 표 만들기 (실습 A Step 5 방식) |
| 주의할 점 | 맨홀과 포트홀은 둘 다 둥근 어두운 물체라 헷갈리기 쉽습니다. 오류 분석에서 꼭 확인하세요. |

### T5. 매장 선반 진열 상품 인식  (물류·유통 · ★★☆ 보통)

| 항목 | 내용 |
|---|---|
| 풀어 볼 질문 | 선반 사진에서 상품 종류별 개수를 세어 진열 상태를 자동으로 점검할 수 있는가? |
| 1순위 데이터 | 캐글 Shelf Detection<br>`krishnayadav456wrsty/shelf-detection`<br>YOLO(설명 기준), 약 1,040장, 11클래스(Bottle, Box, Can, Pack, Snack 등) · 약 98MB · MIT |
| 대체 데이터 | 캐글 Warehouse Delivery Box Detection<br>`zoya77/warehouse-delivery-box-detection-dataset`<br>YOLO txt · 약 867MB · CC0 |
| 성공 기준 예시 | 전체 mAP50 0.6 이상, 사진별 상품 개수 표(CSV) 출력 |
| 더 해 보기 | 사진 한 장의 클래스별 개수를 '기준 진열표'와 비교해 부족한 상품을 알려 주기 |
| 주의할 점 | 클래스 이름 중 '-'처럼 의미가 불분명한 것이 있습니다. 데이터 살펴보기에서 확인하고 필요하면 제외하세요. |

### T6. 낙상 감지 — 넘어진 사람 찾기  (사람 행동 · ★★☆ 보통)

| 항목 | 내용 |
|---|---|
| 풀어 볼 질문 | 실내 영상에서 넘어진 사람을 걷거나 앉은 사람과 구분해 찾을 수 있는가? |
| 1순위 데이터 | 캐글 Fall Detection Dataset<br>`uttejkumarkandagatla/fall-detection-dataset`<br>txt 라벨(클래스 + 박스 4값), 485장(374/111), 3클래스(Fall Detected, Walking, Sitting) · 약 53MB · ODbL |
| 대체 데이터 | 포즈 방식으로 직접 만들기<br>`실습 B 노트북`<br>포즈 모델 + 몸통 기울기 규칙으로 '누운 자세(NOT_STANDING)'를 찾는 방식과 비교 |
| 성공 기준 예시 | Fall Detected 재현율 0.8 이상 |
| 더 해 보기 | 넘어진 상태가 3초 이상 이어질 때만 경보 (위험구역 실습의 체류 시간 로직) |
| 주의할 점 | 이미지가 485장으로 적습니다. 검증 점수가 크게 흔들리므로 실험을 여러 번 비교할 때 주의하세요. 라벨 좌표가 0~1로 정규화되어 있는지 데이터 살펴보기 그림으로 먼저 확인하세요. |

### T7. 작물 해충 탐지  (농업 · ★★☆ 보통)

| 항목 | 내용 |
|---|---|
| 풀어 볼 질문 | 작물 사진에서 해충 12종을 구분해 찾을 수 있는가? 작은 해충도 놓치지 않는가? |
| 1순위 데이터 | 캐글 AgroPest-12<br>`rupankarmajumdar/crop-pests-dataset`<br>YOLO txt, 12종 해충 · 약 563MB · MIT |
| 대체 데이터 | 캐글 Fruit Detection<br>`lakshaytyagi01/fruit-detection`<br>YOLO, 8,479장, 6클래스(Apple, Grapes, Pineapple, Orange, Banana, Watermelon) · 약 528MB · CC0 |
| 성공 기준 예시 | 12종 평균 mAP50 0.5 이상, 성능이 낮은 3종의 원인 분석 |
| 더 해 보기 | `imgsz`를 640 → 960으로 키워 작은 해충 재현율이 얼마나 오르는지 실험 |
| 주의할 점 | 작은 물체가 많습니다. 데이터 살펴보기의 '1% 미만 박스 비율'을 먼저 확인하세요. |

### T8. PCB 표면 결함 검사  (제조 품질 · ★★★ 도전)

| 항목 | 내용 |
|---|---|
| 풀어 볼 질문 | 인쇄회로기판 사진에서 9종의 미세 결함을 찾아낼 수 있는가? |
| 1순위 데이터 | 캐글 DsPCBSD+<br>`enisteper1/dataset-of-pcb-surface-defects-dspcbsd`<br>YOLO(Data_YOLO 폴더), 10,259장(8,208/2,051), 9클래스(Short, Spur, Open, Mouse Bite 등) · 약 65MB · CC BY 4.0 |
| 대체 데이터 | 캐글 NEU Surface Defect (강판 결함)<br>`kaustubhdikshit/neu-surface-defect-database`<br>Pascal VOC xml, 6종 결함 · 약 28MB · 시작 노트북이 자동으로 YOLO 변환 |
| 성공 기준 예시 | 9종 평균 mAP50 0.5 이상, 클래스별 성능 차이 설명 |
| 더 해 보기 | 결함 종류별 '불량 판정 기준(conf)'을 다르게 정해 검사 라인 규칙표 만들기 |
| 주의할 점 | 결함이 매우 작고 이미지 수가 많습니다. 처음에는 `FRACTION=0.3`으로 빠르게 베이스라인을 만든 뒤 늘리세요. |

## 3. 데이터 확보 방법

시작 노트북의 `DATA_SOURCE` 값으로 받는 방법을 고릅니다. 받은 뒤의 형식 확인·변환·분할은 노트북이 자동으로 합니다.

| 방법 | `DATA_SOURCE` | 채울 값 | 언제 쓰나 |
|---|---|---|---|
| 캐글 | `'kaggle'` | `KAGGLE_SLUG` = 주소의 `작성자/이름` | 주제 표의 대부분 데이터 |
| 다운로드 링크 | `'url'` | `ZIP_URL` = zip 파일 직접 링크 | 공공데이터·GitHub 등에서 zip 링크를 제공할 때 |
| 내 Drive | `'drive'` | `DRIVE_ZIP` = Drive 안 zip 경로 | AI Hub처럼 따로 받아 Drive에 올려 둔 경우, 멘토가 공유한 백업 데이터 |
| 예시 데이터 | `'ultralytics'` | `ULTRA_YAML` = 설정 이름 | 연습, 또는 다른 방법이 모두 막혔을 때 (T1의 `construction-ppe.yaml` 등) |

### 3-1. 캐글 데이터 받기

1. kaggle.com에서 데이터셋 페이지를 엽니다. 주소가 `kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow`라면 `KAGGLE_SLUG`는 `snehilsanyal/construction-site-safety-image-dataset-roboflow`입니다.
2. 시작 노트북에서 `DATA_SOURCE = 'kaggle'`, `KAGGLE_SLUG = '...'`로 바꾸고 실행합니다. 공개 데이터셋은 대부분 로그인 없이 받아집니다.
3. 401·403 오류가 나면 인증이 필요한 것입니다. 캐글 로그인 → 오른쪽 위 프로필 → **Settings → API**에서 토큰을 만듭니다.
4. Colab 왼쪽의 **열쇠 아이콘(보안 비밀)** 을 눌러 이름 `KAGGLE_API_TOKEN`, 값에 토큰을 넣고 '노트북 액세스'를 켭니다. 예전 방식의 `kaggle.json` 파일을 받았다면 안의 `username`, `key`를 `KAGGLE_USERNAME`, `KAGGLE_KEY` 두 개로 등록합니다.
5. 다시 실행합니다. 토큰을 코드에 직접 적지 마세요. 노트북을 공유하면 토큰도 함께 퍼집니다.

> **참고** 경진대회(competition) 데이터는 웹에서 대회 규칙에 먼저 동의해야 받을 수 있습니다. 워크숍에서는 경진대회가 아닌 **데이터셋(datasets)** 을 씁니다.

### 3-2. 공공데이터 (AI Hub, 공공데이터포털)

- **AI Hub(aihub.or.kr)** 에는 공사현장 안전장비, 생활 폐기물, 화재, 도로 노면 파손, 낙상 등 주제와 맞는 대용량 데이터가 있습니다. 다만 회원가입 → 이용 신청 → **승인 대기** → 전용 다운로드 도구(`aihubshell`) 순서를 거쳐야 하고, 내국인만 신청할 수 있으며, 용량이 수십 GB 이상인 경우가 많습니다.
- 그래서 워크숍 기간 안에 직접 신청해서 쓰기는 어렵습니다. 관심 있다면 워크숍 이후 개인 프로젝트로 시도하거나, 멘토가 Drive에 준비한 축소본이 있으면 `DATA_SOURCE = 'drive'`로 씁니다.
- AI Hub 라벨은 대부분 JSON 형식이라 YOLO 형식으로 바꾸는 변환 코드가 따로 필요합니다. COCO와 비슷한 구조라면 시작 노트북의 COCO 변환을 참고해 고칠 수 있습니다.
- **공공데이터포털(data.go.kr)** 은 표·API 데이터가 대부분이라 박스 라벨이 있는 이미지 데이터는 드뭅니다. zip 직접 링크가 있는 이미지 데이터를 찾았다면 `DATA_SOURCE = 'url'`로 받습니다.

### 3-3. 라벨 형식

| 형식 | 알아보는 법 | 시작 노트북 처리 |
|---|---|---|
| YOLO | 이미지마다 같은 이름의 `.txt`, 한 줄 = `클래스 cx cy w h` (0~1 사이 값). `data.yaml`이나 `classes.txt`에 클래스 이름 | 그대로 사용. 클래스 이름 파일이 없으면 `CLASS_NAMES` 직접 입력 |
| Pascal VOC | 이미지마다 같은 이름의 `.xml`, 안에 `<xmin> <ymin> <xmax> <ymax>` 픽셀 좌표 | 자동 변환 |
| COCO | `.json` 하나에 `images`, `annotations`, `categories`. `bbox = [x, y, w, h]` 픽셀 | 자동 변환 |
| 세그멘테이션 다각형 | YOLO txt인데 한 줄에 숫자가 5개보다 많음 | 감싸는 박스로 자동 변환 |

## 4. 시작 노트북 사용법

`02_yolo_workshop_starter_v1.1.ipynb`를 Drive에 복사해 Colab으로 엽니다. `✏️` 표시가 있는 셀만 우리 팀 내용으로 고치고, 나머지는 위에서부터 순서대로 실행합니다.

| 단계 | 고칠 값 (✏️) | 결과물 |
|---|---|---|
| 1. 프로젝트 카드 | `PROJECT_NAME`(영문·숫자·밑줄), `TEAM`, `PROBLEM`, `SUCCESS`, `TASK` | Drive에 `yolo_workshop/<PROJECT_NAME>` 폴더 |
| 2. 데이터 받기 | `DATA_SOURCE`와 해당 값 | `raw_data/` 폴더, 파일 종류·폴더 구조 출력 |
| 3. 형식 변환 | 필요할 때만 `CLASS_NAMES` | `dataset/data.yaml` |
| 4. 데이터 살펴보기 | 확인 질문 메모 | `eda.png`, 샘플 그림 |
| 5. 베이스라인 | `QUICK_RUN`, `FRACTION` | `baseline_best.pt` |
| 6. 평가 | `FOCUS_CLASS`, `SERVICE_CONF`, 오류 분류표 | `per_class.csv`, 임계값 표, 오류 사진 |
| 7. 개선 실험 | `EXPERIMENTS`, `RUN_EXPERIMENTS` | `experiments.csv`, 실험별 `*_best.pt` |
| 8. 데모 | `UPLOAD_DEMO` | `demo_1~3.jpg` |
| 9. 결과 정리 | 리포트의 Notes | `project_report.md`, `report.png` |

- **처음에는 `QUICK_RUN = True`** 로 끝까지 한 번 돌려 오류가 없는지 확인한 뒤, `False`로 바꿔 본 학습을 합니다.
- 1 epoch가 10분 넘게 걸리면 `FRACTION`을 0.3 정도로 낮춥니다.
- Colab 무료 GPU는 사용 시간 제한이 있습니다. 학습 결과(`*_best.pt`)와 실험 기록은 Drive에 자동 저장되니, 런타임이 끊기면 다시 연결하고 1~4단계를 다시 실행한 뒤 이어서 하면 됩니다.
- 2인 1조라면 한 사람은 학습을 돌리고, 다른 한 사람은 데이터 살펴보기와 오류 사진 분석을 맡는 식으로 나누면 시간을 아낄 수 있습니다.

## 5. 결과 공유

3회차 끝에 결과를 공유합니다. 방식은 그날 안내합니다. 어느 방식이든 시작 노트북 9단계가 만든 `project_report.md`와 `report.png`를 그대로 씁니다.

| 담을 내용 | 어디서 가져오나 |
|---|---|
| 풀어 본 질문과 성공 기준 | 프로젝트 카드 |
| 데이터 출처·형식·분할 | 리포트의 Data 항목 |
| 베이스라인 → 개선 실험 결과 | `experiments.csv`, `report.png` 왼쪽 그래프 |
| 서비스 임계값과 그 이유 | 6-2 임계값 표, `report.png` 오른쪽 그래프 |
| 잘된 것, 안 된 것, 다음에 할 일 | 리포트 Notes (직접 작성) |

## 6. 자주 막히는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| 캐글 다운로드에서 401·403 오류 | 인증 필요, 또는 보안 비밀의 노트북 액세스가 꺼져 있음 | 3-1의 토큰 등록 후 '노트북 액세스'를 켜고 다시 실행 |
| 라벨 형식: unknown | 라벨 파일이 없거나 지원하지 않는 형식 | 폴더 구조 출력을 보고 멘토와 확인. 대체 데이터로 전환 검토 |
| 클래스 이름을 찾지 못했다는 오류 | YOLO 형식인데 이름 파일이 없음 | 데이터 페이지 설명에서 클래스 순서를 찾아 `CLASS_NAMES`에 입력 |
| 샘플 그림의 박스 위치가 이상함 | 좌표 형식이 예상과 다름(픽셀 좌표, x·y 순서 등) | 원본 라벨 한 줄을 열어 값 범위를 확인하고 멘토와 상의 |
| 학습이 너무 느림 | 데이터가 크거나 GPU가 아님 | `GPU 사용 가능: True` 확인, `FRACTION` 낮추기, `imgsz` 줄이기 |
| CUDA out of memory | 배치가 GPU 메모리보다 큼 | `CFG`의 `batch`를 8로 낮춤 |
| 런타임이 끊김 | Colab 사용 시간·유휴 제한 | 다시 연결 → 1~4단계 실행 → Drive에 저장된 결과로 이어서 진행 |
| 점수가 지나치게 높게 나옴 | 같은 원본의 증강 사본이 학습과 평가에 나뉘어 들어감 | 파일명 패턴을 확인하고 멘토와 분할 방법 상의 |
