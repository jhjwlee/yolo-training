# YOLO Colab 부트캠프 실습 자료

Google Colab에서 Ultralytics YOLO로 따라 하는 단계별 핸즈온 실습 노트북과 해설서입니다. 모든 노트북은 Colab의 **T4 GPU** 런타임 기준으로 작성되었고, 결과는 Google Drive에 저장됩니다.

## 1회차 — 현업 사례로 만드는 포트폴리오

| 노트북 | 내용 | 열기 |
|---|---|---|
| [`yolo_day1_ppe_lab_v1.1.ipynb`](day1/yolo_day1_ppe_lab_v1.1.ipynb) | 건설 현장 안전장비(PPE) 미착용 감지 — 학습·평가·판정 규칙·포트폴리오 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day1/yolo_day1_ppe_lab_v1.1.ipynb) |
| [`yolo_day1_case_demos_v1.1.ipynb`](day1/yolo_day1_case_demos_v1.1.ipynb) | 박스 카운팅 · 운동/낙상 포즈 · 주차장/대기열 사례 시연 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day1/yolo_day1_case_demos_v1.1.ipynb) |
| [`yolo_day1_pill_lab_v1.0.ipynb`](day1/yolo_day1_pill_lab_v1.0.ipynb) | 알약 탐지 — 라벨·IoU·NMS·AP부터 오류 분석까지 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day1/yolo_day1_pill_lab_v1.0.ipynb) |
| [`yolo_day1_brain_tumor_lab_v1.0.ipynb`](day1/yolo_day1_brain_tumor_lab_v1.0.ipynb) | 뇌종양 영상 탐지 — 클래스별 AP·혼동 행렬·분할 점검 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day1/yolo_day1_brain_tumor_lab_v1.0.ipynb) |

1회차 핸즈온 가이드: [웹에서 읽기](day1/yolo_day1_handson_guide_v1.1.md) · [pdf](day1/yolo_day1_handson_guide_v1.1.pdf) · [docx](day1/yolo_day1_handson_guide_v1.1.docx)

## 2회차 — 데모를 넘어 직접 구현하기

| 노트북 | 내용 | 열기 |
|---|---|---|
| [`yolo_day2_box_lab_v1.1.ipynb`](day2/yolo_day2_box_lab_v1.1.ipynb) | 실습 A · 컨베이어 상자 분할 학습 → 추적·라인 집계 → S/M/L 크기별 카운팅 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/yolo_day2_box_lab_v1.1.ipynb) |
| [`yolo_day2_zone_lab_v1.0.ipynb`](day2/yolo_day2_zone_lab_v1.0.ipynb) | 위험구역 침입 감시 — 구역 그리기, 발끝 판정, 진입/이탈 이벤트, 체류 시간, 히트맵 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/yolo_day2_zone_lab_v1.0.ipynb) |
| [`yolo_day2_pose_lab_v1.1.ipynb`](day2/yolo_day2_pose_lab_v1.1.ipynb) | 실습 B · 상자 들기 동작 단위 분석 (포즈 키포인트, 들기 사이클) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/yolo_day2_pose_lab_v1.1.ipynb) |
| [`yolo_day2_monitor_lab_v1.0.ipynb`](day2/yolo_day2_monitor_lab_v1.0.ipynb) | 통합 · 물류센터 모니터링 (상자 집계 + 작업자 동작) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/yolo_day2_monitor_lab_v1.0.ipynb) |
| [`yolo_day2_prompt_lab_v1.0.ipynb`](day2/yolo_day2_prompt_lab_v1.0.ipynb) | 말로 찾는 탐지기 — YOLOE 텍스트 프롬프트, 프롬프트 대결, 학습 모델과 비교 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/yolo_day2_prompt_lab_v1.0.ipynb) |
| [`yolo_day2_autolabel_lab_v1.0.ipynb`](day2/yolo_day2_autolabel_lab_v1.0.ipynb) | 오토 라벨링 공장 — YOLOE 자동 라벨, 품질 검사, 사람/자동/하이브리드 비교 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/yolo_day2_autolabel_lab_v1.0.ipynb) |

### 2회차 해설서

코드를 한 셀씩 설명하고, 실행 결과, 막혔을 때 해결법, 도전 과제를 정리했습니다. 웹에서 바로 읽거나 Word 파일로 받을 수 있습니다.

| 실습 | 해설서 |
|---|---|
| 실습 A · 상자 크기별 집계 | [웹에서 읽기](day2/guides/yolo_day2_box_lab_guide_v1.0.md) · [docx](day2/guides/yolo_day2_box_lab_guide_v1.0.docx) |
| 위험구역 침입 감시 | [웹에서 읽기](day2/guides/yolo_day2_zone_lab_guide_v1.0.md) · [docx](day2/guides/yolo_day2_zone_lab_guide_v1.0.docx) |
| 실습 B · 들기 동작 분석 | [웹에서 읽기](day2/guides/yolo_day2_pose_lab_guide_v1.0.md) · [docx](day2/guides/yolo_day2_pose_lab_guide_v1.0.docx) |
| 통합 · 물류센터 모니터링 | [웹에서 읽기](day2/guides/yolo_day2_monitor_lab_guide_v1.0.md) · [docx](day2/guides/yolo_day2_monitor_lab_guide_v1.0.docx) |
| 말로 찾는 탐지기 | [웹에서 읽기](day2/guides/yolo_day2_prompt_lab_guide_v1.0.md) · [docx](day2/guides/yolo_day2_prompt_lab_guide_v1.0.docx) |
| 오토 라벨링 공장 | [웹에서 읽기](day2/guides/yolo_day2_autolabel_lab_guide_v1.0.md) · [docx](day2/guides/yolo_day2_autolabel_lab_guide_v1.0.docx) |

- 실습 B와 통합 실습은 상자를 드는 동작을 옆에서 찍은 10~20초 영상(`lift_sample.mp4`)이 필요합니다. 휴대폰으로 직접 찍어 노트북의 파일 선택 창에서 올리면 되고, 아이폰 `.mov`도 그대로 쓸 수 있습니다. 찍는 요령은 [실습 B 해설서](day2/guides/yolo_day2_pose_lab_guide_v1.0.md)에 있습니다.
- 말로 찾는 탐지기의 Step 4는 실습 A에서 Drive에 저장한 모델(`MyDrive/yolo_day2/box_seg_best.pt`)을 사용합니다.

## 프로젝트 워크숍 (2~3회차)

주제 8개(산업 안전, 화재·연기, 재활용 쓰레기, 도로 파손, 매장 선반, 낙상, 작물 해충, PCB 결함) 중 하나를 골라 개인 또는 2인 1조로 진행합니다.

| 자료 | 내용 | 열기 |
|---|---|---|
| [`yolo_workshop_starter_v1.0.ipynb`](workshop/yolo_workshop_starter_v1.0.ipynb) | 시작 노트북 — 캐글·URL·Drive 데이터 받기, YOLO/VOC/COCO 형식 자동 변환, 학습·평가·임계값·오류 분석·실험 기록·리포트 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/workshop/yolo_workshop_starter_v1.0.ipynb) |
| 워크숍 안내서 | 진행 흐름, 체크포인트, 주제 카드 8개, 데이터 확보 방법 | [웹에서 읽기](workshop/yolo_workshop_student_guide_v1.0.md) · [docx](workshop/yolo_workshop_student_guide_v1.0.docx) |

> 캐글 API 토큰은 코드에 적지 말고 Colab 왼쪽의 **보안 비밀(열쇠 아이콘)** 에 `KAGGLE_API_TOKEN`으로 등록하세요.

## 참고

- 실습 데이터셋과 사전학습 모델은 [Ultralytics](https://docs.ultralytics.com) 공개 자산(AGPL-3.0)과 각 캐글 데이터셋의 라이선스를 따릅니다.
- 노트북은 ultralytics 8.4 버전에서 동작을 확인했습니다.
