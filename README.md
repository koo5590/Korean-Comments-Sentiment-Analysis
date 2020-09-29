2020 국어 정보 처리 시스템 경진 대회
==================================
영화 리뷰, 스포츠 댓글, TV연예프로그램 댓글 각각에 대한 긍부정 분류기와, 라벨링 데이터 없이 학습한 분류기까지 총 4개의 모델을 제시합니다.

***docker image 다운로드 링크*** 는 제출한 **보고서 마지막 페이지**에 있습니다. 실행방법 또한 보고서에 기재하였습니다. 다운로드 후 아래 방법을 따라 테스트 성능과 실시간 테스트를 확인해주시길 바랍니다. 

2차 경진대회 추가 사항
------
new_model_4 추가. 1차 실험의 model 4에서 learning rate 등 하이퍼파라미터 조정 실험.

테스트 파일 성능 확인 방법

    python main.py \
    --do_test --new_model_4 \
    --test_model_dir ./models/new_model_4 \
    --test_data_dir sports_test


Model 1 : 영화 리뷰 분류기
--------
KC BERT + 네이버 영화 리뷰 데이터 finetuning 모델

테스트 파일 성능 확인 방법

    python main.py \
    --do_test --model_1 \
    --test_model_dir ./models/movie \
    --test_data_dir movie_test

학습 방법

    python main.py \
    --do_train --model_1 \
    --train_data_dir movie_train

Model 2 : 스포츠 댓글 분류기
--------------------------
KC BERT + LSTM + 네이버 영화 리뷰 데이터 finetuning + 스포츠 댓글 데이터 finetuning 모델

### 테스트 파일 성능 확인 방법

    python main.py \
    --do_test --model_2 \
    --test_model_dir ./models/sports/model_sports.pt \
    --test_data_dir sports_test

### 학습 방법

1. 네이버 영화 리뷰로 먼저 finetuning 한다.

        python main.py \
        --do_train --model_2 \
        --train_data_dir movie_train

2.  1.의 모델에 스포츠 댓글 데이터로 finetuning 한다.
        
        python main.py \
        --do_train --model_2 --second_finetuning \
        --test_model_dir ./models/model_sports.pt \
        --train_data_dir sports_train \
        --num_train_epochs 3 \
        --logging_steps 25 --gradient_accumulation_steps 1

Model 3 : TV프로그램 댓글 분류기
-------
KC BERT + 네이버 영화 리뷰 데이터 finetuning + TV프로그램 댓글 finetuning 모델
### 테스트 파일 성능 확인 방법

    python main.py \
    --do_test --model_3 \
    --test_model_dir ./models/tv \
    --test_data_dir tv_test

## 학습 방법

1. 네이버 영화 리뷰로 먼저 finetuning 한다.

        python main.py \
        --do_train --model_3 \
        --train_data_dir movie_train

2.  1.의 모델에 TV프로그램 댓글 데이터로 finetuning 한다. 
        
        python main.py \
        --do_train --model_3 \
        --test_model_dir ./models \
        --train_data_dir tv_train \
        --num_train_epochs 3 \
        --logging_steps 25 --gradient_accumulation_steps 1


Model 4 : 라벨링 데이터 없이 스포츠 댓글 긍부정 분류기 학습
---
1. KC BERT 로 영화, 스포츠 도메인 분류기 학습 (input 데이터가 영화 리뷰인지 스포츠 댓글인지 분류)
2. 1. 에서 학습한 분류기로 영화 리뷰를 분류하여 스포츠 댓글과 최대한 비슷한 영화 리뷰 찾기 (1의 도메인 분류기가 스포츠 댓글이라고 분류하는 것들만 뽑아내기)
3. 2. 에서 얻은 <스포츠 데이터와 비슷한 영화 리뷰> 로 스포츠 댓글을 위한 긍부정 분류기 학습. 이때 <진짜 스포츠 댓글> 도 모델에 넣어 BERT를 통해 나오는 벡터간의 Maximum Mean Discrepancy 를 계산. 긍부정 분류기의 loss에 더하여 SGD로 최적화한다.

### 테스트 파일 성능 확인 방법

    python main.py \
    --do_test --model_4 \
    --test_model_dir ./models/no_labeling \
    --test_data_dir sports_test

### 학습방법
아래 코드로 도메인 분류기 / 스포츠 댓글과 비슷한 영화 리뷰 선별 / 긍부정 분류기 학습이 한번에 수행된다.

        python main.py \
        --do_train --model_4 \
        --train_data_dir movie_train

실시간 테스트
---------
실행방법

1. Model 1 : 영화 테스트

        python main.py \
        --do_interactive --model_1 \
        --test_model_dir ./models/movie \
        --test_data_dir movie_test


2. Model 2 : 스포츠 테스트

        python main.py \
        --do_interactive --model_2 \
        --test_model_dir ./models/sports/model_sports.pt \
        --test_data_dir sports_test

3. Model 3 : TV프로그램 테스트

        python main.py \
        --do_interactive --model_3 \
        --test_model_dir ./models/tv \
        --test_data_dir tv_test

4. Model 4 : 스포츠 테스트

        python main.py \
        --do_interactive --model_4 \
        --test_model_dir ./models/no_labeling \
        --test_data_dir sports_test

결과
---
Model 1 ~ 3
|Model 1 (영화)|Model 2 (스포츠)|Model 3 (TV프로그램)|
|:------:|:---:|:---:|
|**0.901**|**0.823**|**0.872**|


Model 4
|Baseline (no finetuning)|Model 1 (영화)|Model 4 (no label/스포츠)|new_model_4|
|:------:|:---:|:---:|:---:|
|0.501|0.757|<span style="color:red">**0.792**</span>|<span style="color:red">**0.798**</span>|



학습 모델 다운로드 링크
-----
모델 용량이 커 github 업로드가 불가하여 구글 드라이브 업로드 후 공유합니다.

도커 환경에 모델을 넣었으므로 도커 컨테이너 작동 시 별도의 모델 다운로드 없이 테스트 해보실 수 있습니다.

https://drive.google.com/drive/folders/1AnQS4XrQa65UjVNAm_swF_y-lyTZ-lDf?usp=sharing
