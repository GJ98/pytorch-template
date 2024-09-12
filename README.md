# Manual
하이퍼 파라미터를 수정하고 싶은 경우, `config.json` 파일에서 수정하시면 됩니다.
커스터마이징을 하고 싶은 경우, `module` 폴더 안에서 작업하시면 됩니다.
## 1. customizing dataset
1. `module/dataset.py`에서 `BaseDataset`을 상속 받은 클래스를 생성합니다.
2. 해당 클래스에 `preprocessing`함수를 구현합니다.  
  이때, 입력값은 `pandas.core.frame.DataFrame` 객체가 되어야 합니다.  
출력값은 `(inputs, targets)`으로, `(List[List[int]], List[int or long])` 형태가 되어야 합니다.
3. 마지막으로, `config.json` 파일에서 `data_module.args.dataset_name`값을 "데이터셋 클래스 이름"(e.g., `STSDataset`)으로 변경합니다.

## 2. customizing model
1. `module/model.py`에서 `nn.Module`을 상속 받아 pytorch model을 구현합니다.
2. `config.json` 파일에서 `arch.type`값을 "모델 클래스 이름"으로 변경합니다.

참고로, 사전 학습 모델을 변경하고 싶을 경우 `arch.args.plm_name`값을 "변경하고자 하는 사전 학습 모델 이름"(e.g., `klue/roberta-small`)으로 변경하면 됩니다.

## 3. customizing loss
1. `module/loss.py`에서 손실 함수를 구현합니다.  
이때, 손실 함수의 출력값은 스칼라값이여야 합니다.
2. `config.json` 파일에서 `loss`값을 "손실 함수 이름"(e.g., `l2_loss`)으로 변경합니다.

## 4. customizing metrics
1. `module/metric.py`에서 평가 지표 함수를 구현합니다.  
이때, 평가 지표 함의 출력값은 스칼라값이여야 합니다.
2. `config.json` 파일에서 `metrics` 리스트에 "평가 지표 함수 이름"(e.g., `pearson`)을 추가합니다.

참고로, 모델 저장을 할 때 **metrics 리스트의 첫번째 평가 지표**를 사용할 것이며, max/min 기준은 `trainer.mode`에서 설정하시면 됩니다.
그리고, 저장 위치는 `trainer.save_dir`에서 확인할 수 있습니다.


## 추가
- `wandb`를 사용할 경우 `config.json` 파일에서, `wandb.project_name`, `wandb.run_name`을 적절히 설정해야 합니다.

