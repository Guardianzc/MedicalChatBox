# Track1- Task2：MLC-NEAR

赛道一任务二（症状识别）的多标签分类基础模型，供参考。

## 0. Set Up

### 0.1 Dataset

数据集从网站**[第一届智能对话诊疗评测比赛](http://www.fudan-disc.com/sharedtask/imcs21/index.html)**中下载，将dataset文件夹放在Track1文件夹内。

### 0.2 Requirements

- python>=3.7
- torch>=1.5
- transformers>=3.0
- pandas
- sklearn

## 1. Data Preprocess 

预处理训练数据，将在data文件夹下生成processed文件夹

```
cd data
python preprocess.py
```

## 2. Training

```
python train.py
```

## 3. Predicting

```
python inference.py
```
将在data文件夹下输出预测结果文件`submission_track1_task2.json`


## 4. Evaluation

```
python eval_track1_task2.py {gold_data_path} {pred_data_path}
```

`gold_data_path`是具有真实标签的测试集的路径，`pred_data_path`是`submission_track1_task2.json`的路径。将`submission_track1_task2.json`提交后系统将会自动评价。

