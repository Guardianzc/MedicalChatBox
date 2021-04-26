### Description

This is a simple training repo for symptom recognition task in [imcs21](http://www.fudan-disc.com/sharedtask/imcs21/index.html), 
this script is written by <chenwei18@fudan.edu.cn>.

### Usage 

- pre-process 

```shell
python preprocess.py
```

- training

```shell
python train.py
```

- inference
```shell
python inference.py
```

- evaluate
```shell
python evaluate.py './dataset/test_all.json' './dataset/predictions.json'
```

### Experimental details

| Metric               | Value                |
| -------------------- | -------------------- |
| F1 score on test set | 67.9919%             |
| Training Epochs      | 20                   |
| Training Time        | 8h                   |
| CUDA                 | 10.1.243             |
| GPU                  | Tesla P100 PCIe 16GB |
| Linux Release        | Ubuntu 18.04.5 LTS   |


| Package              | Version              |
| -------------------- | -------------------- |
| Python               | 3.7                  |
| torch                | 1.8.1                |
| transformers         | 4.5.1                |
| pandas               | 1.2.0                |
| numpy                | 1.19.2               |

