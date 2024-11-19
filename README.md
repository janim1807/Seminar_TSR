# Seminar_TSR Readme

## Overview

Welcome to the Seminar_TSR project! This project focuses on Benchmarking different saliency methods for explainable AI with or without TSR based on: https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark

## Prerequisites

Before running the code, ensure that your directory structure looks as follows:

- Accuracy_Metrics
- MaskingData
- MaskingImages
- Models
- Precision_Recall

Also, make sure to run:

```bash
pip install -r requirements.txt
```

## Calling the Benchmarking Function

To evaluate the saliency methods, use the `Benchmarking` class with the following parameters (All parameters are optional, if you don't choose anything, we use a Middle_AutoRegressive Dataset and train it on a Transformer, for the explainer we use IGFlag and TSR is true )

```python
Benchmarking(training_data_path, training_meta_path, testing_data_path, testing_meta_path, model_to_explain, saliency_method, TSR)
```


for the saliency method check this dictionary choose this:
```python

 {
                'GradFlag': 'GRAD',
                'IGFlag': 'IG',
                'DLFlag': 'DL',
                'GSFlag': 'GS',
                'DLSFlag': 'DLS',
                'SGFlag': 'SG',
                'ShapleySamplingFlag': 'SVS',
                'FeatureAblationFlag': 'FA',
                'OcclusionFlag': 'FO'
            }
```