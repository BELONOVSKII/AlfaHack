This repository contains the code for the first stage of the **AlfaHack AutoML hackaton**.

Team: ***DDDrkBBB***.
Members: [Peter Belonovskiy](https://github.com/BELONOVSKII), [Kristina Galuzina](https://github.com/galuzina-k), [Timofey Lashukov](https://github.com/M1croZavr)

<center><img src="assets/sber_kot.gif" width="200" height="200" /></center>

## General info
* Track: *Отток юридических лиц из расчетно-кассового обслуживания.*
* Leader board score: $81.7015322279558$
* Leader board position: **3**.


## TL;DR 
Tune + fit 7 models on selected features. Each model is a mean blend on 5 stratified folds. Blend the best model with the model staked on the selected features + oof predictions.

## Repository structure
```
.
├── README.md
├── assets
├── automl      <-- clone of an open source automl package.
├── best_res    <-- notebooks to reproduce the best result.
│   ├── blend.ipynb
│   ├── create_stack_df.ipynb
│   ├── data_processing.ipynb
│   ├── fit_cb.ipynb
│   ├── fit_lama.ipynb
│   ├── fit_lama_autoint.ipynb
│   ├── fit_lama_fttransformer.ipynb
│   ├── fit_lama_stack.ipynb
│   ├── fit_lama_utilized.ipynb
│   ├── fit_lgb.ipynb
│   ├── fit_xgb.ipynb
│   └── select_features.ipynb
├── configs
│   └── config.yaml   <-- config file.
├── data
│   ├── models        <-- folder with model artifacts.
│   ├── train         <-- raw train data.
│   ├── test          <-- raw test data.
│   └── .             <-- processed data files.
├── notebooks   <-- many-many-many-many various experiments.
├── requirements_gpu.txt  <-- python requirements on a gpu server.
└── requirements.txt  <-- python requirements on a cpu server.
```

## Reproduce results
To reproduce the results run the notebooks in `best_res` in the following order:
1. `best_res/data_processing.ipynb` - Basic feature processing. Saves processed dataset files in `data/` folder.
2. `best_res/select_features.ipynb` - Selects features. Saves selected features to the `configs/config.yaml`.
3. `best_res/fit_lgb.ipynb` - fits + tunes LightGBM. Saves model file, oof predictions, model params and test predictions in `data/model/lgb_8122_full_dataset/`.
4. `best_res/fit_xgb.ipynb` - fits + tunes XGBoost. Saves model file, oof predictions, model params and test predictions in `data/model/xgb_81325_full_dataset/`.
5. `best_res/fit_cb.ipynb` - fits + tunes CatBoost. Saves model file, oof predictions, model params and test predictions in `data/model/cb_8114_full_dataset/`.
6. `best_res/fit_lama.ipynb` - fits + tunes LightAutoML. Saves model file, oof predictions, model params and test predictions in `data/model/lama_81298_full_dataset/`.
7. `best_res/fit_lama_utilized.ipynb` - fits + tunes LightAutoMLUtilized. Saves model file, oof predictions, model params and test predictions in `data/model/lamau_81425_full_dataset/`.
8. `best_res/fit_lama_autoint.ipynb` - fits LightAutoML AutoInt. Saves model file, oof predictions, model params and test predictions in `data/model/lamann_autoint_8053_full_dataset/`. **IMPORTANT: GPU is required**.
9. `best_res/fit_lama_fttransformer.ipynb` - fits LightAutoML AutoInt. Saves model file, oof predictions, model params and test predictions in `data/model/lamann_fttransformer_8050_full_dataset/`. **IMPORTANT: GPU is required**.
10. `best_res/create_stack_df.ipynb` - Adds out of fold predictions as features to the dataset. Saves stacked datasets in `data/` folder.
11. `best_res/fit_lama_stack.ipynb` - fits + tunes stack LightAutoML on a time series cross-validation. Saves model file, model params and test predictions in `data/model/lama_stack_time_series/`.
12. `blend.ipynb` - blends the predictions of `lamau_81425_full_dataset` and `lama_stack_time_series` models and produces the final submission.

## Solution explanation
Firstly, we explored the dataset. Data has appeared to be pretty clean and well-prepared for modelling even in a raw format. Due to the lack of the information about features (they are depersonalized), feature engineering was impossible. The only thing we have done - found categorical columns in the data (the ones that contain less than *150* unique values). Finally, basic features processing (standard scaling + ordinal encoding) was applied.

## Hardware
* CPU server:
    * CPU: *Intel Xeon (Cascadelake) (16) @ 2.992GHz*
    * Memory: *16 GB*
    * Python: *Python 3.10.12*
* GPU server:
    * CPU: *Intel Xeon Gold 6240R (6) @ 2.400GHz*
    * GPU: *NVIDIA GeForce RTX 2080 Ti Rev. A*
    * Memory: *16 GB*
    * Python: *Python 3.10.0*