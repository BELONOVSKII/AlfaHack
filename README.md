This repository contains the code for the first stage of the **AlfaHack AutoML hackaton**.

Team: ***DDDrkBBB***.

Members: [Peter Belonovskiy](https://github.com/BELONOVSKII), [Kristina Galuzina](https://github.com/galuzina-k), [Timofey Lashukov](https://github.com/M1croZavr)

<p align="center"><img src="assets/sber_kot.gif" width="200" height="200" /></p>

## General info
* Track: *Отток юридических лиц из расчетно-кассового обслуживания.*
* Leaderboard score: $81.7015322279558$
* Leaderboard position: **3**.


## TL;DR 
Tune + fit 7 models on selected features. Each model is a mean blend on 5 stratified folds. Blend the best model with the model staked on the selected features and oof predictions.

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
10. `best_res/create_stack_df.ipynb` - Adds out-of-fold predictions as features to the dataset. Saves stacked datasets in `data/` folder.
11. `best_res/fit_lama_stack.ipynb` - fits + tunes stack LightAutoML on a time series cross-validation. Saves model file, model params and test predictions in `data/model/lama_stack_time_series/`.
12. `blend.ipynb` - blends the predictions of `lamau_81425_full_dataset` and `lama_stack_time_series` models and produces the final submission.

## Solution explanation
Firstly, we have explored the dataset. Data has appeared to be pretty clean and well-prepared for modeling, even in a raw format.
Due to the lack of information about features (they are depersonalized), feature engineering is impossible. The only thing we have done is found categorical columns in the data (the ones that contain less than *150* unique values).
Finally, basic feature processing (standard scaling + ordinal encoding) was applied.

Next, we have understood that data contains too many features, and some of them are useless.
Thus, we have decided to perform feature selection. This evidently speeds up training and results in high scores. As for the feature selection algorithm, we have chosen [Catboost Shapley values](https://catboost.ai/en/docs/concepts/shap-values?ysclid=m3fn2ebwpf967485278) feature selection.
From our point of view, this is the most unbiased way to find really important features.

Then we have proceeded to the model training. From the very beginning, we have decided that we will apply stacking due to its dominance in tabular tasks.
For stacking, we need to train several base models with different structures, save their out-of-fold predictions, and then train the final model on the default features + out-of-fold predictions.

For base models we have chosen: (out-of-fold scores are shown in bold)
* CatBoost **0.8114**
* LightGBM **0.8122**
* XGBoost **0.8132**
* LightAutoML **0.81298**
* LightAutoMLUtilized **0.81425**
* LightAutoML AutoInt (*Tabular neural network*) **0.8053**
* LightAutoML FtTransformer (*Tabular neural network*) **0.805**

For the stacking model:
* LightAutoML

Wrapped implementation of all these models, that significantly eases the workflow, has been taken from the open-source [automl](https://github.com/dertty/automl) package. This package allows to automatically tune parameters for each model and then fit the model with the best parameters on the cross-validated folds. Such fitting strategy reduces the variance of predictions and allows for out-of-fold predictions. Training of all the models except for the *LightAutoML AutoInt* and *LightAutoML FtTransformer* has been performed on the <ins>CPU server</ins>. Tabular neural networks have been trained on the <ins>GPU server</ins>
We have tried two cross-validation strategies: stratified and time-series. For the base models, stratified cross-validation has shown much better results, while for the stacking model, time-series cross-validation has made the deal.

Each model has been tuned with the timeout of 1 hour (2 hours for LightAutoMLUtilized). The full train dataset has been used because we have observed that decreasing the train size significantly worsens the results.
The best model by out-of-fold (*LightAutoMLUtilized*), was also the best on the leaderboard $\approx 0.8164$.
Obtained out-of-fold predictions of all models have been concatenated to the train/test datasets.

Then, stacking has been performed by fitting LightAutoML on the enriched training dataset. As mentioned earlier, time-series cross validation has shown better results. The score $\approx 0.8168$ on the leaderboard has been achieved.


Finally, we have decided to blend the predictions of the best base model (*LightAutoMLUtilized*) and the stacking model with weights $[0.15, 0.85]$ respectively. The weights were chosen via the leaderboard.

Final leaderboard score: $\mathbf{81.7015322279558}$


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
