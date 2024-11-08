import sys; sys.path.append("../../../../automl/")

from pathlib import Path
import yaml
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.automl.model.lama import TabularLamaUtilized
from src.automl.loggers import configure_root_logger
from src.automl.constants import create_ml_data_dir
from src.automl.model.metrics import RocAuc

create_ml_data_dir()
configure_root_logger()

RANDOM_SEED = 77
DATA_PATH = Path("../../../../data/")
CONFIG_PATH = Path("../../../../configs/config.yaml")
N_JOBS = 16

with CONFIG_PATH.open() as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
cfg["stack_features"] = ["lamau_814_full_dataset", "lamann_autoint_8053_full_dataset"]
    
df_train = pd.read_parquet(DATA_PATH / "train_preproc_oof.parquet")

df_train["target"].value_counts(normalize=True)

cat_columns = df_train.drop(columns=["target", "id"]).select_dtypes(int).columns.values.tolist()

X_train, y_train = df_train[cfg["selected_features"] + cfg["stack_features"] + cat_columns], df_train["target"]


metric = RocAuc()

model = TabularLamaUtilized(n_jobs=16, task="classification")
model.tune(X_train, y_train, metric, timeout=60 * 60, categorical_features=cat_columns)
oof = model.fit(X_train, y_train, categorical_features=cat_columns)

print(metric(y_train, oof))


MODEL_NAME = "lamau_stack_##_full_dataset"
MODEL_DIR = Path(f"../../../../data/models/{MODEL_NAME}")
MODEL_DIR.mkdir(exist_ok=True)

res = pd.DataFrame()
res[MODEL_NAME] = oof[:, 1]
res.to_csv(MODEL_DIR / "oof.csv", index=False)
joblib.dump(model, MODEL_DIR / f"{MODEL_NAME}.joblib")

with (MODEL_DIR / "params.yaml").open("w") as f:
    yaml.dump(model.params, f)

with (MODEL_DIR / "score.txt").open("w") as f:
    print("OOF:", metric(y_train, oof), file=f)
    
test = pd.read_parquet(DATA_PATH / "test_preproc_oof.parquet")
test["target"] = model.predict(test[cfg["selected_features"] + cfg["stack_features"] + cat_columns])[:, 1]
test[['id', 'target']].to_csv(MODEL_DIR / f'{MODEL_NAME}.csv', index=False)

# test = pd.read_parquet(DATA_PATH / "test_preproc_oof.parquet")
# test["target"] = model.predict(test[cfg["selected_features"] + cfg["stack_features"] + cat_columns])[:, 1]
# test[['id', 'target']].to_csv(f'{MODEL_NAME}.csv', index=False)