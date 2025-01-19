import numpy as np
from sklearn.calibration import LabelEncoder


def preprocess_data(data, target_col, discrete_cols, past_steps=240, future_steps=240):
    # 编码离散型变量
    label_encoders = {col: LabelEncoder() for col in discrete_cols}
    for col in discrete_cols:
        data[col] = label_encoders[col].fit_transform(data[col])

    # 构建序列
    X_cont, X_cat, y = [], [], []
    for i in range(len(data) - past_steps - future_steps):
        X_cont.append(
            data.iloc[i : i + past_steps].drop([*discrete_cols, target_col], axis=1).values
        )
        X_cat.append(data.iloc[i : i + past_steps][discrete_cols].values)
        y.append(data.iloc[i + past_steps : i + past_steps + future_steps][target_col].values)

    return np.array(X_cont), np.array(X_cat), np.array(y)
