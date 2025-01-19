import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def preprocess_data(data, target_col, discrete_cols, past_steps=240, future_steps=240):
    """
    数据预处理: 包括离散特征编码、时间序列切片。
    """
    label_encoders = {col: LabelEncoder() for col in discrete_cols}
    for col in discrete_cols:
        data[col] = label_encoders[col].fit_transform(data[col])

    scaler = MinMaxScaler()
    continuous_cols = [col for col in data.columns if col not in {*discrete_cols, target_col}]
    data[[*continuous_cols, target_col]] = scaler.fit_transform(
        data[[*continuous_cols, target_col]]
    )

    X_cont, X_cat, y = [], [], []
    for i in range(len(data) - past_steps - future_steps):
        X_cont.append(data.iloc[i : i + past_steps][continuous_cols].values)
        X_cat.append(data.iloc[i : i + past_steps][discrete_cols].values)
        y.append(data.iloc[i + past_steps : i + past_steps + future_steps][target_col].values)

    return np.array(X_cont), np.array(X_cat), np.array(y), scaler
