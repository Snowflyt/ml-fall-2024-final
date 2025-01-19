from ml_fall_2024_final.dataset import BikeDataset, read_test_data, read_train_data
from ml_fall_2024_final.transformer.model import TransformerTimeSeries
from ml_fall_2024_final.transformer.plotting import plot_window_prediction
from ml_fall_2024_final.transformer.trainer import run_experiment
from ml_fall_2024_final.utils import save_fig, save_model

RUN_TIMES = 5


def start():
    """Launched with `poetry run transformer` at root level"""

    train_data = read_train_data()
    test_data = read_test_data()

    train_dataset_short = BikeDataset(train_data, output_window="short")
    test_dataset_short = BikeDataset(test_data, output_window="short")

    train_dataset_long = BikeDataset(train_data, output_window="long")
    test_dataset_long = BikeDataset(test_data, output_window="long")

    #
    # 短期预测 (96 小时)
    #

    print("短期预测 (96 小时)")

    # 训练
    model_short = TransformerTimeSeries(output_window="short")
    preds_short, truth_short, _mse_short, _mae_short = run_experiment(
        model=model_short,
        train_dataset=train_dataset_short,
        test_dataset=test_dataset_short,
        run_times=RUN_TIMES,
    )
    save_model(model_short, "transformer_short")

    # 绘制预测结果 (最后一个滑动窗口)
    last_window_preds_short = tuple(pred[-1] for pred in preds_short)
    last_window_truth_short = truth_short[-1]

    # print_window_mse_mae(last_window_preds_short, last_window_truth_short)
    fig = plot_window_prediction(
        last_window_preds_short,
        last_window_truth_short,
        title="Short-term Prediction (96 Hours)",
        figsize=(8, 6),
    )
    save_fig(fig, "transformer_short")
    fig = plot_window_prediction(
        last_window_preds_short[0],
        last_window_truth_short,
        title="Short-term Prediction (96 Hours)",
        figsize=(8, 6),
    )
    save_fig(fig, "transformer_short_single_model")

    #
    # 长期预测 (240 小时)
    #

    print("\n长期预测 (240 小时)")

    # 训练
    model_long = TransformerTimeSeries(output_window="long")
    preds_long, truth_long, _mse_long, _mae_long = run_experiment(
        model=model_long,
        train_dataset=train_dataset_long,
        test_dataset=test_dataset_long,
        run_times=RUN_TIMES,
    )
    save_model(model_long, "transformer_long")

    # 绘制预测结果 (最后一个滑动窗口)
    last_window_preds_long = tuple(pred[-1] for pred in preds_long)
    last_window_truth_long = truth_long[-1]

    # print_window_mse_mae(last_window_preds_long, last_window_truth_long)
    fig = plot_window_prediction(
        last_window_preds_long,
        last_window_truth_long,
        title="Long-term Prediction (240 Hours)",
        figsize=(8, 6),
    )
    save_fig(fig, "transformer_long")
    fig = plot_window_prediction(
        last_window_preds_long[0],
        last_window_truth_long,
        title="Long-term Prediction (240 Hours)",
        figsize=(8, 6),
    )
    save_fig(fig, "transformer_long_single_model")
