import matplotlib.figure
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import utils.unis as unis
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Any, Sequence


# the params that kwargs can reveive:
# xlabel, ylabel
# train&test labels
def loss_eval(
    train_lists: list[list[float]] | None = None,
    test_lists: list[list[float]] | None = None,
    title: str | None = None,
    **kwargs,
) -> Figure:
    if not train_lists:
        train_lists = None
    if not test_lists:
        test_lists = None
    if train_lists is None and test_lists is None:
        raise Exception("Train&Test Lists are all empty.")
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()
    ax.set_title(title if title is not None else "loss eval")

    xlabel = kwargs.get("xlabel", "epoch")
    ylabel = kwargs.get("ylabel", "loss")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    colors = [
        "red",
        "C0",  # lightblue
        "orange",
        "skyblue",
        "green",
        "blue",
        "pink",
        "purple",
        "brown",
        "gray",
        "black",
    ]

    if len(train_lists) > len(colors):
        raise Exception("Too many lines in one figure")

    labels = kwargs.get("labels", None)
    if train_lists is None:
        for i, test_list in enumerate(test_lists):
            size_n = len(test_list)
            indices = [_ for _ in range(size_n)]
            if labels is None:
                ax.plot(indices, test_list, c=colors[i])
            else:
                ax.plot(indices, test_list, c=colors[i], label=labels[i])
    elif test_lists is None:
        for i, train_list in enumerate(train_lists):
            size_n = len(train_list)
            indices = [_ for _ in range(size_n)]
            if labels is None:
                ax.plot(indices, train_list, c=colors[i])
            else:
                ax.plot(indices, train_list, c=colors[i], label=labels[i])
    else:
        for i, (train_list, test_list) in enumerate(list(zip(train_lists, test_lists))):
            size_n = len(train_list)
            indices = [_ for _ in range(size_n)]
            ax.plot(indices, train_list, c=colors[0], label="train")
            if labels is None:
                ax.plot(indices, test_list, c=colors[1])
            else:
                ax.plot(indices, test_list, c=colors[1], label="test")
    if labels is not None:
        ax.legend(fontsize=10)
    return fig


class RocType:
    def __init__(
        self,
        label: str | None = None,
        data: Sequence[Sequence[Any]] | None = None,
    ):
        self.label = label
        self.data = data

    def __iter__(self):
        return iter((self.label, self.data))


def roc_eval(
    roclist: Sequence[RocType | tuple[str | None, Sequence[Sequence[Any]]]],
    title: str | None = None,
    logger=None,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots()
    ax.set_title(title if title is not None else "roc eval")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    for label, out_tuples in roclist:
        outs = np.array([int(np.asarray(out_tuple[0]).ravel()[0]) for out_tuple in out_tuples])
        probs = np.array([float(np.asarray(out_tuple[-1]).ravel()[0]) for out_tuple in out_tuples])

        if not np.isin(outs, [0, 1]).all():
            raise ValueError("roc_eval expects binary labels in the first field.")

        P = (outs == 1).sum()
        N = (outs == 0).sum()

        if P == 0 or N == 0:
            print(f"ROC [{label}]: can't plot, Positive or Negative samples are 0")
            continue

        order_indices = np.argsort(-probs, kind="mergesort")
        outs_sorted = outs[order_indices]
        probs_sorted = probs[order_indices]

        tps = np.cumsum(outs_sorted == 1)
        fps = np.cumsum(outs_sorted == 0)

        distinct_indices = np.r_[np.where(np.diff(probs_sorted))[0], probs.size - 1]
        x = np.r_[0.0, fps[distinct_indices] / N]
        y = np.r_[0.0, tps[distinct_indices] / P]

        auc = unis.trapz(x, y)
        if logger is None:
            print(f"[{label}] auc: {auc:.4f}")
        else:
            logger.info(f"[{label}] auc: {auc:.4f}")
        ax.plot(x, y, label=f"{label} (AUC={auc:.2f})", lw=1)

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="-.")
    ax.legend(loc="lower right", fontsize=7)
    return fig


def roc_eval_multicls(
    roclist: Sequence[RocType | tuple[str | None, Sequence[Sequence[Any]]]],
    title: str | None = None,
    class_labels: Sequence[str] | None = None,
    class_count: int | None = None,
    logger=None,
) -> matplotlib.figure.Figure:
    binary_roclist: list[RocType] = []

    for model_label, out_tuples in roclist:
        if not out_tuples:
            continue

        y_true = np.array([int(np.asarray(out_tuple[0]).ravel()[0]) for out_tuple in out_tuples])
        probs = np.array([np.asarray(out_tuple[-1], dtype=float) for out_tuple in out_tuples])

        if probs.ndim != 2:
            raise ValueError("roc_eval_multicls expects prob vectors in the last field.")

        n_classes = class_count if class_count is not None else probs.shape[1]
        if probs.shape[1] < n_classes:
            raise ValueError("prob vector length is smaller than class_count.")
        if class_labels is not None and len(class_labels) < n_classes:
            raise ValueError("class_labels length is smaller than class_count.")

        for class_index in range(n_classes):
            class_name = class_labels[class_index] if class_labels is not None else f"class-{class_index}"
            label_prefix = model_label if model_label is not None else "model"
            binary_data = [
                (int(target == class_index), None, float(prob[class_index])) for target, prob in zip(y_true, probs)
            ]
            binary_roclist.append(RocType(f"{label_prefix}-{class_name}", binary_data))

    return roc_eval(
        binary_roclist,
        title=title if title is not None else "multiclass roc eval",
        logger=logger,
    )
