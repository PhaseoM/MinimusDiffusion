import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import unis
from utils.data import roc_eval, roc_eval_multicls, loss_eval


def load_pkl(dump_file: Path):
    obj = []
    with open(dump_file, "rb") as f:
        obj.append(pickle.load(f))
        obj.append(pickle.load(f))
        obj.append(pickle.load(f))
        obj.append(pickle.load(f))
        obj.append(pickle.load(f))
        obj.append(pickle.load(f))
    return obj


def load_pkl_all(dump_folder: Path):
    obj_list = []
    for dump_file in dump_folder.iterdir():
        obj = []
        with open(dump_file, "rb") as f:
            obj.append(pickle.load(f))
            obj.append(pickle.load(f))
            obj.append(pickle.load(f))
            obj.append(pickle.load(f))
            obj.append(pickle.load(f))
            obj.append(pickle.load(f))
        obj_list.append((str(dump_file.stem), obj))
    return obj_list


# - - is train
# --- is test
def data_process(conf):
    fig_arr = []
    model_name = Path(conf.MODEL.NAME)
    dump_folder = Path(conf.PATH.DATA_DUMP)
    dump_file = dump_folder / Path(str(model_name) + ".pkl")
    eval_path = Path(conf.PATH.DATA_EVAL)
    eval_folder = eval_path / model_name

    dump_lists = load_pkl(dump_file)

    train_loss_list = [dump_lists[0]]
    test_loss_list = [dump_lists[1]]
    train_iou_list = [dump_lists[2]]
    test_iou_list = [dump_lists[3]]
    train_dice_list = [dump_lists[4]]
    test_dice_list = [dump_lists[5]]

    labels = [str(model_name)]
    fig_arr.append(loss_eval(train_loss_list, test_loss_list, "Loss-" + str(model_name), labels=labels))
    fig_arr.append(loss_eval(train_iou_list, test_iou_list, "IoU-" + str(model_name), labels=labels, ylabel="IoU"))
    fig_arr.append(loss_eval(train_dice_list, test_dice_list, "Dice-" + str(model_name), labels=labels, ylabel="Dice"))

    eval_folder.mkdir(parents=True, exist_ok=True)
    for fig in fig_arr:
        title = fig.gca().get_title()
        fig.savefig(eval_folder / f"{title}.png", dpi=150, bbox_inches="tight")
