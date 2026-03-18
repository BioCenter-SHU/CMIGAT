
from train_test_GCN import train_test  # Uncomment if using GCN instead of GAT

def run_experiment(
        run_label,
        data_folder,
        view_list,
        num_class,
        optimizer_cfg,
        schedule_cfg,
        testonly=True):
    print(f"\n===== Running {run_label} =====")
    train_test(
        data_folder,
        view_list,
        num_class,
        optimizer_cfg["lr_pretrain"],
        optimizer_cfg["lr_main"],
        testonly,
        schedule_cfg["num_epoch_pretrain"],
        schedule_cfg["num_epoch"],
    )

if __name__ == "__main__":
    data_folder = 'BRCA'
    view_list = [1, 2, 3]
    testonly = False

    schedule_cfg = {
        "num_epoch_pretrain": 300,
        "num_epoch": 1500,
    }
    optimizer_cfg = {"lr_pretrain": 5e-3, "lr_main": 1e-4}

    if data_folder == 'BRCA':
        num_class = 5

    run_experiment(
        "Baseline",
        data_folder,
        view_list,
        num_class,
        optimizer_cfg,
        schedule_cfg,
        testonly=testonly,
    )
