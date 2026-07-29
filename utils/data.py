import os

from datasets import load_dataset, Dataset, concatenate_datasets
from config.train_sft_model import DatasetConfig


def _load_one(name_or_path, split):
    """Load a HF hub dataset, or a local JSONL file / directory of
    <split>.jsonl files (used by the OSIM-filtered OpenR1 build)."""
    if name_or_path.endswith((".jsonl", ".json")):
        return load_dataset("json", data_files=name_or_path, split="train")
    local = os.path.join(name_or_path, f"{split}.jsonl")
    if os.path.isdir(name_or_path) and os.path.exists(local):
        return load_dataset("json", data_files=local, split="train")
    return load_dataset(name_or_path, split=split)


def load_datasets(cfg: DatasetConfig, seed: int) -> Dataset:
    """
    Load training and validation datasets based on the configuration.
    Args:
        cfg (DatasetConfig): Configuration containing dataset information.
        seed (int): Random seed for reproducibility.
    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
        
    If no datasets are provided or an error occurs, returns None for the respective dataset.
    """
    train_datasets, val_datasets = [], []

    try:
        for dataset in cfg.train_datasets:
            dataset = _load_one(dataset.name_or_path, dataset.split)
            train_datasets.append(dataset)

        # We sample based on max_examples and ratios.
        if cfg.max_train_examples is not None:
            ratios = [dataset.ratio for dataset in cfg.train_datasets]
            total_ratio = sum(ratios)
            num_samples = cfg.max_train_examples

            # Sample based on ratios
            if num_samples == -1:
                for i, dataset in enumerate(train_datasets):
                    train_datasets[i] = dataset.shuffle(seed=seed)
            else:
                samples_per_dataset = [
                    int(num_samples * ratio / total_ratio) for ratio in ratios
                ]
                for i, dataset in enumerate(train_datasets):
                    train_datasets[i] = dataset.shuffle(seed=seed).select(
                        range(samples_per_dataset[i])
                    )
        # Concatenate the datasets
        train_datasets = concatenate_datasets(train_datasets)
        train_datasets = train_datasets.shuffle(seed=seed)
    except Exception as e:
        train_datasets = None
        print("No training datasets provided or an error occurred while loading them.")
        print(e)

    try:
        for dataset in cfg.eval_datasets:
            dataset = _load_one(dataset.name_or_path, dataset.split)
            val_datasets.append(dataset)

        if len(val_datasets) == 0:
            return train_datasets, None

        if cfg.max_val_examples is not None:
            ratios = [dataset.ratio for dataset in cfg.eval_datasets]
            total_ratio = sum(ratios)
            num_samples = cfg.max_val_examples

            # Sample based on ratios
            if num_samples == -1:
                for i, dataset in enumerate(val_datasets):
                    val_datasets[i] = dataset.shuffle(seed=seed)
            else:
                samples_per_dataset = [
                    int(num_samples * ratio / total_ratio) for ratio in ratios
                ]
                for i, dataset in enumerate(val_datasets):
                    val_datasets[i] = dataset.shuffle(seed=seed).select(
                        range(samples_per_dataset[i])
                    )

        val_datasets = concatenate_datasets(val_datasets)
        val_datasets = val_datasets.shuffle(seed=seed)
    except Exception as e:
        val_datasets = None
        print(
            "No validation datasets provided or an error occurred while loading them."
        )
        print(e)

    return train_datasets, val_datasets
