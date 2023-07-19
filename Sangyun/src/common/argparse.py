from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    model_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model"}
    )
    data_dir: Optional[str] = field(
        default="../data", metadata={"help": "local dataset stored location"},
    )
    train_file: Optional[str] = field(
        default="train.csv", metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    test_file: Optional[str] = field(
        default="test.csv", metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."}
    )
    num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    output_dir: str = field(
        default="../models",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_test: bool = field(default=False, metadata={"help": "Whether to run test on the test set."})

    batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
