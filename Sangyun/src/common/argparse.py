from dataclasses import dataclass, field
from typing import Optional



@dataclass
class TrainingArguments:
    # Task Configuration
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_test: bool = field(default=False, metadata={"help": "Whether to run test on the test set."})
    output_dir: str = field(
        default="../models",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    # Data Configuration
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

    # Model Configuration
    architecture: str = field(
        default=None, 
        metadata={"help": "Model architecture to be used for training."}
    )
    encoder_name: str = field(
        default=None, 
        metadata={"help": "Model backbone encoder name"}
    )
    encoder_weights: str = field(
        default=None, 
        metadata={"help": "Model backbone pretrained encoder weights"}
    )
    classes: int = field(
        default=None, 
        metadata={"help": "Model number of classes"}
    )
    activation: str = field(
        default=None, 
        metadata={"help": "Model activation function"}
    )

    # Training Configuration
    epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    lr: float = field(default=5e-5, metadata={"help": "The initial learning rate for optimizer."})
    batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
    verbose: bool = field(default=True, metadata={"help": "Training verbosity"})
    
    losses: str = field(
        default=None, 
        metadata={"help": "Model loss function"}
    )
    metrics: str = field(
        default=None, 
        metadata={"help": "Model metric function"}
    )
    optimizer: str = field(
        default=None, 
        metadata={"help": "Model optimizer function"}
    )
    scheduler: str = field(
        default=None, 
        metadata={"help": "Model scheduler function"}
    )
    train_transform: str = field(
        default='train_transform_1', 
        metadata={"help": "Model training augmentation function"}
    )
    test_transform: str = field(
        default='test_transform_1', 
        metadata={"help": "Model test augmentation function"}
    )

    # Other Configuration
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
