#!/usr/bin/env python3
"""
TypeScript SLM CLI - Unified command-line interface for training and managing TypeScript SLM models
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

app = typer.Typer(
    name="slm",
    help="TypeScript Small Language Model - Training and Management CLI",
    add_completion=True,
)

console = Console()

# Import command modules
from scripts import data_collection
from scripts import data_preprocessing
from scripts import evaluation
from scripts import upload_to_hf
from scripts import training


@app.command()
def collect(
    min_stars: int = typer.Option(1000, "--min-stars", "-s", help="Minimum GitHub stars for repositories"),
    repo_limit: int = typer.Option(5, "--repo-limit", "-r", help="Number of repositories to clone per framework"),
    so_limit: int = typer.Option(50, "--so-limit", help="Number of StackOverflow questions to fetch"),
):
    """
    Collect training data from GitHub and StackOverflow.

    This command will:
    - Search for TypeScript repositories on GitHub
    - Clone and extract TypeScript files
    - Fetch Q&A pairs from StackOverflow
    """
    console.print("\n[bold blue]Starting data collection...[/bold blue]\n")

    try:
        data_collection.main()
        console.print("\n[bold green]✓ Data collection completed successfully![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during data collection: {e}[/bold red]\n")
        raise typer.Exit(code=1)


@app.command()
def preprocess(
    input_dir: Path = typer.Option(Path("data/raw"), "--input", "-i", help="Input directory with raw data"),
    output_dir: Path = typer.Option(Path("data/processed"), "--output", "-o", help="Output directory for processed data"),
):
    """
    Preprocess and clean collected data.

    This command will:
    - Clean and deduplicate code samples
    - Process StackOverflow Q&A pairs
    - Split into train/validation sets
    - Save as JSONL format
    """
    console.print("\n[bold blue]Starting data preprocessing...[/bold blue]\n")

    try:
        data_preprocessing.main()
        console.print("\n[bold green]✓ Data preprocessing completed successfully![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during preprocessing: {e}[/bold red]\n")
        raise typer.Exit(code=1)


@app.command()
def train(
    model_name: str = typer.Option(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "--model",
        "-m",
        help="Base model from Hugging Face"
    ),
    data_path: Path = typer.Option(
        Path("data/processed/train.jsonl"),
        "--data",
        "-d",
        help="Path to training data (JSONL file)"
    ),
    output_dir: Path = typer.Option(
        Path("./models/typescript-slm-1.5b"),
        "--output",
        "-o",
        help="Output directory for trained model"
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Training batch size per device (optimized for M4 24GB)"),
    gradient_accumulation: int = typer.Option(
        2,
        "--grad-accum",
        "-g",
        help="Gradient accumulation steps (optimized for M4)"
    ),
    learning_rate: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    max_seq_length: int = typer.Option(1024, "--max-length", help="Maximum sequence length"),
    lora_r: int = typer.Option(64, "--lora-r", help="LoRA rank"),
    lora_alpha: int = typer.Option(16, "--lora-alpha", help="LoRA alpha (scaling factor)"),
    save_steps: int = typer.Option(500, "--save-steps", help="Save checkpoint every N steps"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint path"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit dataset to N samples (for testing)"),
):
    """
    Train the TypeScript SLM model locally (optimized for Mac M4).

    This command will:
    - Load the base model from Hugging Face
    - Apply LoRA for parameter-efficient fine-tuning
    - Train on your TypeScript dataset
    - Save checkpoints and final model

    Optimized for Mac M4 with 24GB RAM using MPS acceleration.
    """
    console.print("\n[bold blue]Starting model training...[/bold blue]\n")
    console.print(f"[cyan]Model:[/cyan] {model_name}")
    console.print(f"[cyan]Data:[/cyan] {data_path}")
    console.print(f"[cyan]Output:[/cyan] {output_dir}")
    console.print(f"[cyan]Epochs:[/cyan] {epochs}")
    console.print(f"[cyan]Batch size:[/cyan] {batch_size}")
    console.print(f"[cyan]Effective batch size:[/cyan] {batch_size * gradient_accumulation}\n")

    try:
        training.train(
            model_name=model_name,
            data_path=str(data_path),
            output_dir=str(output_dir),
            num_epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            save_steps=save_steps,
            resume_from_checkpoint=str(resume) if resume else None,
            max_samples=max_samples,
        )
        console.print("\n[bold green]✓ Training completed successfully![/bold green]\n")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠ Training interrupted by user[/bold yellow]\n")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during training: {e}[/bold red]\n")
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    adapter_path: Optional[Path] = typer.Option(
        Path("./models/typescript-slm-1.5b"),
        "--adapter",
        "-a",
        help="Path to trained adapter"
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "--model",
        "-m",
        help="Base model name"
    ),
):
    """
    Evaluate the trained model with test prompts.

    This command will:
    - Load the base model and trained adapter
    - Generate code from test prompts
    - Display results and generation speed
    """
    console.print("\n[bold blue]Starting model evaluation...[/bold blue]\n")

    try:
        evaluation.main()
        console.print("\n[bold green]✓ Evaluation completed successfully![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during evaluation: {e}[/bold red]\n")
        raise typer.Exit(code=1)


@app.command()
def upload(
    model_path: Path = typer.Option(
        Path("./models/typescript-slm-1.5b"),
        "--model",
        "-m",
        help="Path to model directory"
    ),
    username: str = typer.Option(..., "--username", "-u", help="Hugging Face username"),
    model_name: str = typer.Option(
        "typescript-slm-1.5b",
        "--name",
        "-n",
        help="Model name on Hugging Face"
    ),
):
    """
    Upload trained model to Hugging Face Hub.

    This command will:
    - Create a repository on Hugging Face
    - Upload model files and configuration
    - Make the model available for sharing

    Requires HF_TOKEN environment variable to be set.
    """
    console.print("\n[bold blue]Uploading model to Hugging Face...[/bold blue]\n")
    console.print(f"[cyan]Repository:[/cyan] {username}/{model_name}\n")

    try:
        upload_to_hf.upload_model(
            model_path=str(model_path),
            username=username,
            model_name=model_name
        )
        console.print("\n[bold green]✓ Upload completed successfully![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during upload: {e}[/bold red]\n")
        raise typer.Exit(code=1)


@app.command()
def info():
    """
    Display information about the SLM project and environment.
    """
    import torch
    from rich.table import Table

    console.print("\n[bold]TypeScript SLM - Project Information[/bold]\n")

    # System info table
    table = Table(title="System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("PyTorch Version", torch.__version__)

    # Device detection
    if torch.backends.mps.is_available():
        device = "Apple Metal (MPS)"
    elif torch.cuda.is_available():
        device = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = "CPU"
    table.add_row("Device", device)

    # Check for data
    data_dir = Path("data/processed")
    if data_dir.exists():
        train_file = data_dir / "train.jsonl"
        if train_file.exists():
            with open(train_file) as f:
                num_samples = sum(1 for _ in f)
            table.add_row("Training Samples", str(num_samples))
        else:
            table.add_row("Training Samples", "Not found")
    else:
        table.add_row("Training Samples", "Not found")

    # Check for models
    models_dir = Path("models")
    if models_dir.exists():
        models = list(models_dir.iterdir())
        table.add_row("Trained Models", str(len(models)))
    else:
        table.add_row("Trained Models", "0")

    console.print(table)
    console.print()


@app.command()
def pipeline(
    collect_data: bool = typer.Option(None, "--collect/--no-collect", help="Run data collection (auto-detects if None)"),
    preprocess_data: bool = typer.Option(None, "--preprocess/--no-preprocess", help="Run preprocessing (auto-detects if None)"),
    train_model: bool = typer.Option(True, "--train/--no-train", help="Run training"),
    evaluate_model: bool = typer.Option(True, "--evaluate/--no-evaluate", help="Run evaluation"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit training dataset to N samples (for testing)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download and re-preprocess even if data exists"),
):
    """
    Run the complete pipeline from data collection to training.

    This command orchestrates the full workflow:
    1. Collect data from GitHub and StackOverflow (skipped if data exists)
    2. Preprocess and clean the data (skipped if processed data exists)
    3. Train the model
    4. Evaluate the results

    The pipeline is smart - it automatically detects existing data and skips
    unnecessary steps. Use --force to re-download and re-preprocess.
    """
    console.print("\n[bold magenta]Running complete SLM pipeline...[/bold magenta]\n")

    # Smart detection of existing data
    raw_data_dir = Path("data/raw")
    processed_data_dir = Path("data/processed")
    train_file = processed_data_dir / "train.jsonl"

    # Auto-detect what needs to be done
    if collect_data is None:
        if force:
            collect_data = True
            console.print("[cyan]ℹ --force flag set: Will re-download data[/cyan]")
        elif raw_data_dir.exists() and any(raw_data_dir.iterdir()):
            collect_data = False
            console.print("[cyan]ℹ Raw data exists, skipping collection (use --force to re-download)[/cyan]")
        else:
            collect_data = True

    if preprocess_data is None:
        if force:
            preprocess_data = True
            console.print("[cyan]ℹ --force flag set: Will re-preprocess data[/cyan]")
        elif train_file.exists():
            # Check file size to ensure it's not empty
            if train_file.stat().st_size > 1000:
                preprocess_data = False
                with open(train_file) as f:
                    num_samples = sum(1 for _ in f)
                console.print(f"[cyan]ℹ Processed data exists ({num_samples} samples), skipping preprocessing (use --force to re-process)[/cyan]")
            else:
                preprocess_data = True
        else:
            preprocess_data = True

    console.print()

    try:
        if collect_data:
            console.print("[bold blue]Step 1/4: Data Collection[/bold blue]")
            data_collection.main()
            console.print("[bold green]✓ Collection complete[/bold green]\n")
        else:
            console.print("[bold yellow]⊘ Step 1/4: Data Collection (skipped)[/bold yellow]\n")

        if preprocess_data:
            console.print("[bold blue]Step 2/4: Data Preprocessing[/bold blue]")
            data_preprocessing.main()
            console.print("[bold green]✓ Preprocessing complete[/bold green]\n")
        else:
            console.print("[bold yellow]⊘ Step 2/4: Data Preprocessing (skipped)[/bold yellow]\n")

        if train_model:
            console.print("[bold blue]Step 3/4: Model Training[/bold blue]")
            training.train(max_samples=max_samples)
            console.print("[bold green]✓ Training complete[/bold green]\n")
        else:
            console.print("[bold yellow]⊘ Step 3/4: Model Training (skipped)[/bold yellow]\n")

        if evaluate_model:
            console.print("[bold blue]Step 4/4: Model Evaluation[/bold blue]")
            evaluation.main()
            console.print("[bold green]✓ Evaluation complete[/bold green]\n")
        else:
            console.print("[bold yellow]⊘ Step 4/4: Model Evaluation (skipped)[/bold yellow]\n")

        console.print("[bold green]✓ Pipeline completed successfully![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed: {e}[/bold red]\n")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
