"""Command-line interface for privacy-preserving agent finetuner."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
import yaml

from .core import PrivateTrainer, PrivacyConfig, ContextGuard, RedactionStrategy
from .utils.monitoring import PrivacyBudgetMonitor

app = typer.Typer(
    name="privacy-finetuner",
    help="Privacy-Preserving Agent Finetuner CLI"
)
console = Console()


@app.command()
def train(
    model_name: str = typer.Argument(..., help="HuggingFace model name"),
    dataset: Path = typer.Argument(..., help="Training dataset path"),
    config: Optional[Path] = typer.Option(None, help="Privacy configuration file"),
    epsilon: float = typer.Option(1.0, help="Privacy budget (epsilon)"),
    delta: float = typer.Option(1e-5, help="Privacy parameter (delta)"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(8, help="Training batch size"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate"),
):
    """Train a model with differential privacy guarantees."""
    console.print(f"🔒 Starting private training for {model_name}", style="bold green")
    
    # Load configuration
    if config:
        privacy_config = PrivacyConfig.from_yaml(config)
        console.print(f"📝 Loaded config from {config}")
    else:
        privacy_config = PrivacyConfig(epsilon=epsilon, delta=delta)
        console.print(f"⚙️  Using default config: ε={epsilon}, δ={delta}")
    
    # Initialize trainer
    trainer = PrivateTrainer(
        model_name=model_name,
        privacy_config=privacy_config
    )
    
    # Start training
    try:
        result = trainer.train(
            dataset=str(dataset),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        console.print("✅ Training completed successfully!", style="bold green")
        
        # Display privacy report
        privacy_report = trainer.get_privacy_report()
        table = Table(title="Privacy Budget Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in privacy_report.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Training failed: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def protect(
    text: str = typer.Argument(..., help="Text to protect"),
    strategies: str = typer.Option("pii_removal", help="Comma-separated redaction strategies"),
    sensitivity: str = typer.Option("medium", help="Sensitivity level (low/medium/high)"),
):
    """Protect sensitive text using privacy guards."""
    console.print("🛡️  Applying privacy protection...", style="bold yellow")
    
    # Parse strategies
    strategy_names = [s.strip() for s in strategies.split(",")]
    redaction_strategies = []
    
    for name in strategy_names:
        try:
            strategy = RedactionStrategy(name)
            redaction_strategies.append(strategy)
        except ValueError:
            console.print(f"⚠️  Unknown strategy: {name}", style="yellow")
    
    # Apply protection
    guard = ContextGuard(redaction_strategies)
    protected_text = guard.protect(text, sensitivity)
    
    console.print("\n📝 Original text:", style="bold")
    console.print(text)
    
    console.print("\n🔒 Protected text:", style="bold green")
    console.print(protected_text)
    
    # Show redaction explanation
    explanation = guard.explain_redactions(text)
    if explanation["total_redactions"] > 0:
        console.print(f"\n🔍 Applied {explanation['total_redactions']} redactions", style="bold cyan")


@app.command()
def validate_config(
    config_path: Path = typer.Argument(..., help="Path to configuration file")
):
    """Validate privacy configuration file."""
    try:
        privacy_config = PrivacyConfig.from_yaml(config_path)
        privacy_config.validate()
        console.print("✅ Configuration is valid!", style="bold green")
        
        # Display configuration
        table = Table(title="Privacy Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Epsilon", str(privacy_config.epsilon))
        table.add_row("Delta", str(privacy_config.delta))
        table.add_row("Max Grad Norm", str(privacy_config.max_grad_norm))
        table.add_row("Noise Multiplier", str(privacy_config.noise_multiplier))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Configuration validation failed: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8080, help="Server port"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the FastAPI server for the privacy-preserving training API."""
    import uvicorn
    from .api.server import create_app
    
    console.print(f"🚀 Starting Privacy-Preserving Agent Finetuner API", style="bold green")
    console.print(f"📡 Server: http://{host}:{port}")
    console.print(f"📚 API Documentation: http://{host}:{port}/docs")
    
    app_instance = create_app()
    
    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload
    )


@app.command() 
def estimate_cost(
    epsilon: float = typer.Argument(..., help="Target epsilon budget"),
    steps: int = typer.Option(1000, help="Number of training steps"),
    sample_rate: float = typer.Option(0.01, help="Sample rate for training"),
    noise_multiplier: float = typer.Option(0.5, help="Noise multiplier"),
):
    """Estimate privacy cost for given training parameters."""
    privacy_config = PrivacyConfig(
        epsilon=epsilon,
        noise_multiplier=noise_multiplier
    )
    
    estimated_cost = privacy_config.estimate_privacy_cost(steps, sample_rate)
    
    table = Table(title="Privacy Cost Estimation")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Target Epsilon", str(epsilon))
    table.add_row("Training Steps", str(steps))
    table.add_row("Sample Rate", str(sample_rate))
    table.add_row("Noise Multiplier", str(noise_multiplier))
    table.add_row("Estimated Cost", f"{estimated_cost:.6f}")
    table.add_row("Remaining Budget", f"{max(0, epsilon - estimated_cost):.6f}")
    
    console.print(table)
    
    if estimated_cost > epsilon:
        console.print("⚠️  Warning: Estimated cost exceeds budget!", style="bold red")
    else:
        console.print("✅ Budget is sufficient for training", style="bold green")


if __name__ == "__main__":
    app()