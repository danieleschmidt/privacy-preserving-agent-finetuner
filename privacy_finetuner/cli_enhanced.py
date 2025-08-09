"""Enhanced command-line interface for privacy-preserving agent finetuner.

This CLI provides comprehensive access to privacy-preserving training, context protection,
and monitoring capabilities with rich output formatting and detailed feedback.
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from . import __version__
from .core import PrivateTrainer, PrivacyConfig, ContextGuard, RedactionStrategy

# Enhanced CLI application
app = typer.Typer(
    name="privacy-finetuner",
    help="🔒 Privacy-Preserving Agent Finetuner - Advanced CLI with rich output",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()

# Build information
build_info = {
    "build_id": os.getenv("BUILD_ID", "development"),
    "build_date": os.getenv("BUILD_DATE", datetime.now().isoformat()),
    "commit_hash": os.getenv("COMMIT_HASH", "unknown")
}


def version_callback(value: bool):
    """Show version information with system details."""
    if value:
        import platform
        console.print(Panel(
            f"[bold green]Privacy-Preserving Agent Finetuner[/bold green] v{__version__}\n"
            f"[dim]Build:[/dim] {build_info.get('build_id', 'development')}\n"
            f"[dim]Python:[/dim] {sys.version.split()[0]}\n"
            f"[dim]Platform:[/dim] {platform.system()} {platform.release()}\n"
            f"[dim]Architecture:[/dim] {platform.machine()}",
            title="Version Information",
            border_style="blue"
        ))
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V", callback=version_callback, 
        help="Show version information"
    )
):
    """Privacy-Preserving Agent Finetuner CLI"""
    pass


@app.command()
def train(
    dataset: str = typer.Argument(..., help="Path to training dataset (JSONL format)"),
    model_name: str = typer.Option("microsoft/DialoGPT-small", "--model", "-m", help="HuggingFace model identifier"),
    epsilon: float = typer.Option(1.0, "--epsilon", "-e", help="Privacy budget (epsilon)", min=0.01, max=50.0),
    delta: float = typer.Option(1e-5, "--delta", "-d", help="Privacy parameter (delta)", min=1e-10, max=1e-2),
    epochs: int = typer.Option(3, "--epochs", help="Number of training epochs", min=1, max=100),
    batch_size: int = typer.Option(8, "--batch-size", help="Training batch size", min=1, max=128),
    learning_rate: float = typer.Option(5e-5, "--learning-rate", help="Learning rate", min=1e-8, max=1e-1),
    output_dir: str = typer.Option("./models", "--output", "-o", help="Output directory for trained model"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    use_mcp: bool = typer.Option(True, "--mcp/--no-mcp", help="Enable MCP gateway"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate configuration without training"),
    checkpoints: bool = typer.Option(True, "--checkpoints/--no-checkpoints", help="Enable model checkpointing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Train a model with differential privacy guarantees.
    
    This command fine-tunes a language model using differential privacy to ensure
    sensitive training data never leaks into the model. Supports various privacy
    budgets and advanced monitoring.
    """
    
    # Display training banner with configuration summary
    config_summary = Table(title="Training Configuration", show_header=True, header_style="bold magenta")
    config_summary.add_column("Parameter", style="cyan")
    config_summary.add_column("Value", style="green")
    
    config_summary.add_row("Model", model_name)
    config_summary.add_row("Dataset", dataset)
    config_summary.add_row("Privacy Budget (ε)", f"{epsilon}")
    config_summary.add_row("Privacy Parameter (δ)", f"{delta}")
    config_summary.add_row("Epochs", str(epochs))
    config_summary.add_row("Batch Size", str(batch_size))
    config_summary.add_row("Learning Rate", f"{learning_rate}")
    
    console.print(Panel(
        config_summary,
        title="🚀 Privacy-Preserving Training",
        border_style="green"
    ))
    
    # Load or create privacy configuration
    if config_file:
        try:
            privacy_config = PrivacyConfig.from_yaml(Path(config_file))
            console.print(f"[green]📝 Loaded configuration from {config_file}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to load config: {e}[/red]")
            raise typer.Exit(1)
    else:
        privacy_config = PrivacyConfig(
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=1.0,
            noise_multiplier=0.5
        )
        console.print("[yellow]⚙️ Using CLI parameters for privacy configuration[/yellow]")
    
    # Validate configuration
    try:
        privacy_config.validate()
        console.print("[green]✅ Privacy configuration validated[/green]")
        
        # Show privacy risk assessment
        risk_assessment = privacy_config.privacy_risk_assessment()
        risk_color = {
            "MINIMAL": "green",
            "LOW": "yellow", 
            "MEDIUM": "orange",
            "HIGH": "red"
        }.get(risk_assessment['risk_level'], "white")
        
        console.print(f"[{risk_color}]🛡️ Privacy Risk: {risk_assessment['risk_level']}[/{risk_color}]")
        
        if risk_assessment['mitigation_needed']:
            console.print("[yellow]⚠️ Consider reviewing privacy parameters for stronger guarantees[/yellow]")
        
    except ValueError as e:
        console.print(f"[red]❌ Invalid privacy configuration: {e}[/red]")
        raise typer.Exit(1)
    
    # Dry run - validate only
    if dry_run:
        console.print(Panel(
            "Configuration validated successfully. No training performed in dry run mode.",
            title="🧪 Dry Run Complete",
            border_style="blue"
        ))
        return
    
    # Confirm training start if high risk
    if risk_assessment['risk_level'] in ['MEDIUM', 'HIGH']:
        if not Confirm.ask(f"Privacy risk is {risk_assessment['risk_level']}. Continue with training?"):
            console.print("[yellow]Training cancelled by user[/yellow]")
            raise typer.Exit(0)
    
    # Initialize trainer
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing trainer...", total=None)
            
            trainer = PrivateTrainer(
                model_name=model_name,
                privacy_config=privacy_config,
                use_mcp_gateway=use_mcp
            )
            
            progress.update(task, description="Trainer initialized ✅")
        
        # Start training
        console.print("[bold green]🚀 Starting training process...[/bold green]")
        
        results = trainer.train(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_interval=100 if checkpoints else None
        )
        
        # Display success summary with results
        results_table = Table(title="Training Results", show_header=True, header_style="bold green")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Status", "✅ Completed Successfully")
        results_table.add_row("Model Path", results.get('model_path', 'N/A'))
        results_table.add_row("Final Loss", f"{results.get('final_loss', 0):.4f}")
        results_table.add_row("Privacy Spent (ε)", f"{results.get('privacy_spent', 0):.6f}")
        results_table.add_row("Training Steps", str(results.get('total_steps', 0)))
        results_table.add_row("Epochs Completed", str(results.get('epochs_completed', 0)))
        
        console.print(Panel(
            results_table,
            title="🎉 Training Complete",
            border_style="green"
        ))
        
        # Generate and display privacy report
        privacy_report = trainer.get_privacy_report()
        remaining_budget = privacy_report.get('remaining_budget', 0)
        
        if remaining_budget > 0:
            console.print(f"[green]💰 Remaining privacy budget: ε = {remaining_budget:.6f}[/green]")
        else:
            console.print("[yellow]⚠️ Privacy budget fully consumed[/yellow]")
    
    except Exception as e:
        error_table = Table(title="Training Error", show_header=False, border_style="red")
        error_table.add_column("Details", style="red")
        error_table.add_row(f"Error: {type(e).__name__}")
        error_table.add_row(f"Message: {str(e)}")
        
        console.print(Panel(
            error_table,
            title="❌ Training Failed",
            border_style="red"
        ))
        
        if verbose:
            import traceback
            console.print("[bold red]Full Traceback:[/bold red]")
            console.print(f"[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(1)


@app.command()
def assess(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file to assess"),
    epsilon: Optional[float] = typer.Option(None, "--epsilon", help="Privacy budget to assess"),
    delta: Optional[float] = typer.Option(None, "--delta", help="Privacy parameter to assess"),
    export_report: Optional[str] = typer.Option(None, "--export", help="Export assessment report to file"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed privacy analysis")
):
    """Assess privacy configuration and provide comprehensive recommendations."""
    
    console.print(Panel(
        "Privacy Configuration Assessment",
        title="🔍 Privacy Analyzer",
        border_style="cyan"
    ))
    
    # Load configuration
    if config_file:
        try:
            privacy_config = PrivacyConfig.from_yaml(Path(config_file))
            console.print(f"[green]📝 Loaded configuration from {config_file}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to load config: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Use provided parameters or defaults
        privacy_config = PrivacyConfig(
            epsilon=epsilon or 1.0,
            delta=delta or 1e-5
        )
        console.print("[yellow]⚙️ Using provided parameters or defaults[/yellow]")
    
    try:
        privacy_config.validate()
        
        # Display detailed assessment
        assessment_table = Table(title="Configuration Analysis", show_header=True, header_style="bold cyan")
        assessment_table.add_column("Parameter", style="cyan")
        assessment_table.add_column("Value", style="white")
        assessment_table.add_column("Assessment", style="yellow")
        
        assessment_table.add_row("Privacy Budget (ε)", f"{privacy_config.epsilon}", "✅ Valid" if privacy_config.epsilon > 0 else "❌ Invalid")
        assessment_table.add_row("Privacy Parameter (δ)", f"{privacy_config.delta}", "✅ Valid" if 0 < privacy_config.delta < 1 else "❌ Invalid")
        assessment_table.add_row("Noise Multiplier", f"{privacy_config.noise_multiplier}", "✅ Adequate" if privacy_config.noise_multiplier >= 0.5 else "⚠️ Low")
        assessment_table.add_row("Gradient Clipping", f"{privacy_config.max_grad_norm}", "✅ Standard" if privacy_config.max_grad_norm == 1.0 else "⚠️ Non-standard")
        
        console.print(assessment_table)
        
        # Privacy risk assessment
        risk_assessment = privacy_config.privacy_risk_assessment()
        risk_color = {
            "MINIMAL": "green",
            "LOW": "yellow", 
            "MEDIUM": "orange",
            "HIGH": "red"
        }.get(risk_assessment['risk_level'], "white")
        
        risk_panel = Panel(
            f"Risk Level: [{risk_color}]{risk_assessment['risk_level']}[/{risk_color}]\n"
            f"Risk Score: {risk_assessment['risk_score']}/10\n"
            f"Mitigation Needed: {'Yes' if risk_assessment['mitigation_needed'] else 'No'}",
            title="🛡️ Privacy Risk Assessment",
            border_style=risk_color
        )
        console.print(risk_panel)
        
        # Recommendations
        recommendations = privacy_config.get_recommendations()
        if recommendations:
            rec_table = Table(title="Recommendations", show_header=True, header_style="bold blue")
            rec_table.add_column("Parameter", style="cyan")
            rec_table.add_column("Recommendation", style="yellow")
            
            for param, recommendation in recommendations.items():
                rec_table.add_row(param.title(), recommendation)
            
            console.print(Panel(
                rec_table,
                title="💡 Configuration Recommendations",
                border_style="blue"
            ))
        
        console.print(f"[green]✅ Assessment completed successfully[/green]")
        
        # Export report if requested
        if export_report:
            try:
                report_data = {
                    "configuration": privacy_config.to_dict(),
                    "risk_assessment": risk_assessment,
                    "recommendations": recommendations,
                    "timestamp": datetime.now().isoformat()
                }
                
                export_path = Path(export_report)
                export_path.parent.mkdir(parents=True, exist_ok=True)
                
                if export_path.suffix.lower() == '.json':
                    import json
                    with open(export_path, 'w') as f:
                        json.dump(report_data, f, indent=2)
                else:
                    with open(export_path, 'w') as f:
                        f.write(f"Privacy Assessment Report\n{'='*40}\n\n")
                        f.write(f"Risk Level: {risk_assessment['risk_level']}\n")
                        f.write(f"Risk Score: {risk_assessment['risk_score']}/10\n\n")
                        f.write("Recommendations:\n")
                        for param, rec in recommendations.items():
                            f.write(f"- {param}: {rec}\n")
                
                console.print(f"[green]✅ Report exported to: {export_path}[/green]")
            except Exception as e:
                console.print(f"[red]❌ Failed to export report: {e}[/red]")
    
    except Exception as e:
        console.print(Panel(
            f"Error: {type(e).__name__}\nMessage: {str(e)}",
            title="❌ Assessment Failed",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def protect(
    text: str = typer.Argument(..., help="Text to protect"),
    strategies: str = typer.Option("pii_removal", "--strategies", "-s", help="Comma-separated redaction strategies"),
    sensitivity: str = typer.Option("medium", "--sensitivity", help="Sensitivity level (low/medium/high)"),
    explain: bool = typer.Option(False, "--explain", help="Show detailed redaction explanation"),
    batch_file: Optional[str] = typer.Option(None, "--batch", help="Process multiple texts from file")
):
    """Protect sensitive text using advanced privacy guards."""
    
    console.print(Panel(
        "Privacy Protection Service",
        title="🛡️ Context Guard",
        border_style="blue"
    ))
    
    # Parse redaction strategies
    strategy_names = [s.strip().upper() for s in strategies.split(",")]
    redaction_strategies = []
    
    for name in strategy_names:
        try:
            strategy = RedactionStrategy[name]
            redaction_strategies.append(strategy)
            console.print(f"[green]✅ Enabled strategy: {strategy.value}[/green]")
        except KeyError:
            console.print(f"[yellow]⚠️ Unknown strategy: {name}[/yellow]")
    
    if not redaction_strategies:
        console.print("[red]❌ No valid redaction strategies specified[/red]")
        raise typer.Exit(1)
    
    # Initialize context guard
    try:
        guard = ContextGuard(redaction_strategies)
        console.print(f"[green]🔧 Context guard initialized with {len(redaction_strategies)} strategies[/green]")
        
        if batch_file:
            # Process batch file
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
                
                console.print(f"[blue]📝 Processing {len(texts)} texts from {batch_file}[/blue]")
                
                results_table = Table(title="Batch Protection Results", show_header=True, header_style="bold green")
                results_table.add_column("Index", style="cyan")
                results_table.add_column("Original (truncated)", style="white", max_width=30)
                results_table.add_column("Protected (truncated)", style="green", max_width=30)
                results_table.add_column("Redactions", style="yellow")
                
                for i, original_text in enumerate(texts, 1):
                    protected = guard.protect(original_text, sensitivity)
                    analysis = guard.analyze_sensitivity(original_text)
                    
                    results_table.add_row(
                        str(i),
                        original_text[:30] + "..." if len(original_text) > 30 else original_text,
                        protected[:30] + "..." if len(protected) > 30 else protected,
                        str(analysis.get('redaction_count', 0))
                    )
                
                console.print(results_table)
                
            except Exception as e:
                console.print(f"[red]❌ Failed to process batch file: {e}[/red]")
                raise typer.Exit(1)
        
        else:
            # Process single text
            protected_text = guard.protect(text, sensitivity)
            
            # Create protection results display
            protection_table = Table(title="Protection Results", show_header=True, header_style="bold green")
            protection_table.add_column("Type", style="cyan")
            protection_table.add_column("Content", style="white")
            
            protection_table.add_row("Original", text)
            protection_table.add_row("Protected", f"[green]{protected_text}[/green]")
            
            console.print(protection_table)
            
            # Show sensitivity analysis
            analysis = guard.analyze_sensitivity(text)
            sensitivity_panel = Panel(
                f"Sensitivity Level: {analysis['sensitivity_level']}\n"
                f"Sensitivity Score: {analysis['sensitivity_score']}/10\n"
                f"Redactions Applied: {analysis.get('redaction_count', 0)}",
                title="📊 Sensitivity Analysis",
                border_style="yellow"
            )
            console.print(sensitivity_panel)
            
            # Show detailed explanation if requested
            if explain:
                try:
                    explanation = guard.explain_redactions(text)
                    if explanation.get("total_redactions", 0) > 0:
                        exp_table = Table(title="Redaction Details", show_header=True, header_style="bold cyan")
                        exp_table.add_column("Pattern", style="cyan")
                        exp_table.add_column("Replacement", style="green")
                        exp_table.add_column("Count", style="yellow")
                        
                        for redaction in explanation.get("redactions", []):
                            exp_table.add_row(
                                redaction.get("pattern", "N/A"),
                                redaction.get("replacement", "N/A"),
                                str(redaction.get("count", 0))
                            )
                        
                        console.print(exp_table)
                    else:
                        console.print("[green]✅ No redactions were necessary[/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠️ Could not generate detailed explanation: {e}[/yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ Protection failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def config(
    init: bool = typer.Option(False, "--init", help="Initialize new configuration file"),
    validate: Optional[str] = typer.Option(None, "--validate", help="Validate existing configuration file"),
    export_env: bool = typer.Option(False, "--export-env", help="Export configuration as environment variables"),
    template: str = typer.Option("basic", "--template", help="Configuration template (basic/advanced/research)")
):
    """Manage privacy configuration files."""
    
    if init:
        console.print(Panel(
            "Configuration Initialization",
            title="⚙️ Config Manager",
            border_style="blue"
        ))
        
        # Create configuration based on template
        if template == "basic":
            config = PrivacyConfig(epsilon=1.0, delta=1e-5)
        elif template == "advanced":
            config = PrivacyConfig(epsilon=3.0, delta=1e-6, noise_multiplier=0.8, federated_enabled=True)
        elif template == "research":
            config = PrivacyConfig(epsilon=0.5, delta=1e-8, noise_multiplier=1.2, attestation_required=True)
        else:
            console.print(f"[red]❌ Unknown template: {template}[/red]")
            raise typer.Exit(1)
        
        config_path = Path(f"privacy_config_{template}.yaml")
        
        try:
            config.save_yaml(config_path)
            console.print(f"[green]✅ Configuration saved to: {config_path}[/green]")
            
            # Display created configuration
            config_table = Table(title=f"Generated Configuration ({template})", show_header=True, header_style="bold green")
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", style="white")
            
            for key, value in config.to_dict().items():
                if value is not None:
                    config_table.add_row(key.replace("_", " ").title(), str(value))
            
            console.print(config_table)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to save configuration: {e}[/red]")
            raise typer.Exit(1)
    
    elif validate:
        console.print(Panel(
            f"Validating: {validate}",
            title="✅ Config Validator",
            border_style="green"
        ))
        
        try:
            config = PrivacyConfig.from_yaml(Path(validate))
            config.validate()
            console.print("[green]✅ Configuration is valid![/green]")
            
            # Show recommendations
            recommendations = config.get_recommendations()
            if recommendations:
                console.print("[blue]💡 Recommendations for optimization:[/blue]")
                for param, rec in recommendations.items():
                    console.print(f"  • {param}: {rec}")
                    
        except Exception as e:
            console.print(f"[red]❌ Configuration validation failed: {e}[/red]")
            raise typer.Exit(1)
    
    elif export_env:
        console.print(Panel(
            "Environment Variable Export",
            title="🔧 Environment Setup",
            border_style="cyan"
        ))
        
        # Export basic environment variables
        env_vars = {
            "PRIVACY_EPSILON": "1.0",
            "PRIVACY_DELTA": "1e-5",
            "PRIVACY_MAX_GRAD_NORM": "1.0",
            "PRIVACY_NOISE_MULTIPLIER": "0.5",
            "PRIVACY_ACCOUNTING_MODE": "rdp"
        }
        
        console.print("[bold]Add these environment variables:[/bold]")
        for var, value in env_vars.items():
            console.print(f"export {var}={value}")
        
        # Save to file
        env_file = Path(".env.privacy")
        with open(env_file, 'w') as f:
            for var, value in env_vars.items():
                f.write(f"{var}={value}\n")
        
        console.print(f"[green]✅ Environment variables saved to: {env_file}[/green]")
    
    else:
        console.print("[yellow]⚠️ Please specify an action: --init, --validate, or --export-env[/yellow]")


if __name__ == "__main__":
    app()