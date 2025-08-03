"""FastAPI server for privacy-preserving training API."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import logging

from ..core import PrivateTrainer, PrivacyConfig
from ..utils.monitoring import PrivacyBudgetMonitor

logger = logging.getLogger(__name__)


class TrainingRequest(BaseModel):
    """Request model for training endpoint."""
    model_name: str
    dataset_path: str
    privacy_config: Dict[str, Any]
    training_params: Dict[str, Any] = {}


class PrivacyReportResponse(BaseModel):
    """Response model for privacy report endpoint."""
    epsilon_spent: float
    delta: float
    remaining_budget: float
    accounting_mode: str


def create_app() -> FastAPI:
    """Create FastAPI application with privacy endpoints."""
    app = FastAPI(
        title="Privacy-Preserving Agent Finetuner",
        description="Enterprise-grade API for differential privacy training",
        version="0.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global trainer instance (in production, use proper session management)
    trainer_instance = None
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "privacy-finetuner"}
    
    @app.post("/train")
    async def start_training(request: TrainingRequest):
        """Start differential privacy training."""
        try:
            nonlocal trainer_instance
            
            # Create privacy configuration
            privacy_config = PrivacyConfig(**request.privacy_config)
            
            # Initialize trainer
            trainer_instance = PrivateTrainer(
                model_name=request.model_name,
                privacy_config=privacy_config
            )
            
            # Start training
            result = trainer_instance.train(
                dataset=request.dataset_path,
                **request.training_params
            )
            
            return {"message": "Training started", "details": result}
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/privacy-report", response_model=PrivacyReportResponse)
    async def get_privacy_report():
        """Get current privacy budget status."""
        if trainer_instance is None:
            raise HTTPException(status_code=404, detail="No active training session")
        
        report = trainer_instance.get_privacy_report()
        return PrivacyReportResponse(**report)
    
    @app.post("/protect-context")
    async def protect_context(text: str, sensitivity: str = "medium", strategies: list = ["pii_removal"]):
        """Protect sensitive context using privacy guards."""
        from ..core.context_guard import ContextGuard, RedactionStrategy
        
        try:
            # Convert string strategies to enum
            redaction_strategies = []
            for strategy_name in strategies:
                if hasattr(RedactionStrategy, strategy_name.upper()):
                    redaction_strategies.append(getattr(RedactionStrategy, strategy_name.upper()))
            
            # Create context guard
            guard = ContextGuard(redaction_strategies)
            
            # Apply protection
            protected_text = guard.protect(text, sensitivity)
            redaction_report = guard.explain_redactions(text)
            
            return {
                "protected_text": protected_text,
                "redaction_report": redaction_report,
                "strategies_applied": strategies
            }
            
        except Exception as e:
            logger.error(f"Context protection failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)