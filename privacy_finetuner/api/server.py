"""FastAPI server for privacy-preserving training API."""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

from ..core import (
    PrivateTrainer, PrivacyConfig, ContextGuard, RedactionStrategy,
    PrivacyBudgetTracker, PrivacyAttackDetector, PrivacyComplianceChecker
)
from ..database import (
    get_database, get_db_session, TrainingJobRepository, 
    ModelRepository, PrivacyBudgetRepository, AuditLogRepository,
    AdvancedPrivacyOperations, QueryOptimizer
)
from ..utils.monitoring import PrivacyBudgetMonitor

logger = logging.getLogger(__name__)


# Security
security = HTTPBearer(auto_error=False)

# Pydantic Models
class TrainingRequest(BaseModel):
    """Request model for training endpoint."""
    job_name: str = Field(..., description="Name for the training job")
    model_name: str = Field(..., description="HuggingFace model identifier")
    dataset_path: str = Field(..., description="Path to training dataset")
    privacy_config: Dict[str, Any] = Field(..., description="Privacy configuration")
    training_params: Dict[str, Any] = Field(default_factory=dict, description="Training parameters")
    
    @validator('privacy_config')
    def validate_privacy_config(cls, v):
        required_fields = ['epsilon', 'delta', 'max_grad_norm', 'noise_multiplier']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required privacy config field: {field}")
        return v


class TrainingResponse(BaseModel):
    """Response model for training endpoint."""
    job_id: str
    status: str
    message: str
    privacy_budget_used: float
    estimated_completion: Optional[datetime] = None


class PrivacyReportResponse(BaseModel):
    """Response model for privacy report endpoint."""
    epsilon_spent: float
    delta: float
    remaining_budget: float
    accounting_mode: str
    budget_utilization_percent: float
    recommendations: List[str]


class ContextProtectionRequest(BaseModel):
    """Request model for context protection."""
    text: str = Field(..., description="Text to protect")
    sensitivity_level: str = Field(default="medium", description="Sensitivity level")
    strategies: List[str] = Field(default=["pii_removal"], description="Protection strategies")


class ContextProtectionResponse(BaseModel):
    """Response model for context protection."""
    protected_text: str
    original_length: int
    protected_length: int
    redactions_applied: int
    sensitivity_analysis: Dict[str, Any]
    compliance_status: Dict[str, bool]


class ModelEvaluationRequest(BaseModel):
    """Request model for model evaluation."""
    model_id: str
    test_dataset_path: str
    evaluation_params: Dict[str, Any] = Field(default_factory=dict)


class PrivacyAnalysisResponse(BaseModel):
    """Response model for privacy analysis."""
    analysis_id: str
    user_budget_summary: Dict[str, Any]
    risk_level: str
    anomalies_detected: List[Dict[str, Any]]
    compliance_score: float
    recommendations: List[str]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime
    version: str
    database_status: str
    cache_status: str
    privacy_engine_status: str


# Global state
app_state = {
    "trainers": {},  # job_id -> trainer_instance
    "budget_tracker": None,
    "attack_detector": None,
    "compliance_checker": None,
    "query_optimizer": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Initializing Privacy-Preserving Agent Finetuner API...")
    
    # Initialize database
    db = get_database()
    
    # Initialize global components
    redis_client = db.get_cache_client()
    app_state["query_optimizer"] = QueryOptimizer(redis_client)
    app_state["budget_tracker"] = PrivacyBudgetTracker(total_epsilon=10.0, total_delta=1e-5)
    app_state["attack_detector"] = PrivacyAttackDetector()
    app_state["compliance_checker"] = PrivacyComplianceChecker()
    
    logger.info("API initialization complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Privacy-Preserving Agent Finetuner API...")
    
    # Clean up active trainers
    for job_id, trainer in app_state["trainers"].items():
        logger.info(f"Cleaning up training job: {job_id}")
    
    app_state["trainers"].clear()
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create FastAPI application with privacy endpoints."""
    app = FastAPI(
        title="Privacy-Preserving Agent Finetuner",
        description="Enterprise-grade API for differential privacy machine learning",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request, call_next):
        start_time = datetime.now()
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
        )
        return response
    
    # Dependency for database session
    def get_session():
        return next(get_db_session())
    
    # Authentication dependency (simplified for demo)
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if not credentials or credentials.scheme != "Bearer":
            return None  # Allow anonymous access for demo
        # In production, validate JWT token here
        return {"user_id": "demo-user", "username": "demo"}
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Comprehensive health check endpoint."""
        db = get_database()
        health_status = db.health_check()
        
        return HealthResponse(
            status="healthy" if health_status["database"] == "healthy" else "degraded",
            timestamp=datetime.now(),
            version="1.0.0",
            database_status=health_status["database"],
            cache_status=health_status["redis"],
            privacy_engine_status="active"
        )
    
    @app.post("/train", response_model=TrainingResponse)
    async def start_training(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        session=Depends(get_session),
        current_user=Depends(get_current_user)
    ):
        """Start differential privacy training job."""
        try:
            # Generate job ID
            job_id = str(uuid4())
            
            # Create privacy configuration
            privacy_config = PrivacyConfig(**request.privacy_config)
            privacy_config.validate()
            
            # Check privacy budget availability
            budget_tracker = app_state["budget_tracker"]
            if not budget_tracker.record_event(
                "training_start", 
                privacy_config.epsilon, 
                privacy_config.delta,
                {"job_id": job_id, "model_name": request.model_name}
            ):
                raise HTTPException(
                    status_code=400, 
                    detail="Insufficient privacy budget for this training job"
                )
            
            # Initialize trainer
            trainer = PrivateTrainer(
                model_name=request.model_name,
                privacy_config=privacy_config
            )
            
            # Store trainer instance
            app_state["trainers"][job_id] = trainer
            
            # Create database record
            training_repo = TrainingJobRepository(session)
            job_record = training_repo.create(
                job_name=request.job_name,
                model_name=request.model_name,
                dataset_id=None,  # Would be resolved from dataset_path
                user_id=current_user["user_id"] if current_user else None,
                target_epsilon=privacy_config.epsilon,
                target_delta=privacy_config.delta,
                noise_multiplier=privacy_config.noise_multiplier,
                max_grad_norm=privacy_config.max_grad_norm,
                accounting_mode=privacy_config.accounting_mode,
                **request.training_params
            )
            session.commit()
            
            # Start training in background
            background_tasks.add_task(
                _execute_training,
                job_id,
                trainer,
                request.dataset_path,
                request.training_params,
                str(job_record.id)
            )
            
            return TrainingResponse(
                job_id=job_id,
                status="started",
                message="Training job initiated successfully",
                privacy_budget_used=privacy_config.epsilon,
                estimated_completion=None  # Would calculate based on dataset size
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _execute_training(job_id: str, trainer: PrivateTrainer, dataset_path: str, training_params: dict, db_job_id: str):
        """Execute training in background."""
        try:
            logger.info(f"Starting background training for job {job_id}")
            
            # Start training
            result = trainer.train(
                dataset=dataset_path,
                **training_params
            )
            
            # Update job status in database
            with get_database().session_scope() as session:
                training_repo = TrainingJobRepository(session)
                training_repo.update(
                    UUID(db_job_id),
                    status="completed",
                    epsilon_spent=result.get("privacy_spent", 0),
                    final_loss=result.get("final_loss", 0),
                    completed_at=datetime.now()
                )
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            
            # Update job status to failed
            with get_database().session_scope() as session:
                training_repo = TrainingJobRepository(session)
                training_repo.update(
                    UUID(db_job_id),
                    status="failed",
                    error_message=str(e)
                )
        finally:
            # Clean up trainer instance
            if job_id in app_state["trainers"]:
                del app_state["trainers"][job_id]
    
    @app.get("/training/{job_id}/status")
    async def get_training_status(job_id: str, session=Depends(get_session)):
        """Get training job status."""
        if job_id in app_state["trainers"]:
            trainer = app_state["trainers"][job_id]
            return {
                "job_id": job_id,
                "status": "running",
                "privacy_report": trainer.get_privacy_report()
            }
        
        # Check database for completed/failed jobs
        training_repo = TrainingJobRepository(session)
        jobs = training_repo.get_all()
        
        # In a real implementation, we'd query by job_id
        return {"job_id": job_id, "status": "not_found"}
    
    @app.get("/privacy-budget", response_model=PrivacyReportResponse)
    async def get_privacy_budget_status(current_user=Depends(get_current_user)):
        """Get current privacy budget status."""
        budget_tracker = app_state["budget_tracker"]
        usage_summary = budget_tracker.get_usage_summary()
        
        return PrivacyReportResponse(
            epsilon_spent=usage_summary["spent_budget"]["epsilon"],
            delta=usage_summary["spent_budget"]["delta"],
            remaining_budget=usage_summary["remaining_budget"]["epsilon"],
            accounting_mode="rdp",
            budget_utilization_percent=usage_summary["utilization"]["epsilon_percent"],
            recommendations=[
                "Monitor budget consumption regularly",
                "Consider increasing noise for better privacy"
            ]
        )
    
    @app.post("/protect-context", response_model=ContextProtectionResponse)
    async def protect_context(request: ContextProtectionRequest):
        """Protect sensitive context using privacy guards."""
        try:
            # Convert string strategies to enum
            redaction_strategies = []
            for strategy_name in request.strategies:
                if hasattr(RedactionStrategy, strategy_name.upper()):
                    redaction_strategies.append(getattr(RedactionStrategy, strategy_name.upper()))
            
            if not redaction_strategies:
                redaction_strategies = [RedactionStrategy.PII_REMOVAL]
            
            # Create context guard
            guard = ContextGuard(redaction_strategies)
            
            # Apply protection
            protected_text = guard.protect(request.text, request.sensitivity_level)
            sensitivity_analysis = guard.analyze_sensitivity(request.text)
            privacy_report = guard.create_privacy_report(request.text, protected_text)
            
            return ContextProtectionResponse(
                protected_text=protected_text,
                original_length=len(request.text),
                protected_length=len(protected_text),
                redactions_applied=privacy_report["redaction_analysis"]["total_redactions"],
                sensitivity_analysis=sensitivity_analysis,
                compliance_status=privacy_report["privacy_compliance"]
            )
            
        except Exception as e:
            logger.error(f"Context protection failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/evaluate-model", response_model=Dict[str, Any])
    async def evaluate_model(
        request: ModelEvaluationRequest,
        session=Depends(get_session),
        current_user=Depends(get_current_user)
    ):
        """Evaluate a trained model with privacy analysis."""
        try:
            model_repo = ModelRepository(session)
            model = model_repo.get_by_id(UUID(request.model_id))
            
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Load trainer for evaluation
            privacy_config = PrivacyConfig(
                epsilon=model.epsilon_spent,
                delta=model.delta_value,
                noise_multiplier=model.noise_multiplier or 0.5,
                max_grad_norm=model.max_grad_norm or 1.0
            )
            
            trainer = PrivateTrainer(
                model_name=model.base_model,
                privacy_config=privacy_config
            )
            
            # Load model checkpoint
            if model.model_path:
                trainer.load_checkpoint(model.model_path)
            
            # Perform evaluation
            eval_results = trainer.evaluate(
                request.test_dataset_path,
                **request.evaluation_params
            )
            
            # Privacy-utility analysis
            advanced_ops = AdvancedPrivacyOperations(session, app_state["query_optimizer"])
            performance_metrics = advanced_ops.analyze_model_privacy_utility_tradeoff(UUID(request.model_id))
            
            return {
                "model_id": request.model_id,
                "evaluation_results": eval_results,
                "performance_metrics": performance_metrics.__dict__,
                "privacy_guarantees": {
                    "epsilon_spent": model.epsilon_spent,
                    "delta_value": model.delta_value,
                    "formal_privacy": True
                }
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/privacy-analysis", response_model=PrivacyAnalysisResponse)
    async def get_privacy_analysis(
        session=Depends(get_session),
        current_user=Depends(get_current_user)
    ):
        """Get comprehensive privacy analysis and recommendations."""
        try:
            analysis_id = str(uuid4())
            
            # Initialize advanced operations
            advanced_ops = AdvancedPrivacyOperations(session, app_state["query_optimizer"])
            
            # Get user budget analysis (using demo user ID)
            user_id = UUID("00000000-0000-0000-0000-000000000000")  # Demo user
            budget_analysis = advanced_ops.analyze_privacy_budget_patterns(user_id)
            
            # Detect anomalies
            anomalies = advanced_ops.detect_privacy_anomalies(lookback_hours=24)
            
            # Generate compliance report
            compliance_report = advanced_ops.generate_privacy_compliance_report(user_id)
            
            return PrivacyAnalysisResponse(
                analysis_id=analysis_id,
                user_budget_summary=budget_analysis.__dict__,
                risk_level=budget_analysis.risk_level,
                anomalies_detected=anomalies,
                compliance_score=compliance_report.get("compliance_summary", {}).get("compliance_rate_percent", 0),
                recommendations=budget_analysis.recommendations
            )
            
        except Exception as e:
            logger.error(f"Privacy analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models")
    async def list_models(
        session=Depends(get_session),
        current_user=Depends(get_current_user),
        limit: int = 50
    ):
        """List available trained models."""
        model_repo = ModelRepository(session)
        
        if current_user:
            models = model_repo.get_by_owner(UUID(current_user["user_id"]), limit=limit)
        else:
            models = model_repo.get_all(limit=limit)
        
        return {
            "models": [
                {
                    "id": str(model.id),
                    "name": model.name,
                    "version": model.version,
                    "base_model": model.base_model,
                    "epsilon_spent": model.epsilon_spent,
                    "eval_accuracy": model.eval_accuracy,
                    "created_at": model.created_at,
                    "is_deployed": model.is_deployed
                }
                for model in models
            ],
            "total": len(models)
        }
    
    return app


# Main application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run server
    uvicorn.run(
        "privacy_finetuner.api.server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8080")),
        workers=1,  # Use 1 worker for development
        reload=os.getenv("DEVELOPMENT_MODE", "false").lower() == "true",
        log_level="info"
    )