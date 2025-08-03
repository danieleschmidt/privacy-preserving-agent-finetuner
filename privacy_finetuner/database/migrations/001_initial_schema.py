"""Initial database schema migration.

Creates all base tables for privacy-preserving agent finetuner.

Revision ID: 001
Create Date: 2024-08-03
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create initial database schema."""
    
    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('username', sa.String(100), nullable=False, unique=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255)),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, default=False),
        sa.Column('privacy_preferences', sa.JSON(), default={}),
        sa.Column('consent_given', sa.Boolean(), nullable=False, default=False),
        sa.Column('consent_date', sa.DateTime()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    
    # Create indexes for users
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_users_email', 'users', ['email'])
    
    # Datasets table
    op.create_table(
        'datasets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('file_path', sa.String(500), nullable=False),
        sa.Column('format', sa.String(50), nullable=False),
        sa.Column('size_bytes', sa.Integer()),
        sa.Column('record_count', sa.Integer()),
        sa.Column('contains_pii', sa.Boolean(), nullable=False, default=False),
        sa.Column('privacy_level', sa.String(20), nullable=False, default='medium'),
        sa.Column('data_classification', sa.String(50)),
        sa.Column('retention_policy', sa.String(100)),
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('is_public', sa.Boolean(), nullable=False, default=False),
        sa.Column('file_hash', sa.String(64)),
        sa.Column('metadata_hash', sa.String(64)),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id']),
        sa.CheckConstraint("privacy_level IN ('low', 'medium', 'high')", name='valid_privacy_level'),
        sa.CheckConstraint("format IN ('jsonl', 'csv', 'parquet', 'json')", name='valid_format'),
    )
    
    # Create indexes for datasets
    op.create_index('idx_datasets_name', 'datasets', ['name'])
    op.create_index('idx_dataset_owner_name', 'datasets', ['owner_id', 'name'])
    
    # Training jobs table
    op.create_table(
        'training_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_name', sa.String(255), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='queued'),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('epochs', sa.Integer(), nullable=False),
        sa.Column('batch_size', sa.Integer(), nullable=False),
        sa.Column('learning_rate', sa.Float(), nullable=False),
        sa.Column('max_steps', sa.Integer()),
        sa.Column('warmup_steps', sa.Integer()),
        sa.Column('target_epsilon', sa.Float(), nullable=False),
        sa.Column('target_delta', sa.Float(), nullable=False),
        sa.Column('noise_multiplier', sa.Float(), nullable=False),
        sa.Column('max_grad_norm', sa.Float(), nullable=False),
        sa.Column('accounting_mode', sa.String(10), nullable=False),
        sa.Column('epsilon_spent', sa.Float(), nullable=False, default=0.0),
        sa.Column('actual_steps', sa.Integer(), default=0),
        sa.Column('sample_rate', sa.Float()),
        sa.Column('started_at', sa.DateTime()),
        sa.Column('completed_at', sa.DateTime()),
        sa.Column('error_message', sa.Text()),
        sa.Column('progress', sa.Float(), default=0.0),
        sa.Column('final_loss', sa.Float()),
        sa.Column('best_eval_loss', sa.Float()),
        sa.Column('training_time_seconds', sa.Integer()),
        sa.Column('gpu_hours_used', sa.Float()),
        sa.Column('checkpoint_path', sa.String(500)),
        sa.Column('log_path', sa.String(500)),
        sa.Column('config_path', sa.String(500)),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.CheckConstraint("status IN ('queued', 'running', 'completed', 'failed', 'cancelled')", name='valid_status'),
        sa.CheckConstraint("target_epsilon > 0", name='positive_target_epsilon'),
        sa.CheckConstraint("target_delta > 0 AND target_delta < 1", name='valid_target_delta'),
        sa.CheckConstraint("progress >= 0 AND progress <= 1", name='valid_progress'),
    )
    
    # Create indexes for training jobs
    op.create_index('idx_training_jobs_status', 'training_jobs', ['status'])
    op.create_index('idx_training_job_user_status', 'training_jobs', ['user_id', 'status'])
    op.create_index('idx_training_job_created', 'training_jobs', ['created_at'])
    
    # Models table
    op.create_table(
        'models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('base_model', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('model_path', sa.String(500), nullable=False),
        sa.Column('config_path', sa.String(500)),
        sa.Column('tokenizer_path', sa.String(500)),
        sa.Column('training_job_id', postgresql.UUID(as_uuid=True)),
        sa.Column('epsilon_spent', sa.Float(), nullable=False),
        sa.Column('delta_value', sa.Float(), nullable=False),
        sa.Column('noise_multiplier', sa.Float()),
        sa.Column('max_grad_norm', sa.Float()),
        sa.Column('eval_loss', sa.Float()),
        sa.Column('eval_accuracy', sa.Float()),
        sa.Column('eval_perplexity', sa.Float()),
        sa.Column('model_size_mb', sa.Float()),
        sa.Column('is_deployed', sa.Boolean(), nullable=False, default=False),
        sa.Column('deployment_url', sa.String(500)),
        sa.Column('deployment_status', sa.String(50), default='not_deployed'),
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['training_job_id'], ['training_jobs.id']),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id']),
        sa.CheckConstraint("epsilon_spent >= 0", name='positive_epsilon'),
        sa.CheckConstraint("delta_value >= 0 AND delta_value <= 1", name='valid_delta'),
    )
    
    # Create indexes and constraints for models
    op.create_index('idx_model_owner_name', 'models', ['owner_id', 'name'])
    op.create_unique_constraint('unique_model_version', 'models', ['name', 'version', 'owner_id'])
    
    # Privacy budget entries table
    op.create_table(
        'privacy_budget_entries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('epsilon_spent', sa.Float(), nullable=False),
        sa.Column('delta_value', sa.Float(), nullable=False),
        sa.Column('operation', sa.String(100), nullable=False),
        sa.Column('training_job_id', postgresql.UUID(as_uuid=True)),
        sa.Column('session_id', sa.String(100)),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('step_number', sa.Integer()),
        sa.Column('epoch_number', sa.Integer()),
        sa.Column('batch_number', sa.Integer()),
        sa.Column('metadata', sa.JSON(), default={}),
        sa.Column('noise_multiplier', sa.Float()),
        sa.Column('sample_rate', sa.Float()),
        sa.Column('accounting_mode', sa.String(10)),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['training_job_id'], ['training_jobs.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.CheckConstraint("epsilon_spent >= 0", name='positive_epsilon_spent'),
        sa.CheckConstraint("delta_value >= 0 AND delta_value <= 1", name='valid_delta_value'),
    )
    
    # Create indexes for privacy budget entries
    op.create_index('idx_privacy_entry_job_step', 'privacy_budget_entries', ['training_job_id', 'step_number'])
    op.create_index('idx_privacy_entry_user_created', 'privacy_budget_entries', ['user_id', 'created_at'])
    
    # Audit logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=False),
        sa.Column('resource_id', sa.String(100)),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True)),
        sa.Column('actor_ip', sa.String(45)),
        sa.Column('user_agent', sa.String(500)),
        sa.Column('session_id', sa.String(100)),
        sa.Column('request_data', sa.JSON()),
        sa.Column('response_data', sa.JSON()),
        sa.Column('status_code', sa.Integer()),
        sa.Column('contains_pii', sa.Boolean(), nullable=False, default=False),
        sa.Column('data_classification', sa.String(50)),
        sa.Column('retention_until', sa.DateTime()),
        sa.Column('metadata', sa.JSON(), default={}),
        sa.Column('tags', sa.JSON(), default=[]),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
    )
    
    # Create indexes for audit logs
    op.create_index('idx_audit_log_event_type', 'audit_logs', ['event_type'])
    op.create_index('idx_audit_log_event_type_created', 'audit_logs', ['event_type', 'created_at'])
    op.create_index('idx_audit_log_user_created', 'audit_logs', ['user_id', 'created_at'])
    op.create_index('idx_audit_log_resource', 'audit_logs', ['resource_type', 'resource_id'])


def downgrade():
    """Drop all tables."""
    op.drop_table('audit_logs')
    op.drop_table('privacy_budget_entries')
    op.drop_table('models')
    op.drop_table('training_jobs')
    op.drop_table('datasets')
    op.drop_table('users')