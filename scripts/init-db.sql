-- Database initialization script for Privacy-Preserving Agent Fine-Tuner
-- Creates necessary databases, users, and security configurations

-- Create development database
CREATE DATABASE IF NOT EXISTS privacy_finetuner_dev;

-- Create test database
CREATE DATABASE IF NOT EXISTS privacy_finetuner_test;

-- Create production database (if running locally)
CREATE DATABASE IF NOT EXISTS privacy_finetuner_prod;

-- Create application user with limited privileges
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'privacy_app') THEN
      
      CREATE ROLE privacy_app LOGIN PASSWORD 'secure_app_password_change_in_production';
   END IF;
END
$do$;

-- Grant privileges to application user
GRANT CONNECT ON DATABASE privacy_finetuner_dev TO privacy_app;
GRANT CONNECT ON DATABASE privacy_finetuner_test TO privacy_app;
GRANT CONNECT ON DATABASE privacy_finetuner_prod TO privacy_app;

-- Switch to development database
\c privacy_finetuner_dev;

-- Create schema for privacy tracking
CREATE SCHEMA IF NOT EXISTS privacy_audit;
CREATE SCHEMA IF NOT EXISTS privacy_metrics;

-- Grant schema access
GRANT USAGE ON SCHEMA privacy_audit TO privacy_app;
GRANT USAGE ON SCHEMA privacy_metrics TO privacy_app;
GRANT CREATE ON SCHEMA privacy_audit TO privacy_app;
GRANT CREATE ON SCHEMA privacy_metrics TO privacy_app;

-- Privacy audit table for tracking privacy budget consumption
CREATE TABLE IF NOT EXISTS privacy_audit.budget_consumption (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    user_id VARCHAR(255),
    operation_type VARCHAR(100) NOT NULL,
    epsilon_consumed DECIMAL(10, 8) NOT NULL CHECK (epsilon_consumed >= 0),
    delta_consumed DECIMAL(15, 12) NOT NULL CHECK (delta_consumed >= 0),
    privacy_mechanism VARCHAR(100) NOT NULL,
    dataset_hash VARCHAR(64),
    model_identifier VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    
    -- Indexes for efficient querying
    CONSTRAINT valid_privacy_params CHECK (
        epsilon_consumed >= 0 AND delta_consumed >= 0
    )
);

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_budget_consumption_session ON privacy_audit.budget_consumption(session_id);
CREATE INDEX IF NOT EXISTS idx_budget_consumption_timestamp ON privacy_audit.budget_consumption(timestamp);
CREATE INDEX IF NOT EXISTS idx_budget_consumption_user ON privacy_audit.budget_consumption(user_id);

-- Privacy metrics aggregation table
CREATE TABLE IF NOT EXISTS privacy_metrics.daily_summary (
    date DATE PRIMARY KEY,
    total_epsilon_consumed DECIMAL(15, 8) NOT NULL DEFAULT 0,
    total_delta_consumed DECIMAL(20, 15) NOT NULL DEFAULT 0,
    unique_sessions INTEGER NOT NULL DEFAULT 0,
    unique_users INTEGER NOT NULL DEFAULT 0,
    operations_count INTEGER NOT NULL DEFAULT 0,
    avg_epsilon_per_operation DECIMAL(10, 8),
    max_epsilon_per_session DECIMAL(10, 8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Training sessions table
CREATE TABLE IF NOT EXISTS privacy_audit.training_sessions (
    session_id UUID PRIMARY KEY,
    user_id VARCHAR(255),
    model_name VARCHAR(255) NOT NULL,
    dataset_name VARCHAR(255),
    privacy_config JSONB NOT NULL,
    total_epsilon_budget DECIMAL(10, 8) NOT NULL,
    total_delta_budget DECIMAL(15, 12) NOT NULL,
    consumed_epsilon DECIMAL(10, 8) DEFAULT 0,
    consumed_delta DECIMAL(15, 12) DEFAULT 0,
    training_steps INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_session_privacy CHECK (
        consumed_epsilon <= total_epsilon_budget AND
        consumed_delta <= total_delta_budget
    )
);

-- Model checkpoints table
CREATE TABLE IF NOT EXISTS privacy_audit.model_checkpoints (
    checkpoint_id UUID PRIMARY KEY,
    session_id UUID REFERENCES privacy_audit.training_sessions(session_id),
    checkpoint_path VARCHAR(500) NOT NULL,
    epsilon_at_checkpoint DECIMAL(10, 8) NOT NULL,
    delta_at_checkpoint DECIMAL(15, 12) NOT NULL,
    training_step INTEGER NOT NULL,
    model_metrics JSONB,
    privacy_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Compliance audit table
CREATE TABLE IF NOT EXISTS privacy_audit.compliance_events (
    event_id UUID PRIMARY KEY,
    session_id UUID REFERENCES privacy_audit.training_sessions(session_id),
    compliance_framework VARCHAR(50) NOT NULL, -- GDPR, HIPAA, CCPA, etc.
    event_type VARCHAR(100) NOT NULL,
    event_details JSONB NOT NULL,
    severity VARCHAR(20) DEFAULT 'info',
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Data subject rights requests (GDPR compliance)
CREATE TABLE IF NOT EXISTS privacy_audit.data_subject_requests (
    request_id UUID PRIMARY KEY,
    request_type VARCHAR(50) NOT NULL, -- access, rectification, erasure, etc.
    data_subject_id VARCHAR(255) NOT NULL,
    request_details JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    requested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    processor_id VARCHAR(255),
    processing_notes TEXT
);

-- Function to automatically update privacy metrics
CREATE OR REPLACE FUNCTION update_privacy_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update daily summary
    INSERT INTO privacy_metrics.daily_summary (
        date, 
        total_epsilon_consumed, 
        total_delta_consumed,
        unique_sessions,
        unique_users,
        operations_count
    )
    VALUES (
        CURRENT_DATE,
        NEW.epsilon_consumed,
        NEW.delta_consumed,
        1,
        CASE WHEN NEW.user_id IS NOT NULL THEN 1 ELSE 0 END,
        1
    )
    ON CONFLICT (date) DO UPDATE SET
        total_epsilon_consumed = privacy_metrics.daily_summary.total_epsilon_consumed + NEW.epsilon_consumed,
        total_delta_consumed = privacy_metrics.daily_summary.total_delta_consumed + NEW.delta_consumed,
        operations_count = privacy_metrics.daily_summary.operations_count + 1,
        updated_at = CURRENT_TIMESTAMP;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update metrics
DROP TRIGGER IF EXISTS budget_consumption_metrics_trigger ON privacy_audit.budget_consumption;
CREATE TRIGGER budget_consumption_metrics_trigger
    AFTER INSERT ON privacy_audit.budget_consumption
    FOR EACH ROW EXECUTE FUNCTION update_privacy_metrics();

-- Function to check privacy budget limits
CREATE OR REPLACE FUNCTION check_privacy_budget()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if the new consumption would exceed session budget
    IF (SELECT consumed_epsilon + NEW.epsilon_consumed 
        FROM privacy_audit.training_sessions 
        WHERE session_id = NEW.session_id) > 
       (SELECT total_epsilon_budget 
        FROM privacy_audit.training_sessions 
        WHERE session_id = NEW.session_id) THEN
        RAISE EXCEPTION 'Privacy budget exceeded: epsilon limit would be surpassed';
    END IF;
    
    IF (SELECT consumed_delta + NEW.delta_consumed 
        FROM privacy_audit.training_sessions 
        WHERE session_id = NEW.session_id) > 
       (SELECT total_delta_budget 
        FROM privacy_audit.training_sessions 
        WHERE session_id = NEW.session_id) THEN
        RAISE EXCEPTION 'Privacy budget exceeded: delta limit would be surpassed';
    END IF;
    
    -- Update session consumption
    UPDATE privacy_audit.training_sessions 
    SET consumed_epsilon = consumed_epsilon + NEW.epsilon_consumed,
        consumed_delta = consumed_delta + NEW.delta_consumed
    WHERE session_id = NEW.session_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to enforce privacy budget limits
DROP TRIGGER IF EXISTS budget_limit_trigger ON privacy_audit.budget_consumption;
CREATE TRIGGER budget_limit_trigger
    BEFORE INSERT ON privacy_audit.budget_consumption
    FOR EACH ROW EXECUTE FUNCTION check_privacy_budget();

-- Create views for common queries
CREATE OR REPLACE VIEW privacy_audit.active_sessions AS
SELECT 
    session_id,
    user_id,
    model_name,
    total_epsilon_budget,
    consumed_epsilon,
    (total_epsilon_budget - consumed_epsilon) AS remaining_epsilon,
    (consumed_epsilon / total_epsilon_budget) * 100 AS epsilon_used_percent,
    training_steps,
    started_at
FROM privacy_audit.training_sessions
WHERE status = 'active';

-- View for daily privacy consumption
CREATE OR REPLACE VIEW privacy_metrics.daily_consumption AS
SELECT 
    date,
    total_epsilon_consumed,
    total_delta_consumed,
    operations_count,
    ROUND(total_epsilon_consumed / operations_count, 6) AS avg_epsilon_per_operation
FROM privacy_metrics.daily_summary
ORDER BY date DESC;

-- Security: Revoke public access and grant specific permissions
REVOKE ALL ON ALL TABLES IN SCHEMA privacy_audit FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA privacy_metrics FROM PUBLIC;

-- Grant necessary permissions to application user
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA privacy_audit TO privacy_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA privacy_metrics TO privacy_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA privacy_audit TO privacy_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA privacy_metrics TO privacy_app;

-- Enable row level security (optional, for multi-tenant setups)
-- ALTER TABLE privacy_audit.budget_consumption ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE privacy_audit.training_sessions ENABLE ROW LEVEL SECURITY;

-- Create policy for user isolation (uncomment if needed)
-- CREATE POLICY user_isolation_policy ON privacy_audit.training_sessions
--     FOR ALL TO privacy_app
--     USING (user_id = current_setting('app.current_user_id', true));

-- Repeat for test database
\c privacy_finetuner_test;

-- Recreate the same structure for test database (abbreviated)
CREATE SCHEMA IF NOT EXISTS privacy_audit;
CREATE SCHEMA IF NOT EXISTS privacy_metrics;
GRANT USAGE ON SCHEMA privacy_audit TO privacy_app;
GRANT USAGE ON SCHEMA privacy_metrics TO privacy_app;
GRANT CREATE ON SCHEMA privacy_audit TO privacy_app;
GRANT CREATE ON SCHEMA privacy_metrics TO privacy_app;

-- Basic tables for testing (minimal structure)
CREATE TABLE IF NOT EXISTS privacy_audit.budget_consumption (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    epsilon_consumed DECIMAL(10, 8) NOT NULL,
    delta_consumed DECIMAL(15, 12) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS privacy_audit.training_sessions (
    session_id UUID PRIMARY KEY,
    total_epsilon_budget DECIMAL(10, 8) NOT NULL,
    consumed_epsilon DECIMAL(10, 8) DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active'
);

-- Grant permissions for test database
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA privacy_audit TO privacy_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA privacy_metrics TO privacy_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA privacy_audit TO privacy_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA privacy_metrics TO privacy_app;

-- Return to main database
\c privacy_finetuner_dev;

-- Insert sample data for development (optional)
-- Uncomment the following for development environment setup

/*
-- Sample training session
INSERT INTO privacy_audit.training_sessions (
    session_id, 
    user_id, 
    model_name, 
    dataset_name, 
    privacy_config,
    total_epsilon_budget, 
    total_delta_budget
) VALUES (
    gen_random_uuid(),
    'dev_user_1',
    'distilbert-base-uncased',
    'sample_dataset',
    '{"epsilon": 1.0, "delta": 1e-5, "noise_multiplier": 0.5}',
    1.0,
    0.00001
);

-- Sample budget consumption
INSERT INTO privacy_audit.budget_consumption (
    session_id,
    user_id,
    operation_type,
    epsilon_consumed,
    delta_consumed,
    privacy_mechanism,
    metadata
) VALUES (
    (SELECT session_id FROM privacy_audit.training_sessions LIMIT 1),
    'dev_user_1',
    'dp_sgd_step',
    0.01,
    0.000001,
    'opacus',
    '{"batch_size": 32, "learning_rate": 0.001}'
);
*/

-- Final permissions check
\du privacy_app;