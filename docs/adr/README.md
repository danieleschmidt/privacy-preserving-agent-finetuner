# Architecture Decision Records (ADR)

This directory contains the Architecture Decision Records for the privacy-preserving-agent-finetuner project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Template

Use the template in `adr-template.md` for new ADRs.

## ADR Lifecycle

1. **Proposed** - The ADR is proposed and under discussion
2. **Accepted** - The ADR is accepted and should be implemented
3. **Deprecated** - The ADR is no longer relevant but kept for historical context
4. **Superseded** - The ADR has been replaced by a newer ADR

## Current ADRs

| Number | Title | Status | Date |
|--------|-------|--------|------|
| [ADR-001](001-differential-privacy-framework.md) | Differential Privacy Framework Choice | Accepted | 2025-08-01 |
| [ADR-002](002-federated-learning-architecture.md) | Federated Learning Architecture | Accepted | 2025-08-01 |
| [ADR-003](003-context-protection-strategies.md) | Context Protection Implementation | Accepted | 2025-08-01 |

## Creating a New ADR

1. Copy `adr-template.md` to a new file: `NNN-short-title.md`
2. Fill in the template with your decision details
3. Submit for review via pull request
4. Update this README with the new ADR entry