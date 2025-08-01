#!/usr/bin/env python3
"""
Terragon Labs Autonomous SDLC - Intelligent Backlog Engine
Continuous value discovery and prioritization system for privacy-preserving-agent-finetuner
"""

import json
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging

class ValueDiscoveryEngine:
    """Advanced value discovery and scoring engine using WSJF + ICE + Technical Debt."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config = self._load_config(config_path)
        self.repo_root = Path(".")
        self.metrics_file = Path(".terragon/value-metrics.json") 
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load Terragon configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the value discovery engine."""
        logger = logging.getLogger("terragon-value-engine")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def discover_value_items(self) -> List[Dict]:
        """Comprehensive value discovery from multiple sources."""
        self.logger.info("Starting comprehensive value discovery...")
        
        items = []
        
        # Git history analysis
        if self.config['discovery']['sources']['git_history']['enabled']:
            items.extend(self._discover_from_git_history())
            
        # Static analysis
        if self.config['discovery']['sources']['static_analysis']['enabled']:
            items.extend(self._discover_from_static_analysis())
            
        # Code metrics analysis  
        if self.config['discovery']['sources']['code_metrics']['enabled']:
            items.extend(self._discover_from_code_metrics())
            
        # External sources
        if self.config['discovery']['sources']['external_apis']['enabled']:
            items.extend(self._discover_from_external_apis())
            
        self.logger.info(f"Discovered {len(items)} potential value items")
        return items
        
    def _discover_from_git_history(self) -> List[Dict]:
        """Discover work items from git history and code comments."""
        items = []
        keywords = self.config['discovery']['sources']['git_history']['keywords']
        
        try:
            # Search for TODO/FIXME/etc in codebase
            for keyword in keywords:
                cmd = ['grep', '-r', '-n', '--include=*.py', keyword, '.']
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            items.append(self._parse_code_comment(line, keyword))
                            
            # Analyze recent commits for debt indicators
            cmd = ['git', 'log', '--oneline', '-n', '50']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                debt_patterns = self.config['discovery']['sources']['git_history']['commit_patterns']
                for line in result.stdout.strip().split('\n'):
                    for pattern in debt_patterns:
                        if pattern.lower() in line.lower():
                            items.append(self._parse_commit_debt(line, pattern))
                            
        except Exception as e:
            self.logger.warning(f"Git history analysis failed: {e}")
            
        return items
        
    def _discover_from_static_analysis(self) -> List[Dict]:
        """Discover issues from static analysis tools."""
        items = []
        
        try:
            # Run ruff for code quality issues
            cmd = ['poetry', 'run', 'ruff', 'check', '--output-format=json', '.']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                try:
                    ruff_results = json.loads(result.stdout)
                    for issue in ruff_results:
                        items.append(self._parse_ruff_issue(issue))
                except json.JSONDecodeError:
                    pass
                    
            # Run security analysis
            cmd = ['poetry', 'run', 'bandit', '-r', 'privacy_finetuner/', '-f', 'json']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                try:
                    bandit_results = json.loads(result.stdout)
                    for issue in bandit_results.get('results', []):
                        items.append(self._parse_security_issue(issue))
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Static analysis failed: {e}")
            
        return items
        
    def _discover_from_code_metrics(self) -> List[Dict]:
        """Discover issues from code complexity and churn analysis.""" 
        items = []
        
        try:
            # Find complex functions (simplified heuristic)
            for py_file in self.repo_root.rglob('*.py'):
                if 'test' not in str(py_file) and '__pycache__' not in str(py_file):
                    with open(py_file, 'r') as f:
                        content = f.read()
                        complexity_score = self._estimate_complexity(content)
                        if complexity_score > self.config['discovery']['sources']['code_metrics']['complexity_threshold']:
                            items.append({
                                'id': f"complexity-{py_file.stem}",
                                'title': f"Reduce complexity in {py_file.name}",
                                'description': f"File has estimated complexity score of {complexity_score}",
                                'category': 'technical_debt',
                                'severity': 'medium',
                                'file_path': str(py_file),
                                'source': 'code_metrics',
                                'complexity_score': complexity_score
                            })
                            
        except Exception as e:
            self.logger.warning(f"Code metrics analysis failed: {e}")
            
        return items
        
    def _discover_from_external_apis(self) -> List[Dict]:
        """Discover items from external APIs (GitHub issues, vulnerabilities, etc)."""
        items = []
        
        # For now, simulate dependency updates as high-value items
        try:
            cmd = ['poetry', 'show', '--outdated']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n')[2:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            package = parts[0]
                            current = parts[1]
                            latest = parts[2]
                            items.append({
                                'id': f"dep-update-{package}",
                                'title': f"Update {package} from {current} to {latest}",
                                'description': f"Dependency update available",
                                'category': 'dependency_update',
                                'severity': 'low',
                                'source': 'external_apis',
                                'package_name': package,
                                'current_version': current,
                                'target_version': latest
                            })
                            
        except Exception as e:
            self.logger.warning(f"External API discovery failed: {e}")
            
        return items
        
    def calculate_composite_score(self, item: Dict) -> float:
        """Calculate composite value score using WSJF + ICE + Technical Debt."""
        
        # WSJF Components
        user_business_value = self._score_user_impact(item)
        time_criticality = self._score_urgency(item) 
        risk_reduction = self._score_risk_mitigation(item)
        opportunity_enablement = self._score_opportunity(item)
        
        cost_of_delay = (user_business_value + time_criticality + 
                        risk_reduction + opportunity_enablement)
        job_size = self._estimate_job_size(item)
        wsjf = cost_of_delay / max(job_size, 0.1)  # Avoid division by zero
        
        # ICE Components
        impact = self._score_business_impact(item)
        confidence = self._score_execution_confidence(item)
        ease = self._score_implementation_ease(item)
        ice = impact * confidence * ease
        
        # Technical Debt Score
        debt_impact = self._calculate_debt_cost(item)
        debt_interest = self._calculate_debt_growth(item)
        hotspot_multiplier = self._get_churn_complexity(item)
        tech_debt_score = (debt_impact + debt_interest) * hotspot_multiplier
        
        # Composite Score with adaptive weighting
        weights = self.config['scoring']['weights']
        composite = (
            weights['wsjf'] * self._normalize_score(wsjf) +
            weights['ice'] * self._normalize_score(ice) +
            weights['technicalDebt'] * self._normalize_score(tech_debt_score) +
            weights['security'] * self._get_security_boost(item)
        )
        
        # Apply category multipliers
        category_multiplier = self.config['scoring']['category_multipliers'].get(
            item.get('category', 'other'), 1.0
        )
        
        final_score = composite * category_multiplier
        
        # Store component scores for analysis
        item['scores'] = {
            'wsjf': wsjf,
            'ice': ice, 
            'technicalDebt': tech_debt_score,
            'composite': final_score,
            'components': {
                'user_business_value': user_business_value,
                'time_criticality': time_criticality,
                'risk_reduction': risk_reduction,
                'opportunity_enablement': opportunity_enablement,
                'impact': impact,
                'confidence': confidence,
                'ease': ease,
                'debt_impact': debt_impact,
                'category_multiplier': category_multiplier
            }
        }
        
        return final_score
        
    def select_next_best_value(self, items: List[Dict]) -> Optional[Dict]:
        """Select the highest value item that meets execution criteria."""
        
        # Score all items
        for item in items:
            item['composite_score'] = self.calculate_composite_score(item)
            
        # Sort by score descending
        prioritized = sorted(items, key=lambda x: x['composite_score'], reverse=True)
        
        # Apply selection filters
        for item in prioritized:
            # Check minimum score threshold
            if item['composite_score'] < self.config['scoring']['thresholds']['minScore']:
                continue
                
            # Check risk threshold
            if self._assess_risk(item) > self.config['scoring']['thresholds']['maxRisk']:
                continue
                
            # Check dependencies (simplified check)
            if not self._are_dependencies_met(item):
                continue
                
            # Found our winner
            return item
            
        return None
        
    # Scoring helper methods (simplified implementations)
    def _score_user_impact(self, item: Dict) -> float:
        """Score user/business impact (1-10)."""
        category_scores = {
            'security': 9.0,
            'compliance': 8.5, 
            'performance': 7.0,
            'technical_debt': 6.0,
            'feature': 8.0,
            'documentation': 4.0,
            'dependency_update': 3.0
        }
        return category_scores.get(item.get('category', 'other'), 5.0)
        
    def _score_urgency(self, item: Dict) -> float:
        """Score time criticality (1-10)."""
        severity_scores = {
            'critical': 10.0,
            'high': 8.0,
            'medium': 5.0,
            'low': 2.0
        }
        return severity_scores.get(item.get('severity', 'medium'), 5.0)
        
    def _score_risk_mitigation(self, item: Dict) -> float:
        """Score risk reduction value (1-10).""" 
        if item.get('category') == 'security':
            return 9.0
        elif item.get('category') == 'technical_debt':
            return 6.0
        return 3.0
        
    def _score_opportunity(self, item: Dict) -> float:
        """Score opportunity enablement (1-10)."""
        if item.get('category') == 'feature':
            return 8.0
        elif item.get('category') == 'performance':
            return 6.0
        return 3.0
        
    def _estimate_job_size(self, item: Dict) -> float:
        """Estimate effort in story points (1-13)."""
        category_effort = {
            'security': 5.0,
            'technical_debt': 8.0,
            'performance': 8.0,
            'feature': 13.0,
            'documentation': 3.0,
            'dependency_update': 2.0
        }
        return category_effort.get(item.get('category', 'other'), 5.0)
        
    def _score_business_impact(self, item: Dict) -> float:
        """Score business impact for ICE (1-10)."""
        return self._score_user_impact(item)
        
    def _score_execution_confidence(self, item: Dict) -> float:
        """Score confidence in successful execution (1-10)."""
        category_confidence = {
            'documentation': 9.0,
            'dependency_update': 8.0, 
            'technical_debt': 7.0,
            'security': 6.0,
            'performance': 5.0,
            'feature': 4.0
        }
        return category_confidence.get(item.get('category', 'other'), 6.0)
        
    def _score_implementation_ease(self, item: Dict) -> float:
        """Score implementation ease (1-10)."""
        return 11.0 - self._estimate_job_size(item)  # Inverse of complexity
        
    def _calculate_debt_cost(self, item: Dict) -> float:
        """Calculate technical debt cost."""
        if item.get('category') == 'technical_debt':
            return item.get('complexity_score', 5.0)
        return 1.0
        
    def _calculate_debt_growth(self, item: Dict) -> float:
        """Calculate debt interest (future cost if not addressed)."""
        if item.get('category') == 'technical_debt':
            return 3.0
        return 0.5
        
    def _get_churn_complexity(self, item: Dict) -> float:
        """Get hotspot multiplier based on file churn."""
        # Simplified: assume some files are hotspots
        file_path = item.get('file_path', '')
        if any(hot in file_path for hot in ['core/', 'api/', 'trainer.py']):
            return 2.0
        return 1.0
        
    def _get_security_boost(self, item: Dict) -> float:
        """Get security priority boost."""
        if item.get('category') == 'security':
            return self.config['scoring']['thresholds']['securityBoost']
        return 1.0
        
    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-100 range."""
        return min(max(score, 0), 100)
        
    def _assess_risk(self, item: Dict) -> float:
        """Assess execution risk (0-1)."""
        category_risk = {
            'documentation': 0.1,
            'dependency_update': 0.3,
            'technical_debt': 0.5,
            'performance': 0.6,
            'security': 0.4,
            'feature': 0.8
        }
        return category_risk.get(item.get('category', 'other'), 0.5)
        
    def _are_dependencies_met(self, item: Dict) -> bool:
        """Check if item dependencies are satisfied."""
        # Simplified check - assume all dependencies met for now
        return True
        
    # Parsing helper methods
    def _parse_code_comment(self, line: str, keyword: str) -> Dict:
        """Parse a code comment containing debt indicators."""
        parts = line.split(':', 2)
        file_path = parts[0] if len(parts) > 0 else 'unknown'
        line_num = parts[1] if len(parts) > 1 else '0'
        comment = parts[2] if len(parts) > 2 else line
        
        return {
            'id': f"{keyword.lower()}-{hash(line) % 10000}",
            'title': f"Address {keyword} in {Path(file_path).name}",
            'description': comment.strip(),
            'category': 'technical_debt',
            'severity': 'medium' if keyword in ['TODO', 'FIXME'] else 'high',
            'file_path': file_path,
            'line_number': line_num,
            'source': 'git_history',
            'keyword': keyword
        }
        
    def _parse_commit_debt(self, commit_line: str, pattern: str) -> Dict:
        """Parse commit indicating technical debt."""
        commit_hash = commit_line.split()[0]
        commit_msg = ' '.join(commit_line.split()[1:])
        
        return {
            'id': f"debt-commit-{commit_hash}",
            'title': f"Review and improve: {commit_msg[:50]}...",
            'description': f"Commit suggests technical debt: {commit_msg}",
            'category': 'technical_debt',
            'severity': 'medium',
            'source': 'git_history',
            'commit_hash': commit_hash,
            'debt_pattern': pattern
        }
        
    def _parse_ruff_issue(self, issue: Dict) -> Dict:
        """Parse ruff static analysis issue."""
        return {
            'id': f"ruff-{issue.get('code', 'unknown')}-{hash(str(issue)) % 10000}",
            'title': f"Fix {issue.get('code', 'code issue')}: {issue.get('message', 'Unknown issue')[:50]}",
            'description': issue.get('message', 'Ruff static analysis issue'),
            'category': 'technical_debt',
            'severity': 'low',
            'file_path': issue.get('filename', 'unknown'),
            'line_number': issue.get('location', {}).get('row', 0),
            'source': 'static_analysis',
            'tool': 'ruff',
            'rule_code': issue.get('code')
        }
        
    def _parse_security_issue(self, issue: Dict) -> Dict:
        """Parse bandit security issue."""
        return {
            'id': f"security-{issue.get('test_id', 'unknown')}-{hash(str(issue)) % 10000}",
            'title': f"Security: {issue.get('issue_text', 'Security issue')[:50]}",
            'description': issue.get('issue_text', 'Security vulnerability detected'),
            'category': 'security',
            'severity': issue.get('issue_severity', 'medium').lower(),
            'file_path': issue.get('filename', 'unknown'),
            'line_number': issue.get('line_number', 0),
            'source': 'static_analysis',
            'tool': 'bandit',
            'test_id': issue.get('test_id'),
            'confidence': issue.get('issue_confidence', 'medium')
        }
        
    def _estimate_complexity(self, content: str) -> int:
        """Estimate code complexity (simplified McCabe-like metric)."""
        # Count complexity indicators
        complexity = 1  # Base complexity
        
        # Branching statements
        complexity += len(re.findall(r'\b(if|elif|else|for|while|try|except|finally)\b', content))
        
        # Logical operators
        complexity += len(re.findall(r'\b(and|or|not)\b', content))
        
        # Function definitions (add 1 per function)
        complexity += len(re.findall(r'\bdef\s+\w+', content))
        
        return complexity
        
    def update_metrics(self, executed_item: Dict, actual_impact: Dict) -> None:
        """Update metrics based on execution results."""
        try:
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
                
            # Add to execution history
            execution_record = {
                'timestamp': datetime.now().isoformat() + 'Z',
                'itemId': executed_item['id'],
                'title': executed_item['title'],
                'category': executed_item.get('category', 'unknown'),
                'scores': executed_item.get('scores', {}),
                'estimatedEffort': executed_item.get('scores', {}).get('components', {}).get('estimated_effort', 0),
                'actualEffort': actual_impact.get('actual_effort', 0),
                'status': 'completed',
                'impact': actual_impact
            }
            
            metrics['execution_history'].append(execution_record)
            
            # Update learning metrics
            predicted_impact = executed_item.get('composite_score', 0)
            actual_impact_score = actual_impact.get('impact_score', predicted_impact)
            accuracy = min(actual_impact_score / max(predicted_impact, 0.1), 2.0)  # Cap at 2x
            
            current_accuracy = metrics['learning_metrics']['estimation_accuracy']
            metrics['learning_metrics']['estimation_accuracy'] = (current_accuracy * 0.9 + accuracy * 0.1)
            
            # Update repository improvements
            metrics['repository_improvements']['technical_debt_reduction_percent'] += actual_impact.get('debt_reduction', 0)
            
            # Save updated metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")


def main():
    """Main execution function for value discovery."""
    engine = ValueDiscoveryEngine()
    
    # Discover value items
    items = engine.discover_value_items()
    
    if not items:
        engine.logger.info("No value items discovered")
        return
        
    # Select next best value item
    next_item = engine.select_next_best_value(items)
    
    if next_item:
        engine.logger.info(f"Next best value item: {next_item['title']} (score: {next_item['composite_score']:.1f})")
        print(json.dumps(next_item, indent=2))
    else:
        engine.logger.info("No items meet execution criteria")
        
    # Update backlog file
    backlog = {
        'last_updated': datetime.now().isoformat() + 'Z',
        'next_best_item': next_item,
        'discovered_items': len(items),
        'all_items': sorted(items, key=lambda x: x.get('composite_score', 0), reverse=True)[:20]  # Top 20
    }
    
    with open('.terragon/current-backlog.json', 'w') as f:
        json.dump(backlog, f, indent=2)


if __name__ == '__main__':
    main()