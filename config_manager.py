"""
Production Configuration Management System for LogNarrator AI

Real configuration management including:
- Rule customization
- Threshold management
- User preferences
- System settings
- Environment-specific configs

Production functionality only.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import yaml

from database_manager import get_database_manager

logger = logging.getLogger(__name__)


@dataclass
class ConfigRule:
    """Configuration rule definition"""
    name: str
    category: str
    value: Any
    value_type: str  # 'string', 'int', 'float', 'bool', 'json', 'list'
    description: str
    default_value: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    requires_restart: bool = False
    last_modified: Optional[datetime] = None
    modified_by: Optional[str] = None


class ConfigManager:
    """Production configuration management"""
    
    def __init__(self, config_file: str = "lognarrator_config.yaml"):
        self.config_file = Path(config_file)
        self.db_manager = get_database_manager()
        
        # Default configuration
        self.default_config = {
            # Analysis Configuration
            'analysis': {
                'confidence_threshold': 0.6,
                'min_data_points': 5,
                'max_entries_per_session': 10000,
                'enable_multiline_parsing': True,
                'enable_json_extraction': True,
                'enable_xml_extraction': True,
                'enable_stack_trace_parsing': True,
                'anomaly_detection_threshold': 2.0,
                'trend_analysis_window_days': 30,
                'correlation_threshold': 0.3,
                'learning_enabled': True,
                'auto_pattern_updates': True,
                'max_pattern_cache_size': 1000
            },
            
            # Performance Configuration
            'performance': {
                'max_concurrent_analyses': 4,
                'cache_ttl_hours': 24,
                'db_connection_pool_size': 10,
                'processing_timeout_seconds': 300,
                'memory_limit_mb': 2048,
                'temp_file_cleanup_hours': 24,
                'log_level': 'INFO',
                'enable_performance_monitoring': True,
                'profile_analysis_performance': False
            },
            
            # Export Configuration
            'export': {
                'default_format': 'pdf',
                'include_charts': True,
                'include_trends': True,
                'include_correlations': True,
                'pdf_page_size': 'A4',
                'excel_include_formulas': True,
                'json_indent': 2,
                'csv_delimiter': ',',
                'custom_branding_enabled': False,
                'watermark_enabled': False,
                'compression_enabled': True
            },
            
            # Alert Configuration
            'alerts': {
                'enabled': True,
                'email_notifications': False,
                'slack_notifications': False,
                'webhook_notifications': False,
                'degradation_threshold_percent': 20,
                'critical_threshold_percent': 50,
                'anomaly_alert_threshold': 3.0,
                'alert_cooldown_minutes': 60,
                'max_alerts_per_hour': 10,
                'alert_severity_levels': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            },
            
            # UI Configuration
            'ui': {
                'theme': 'light',
                'auto_refresh_interval': 30,
                'max_display_entries': 1000,
                'enable_real_time_updates': True,
                'show_confidence_scores': True,
                'show_processing_times': True,
                'enable_advanced_filters': True,
                'default_time_range': 'last_24_hours',
                'chart_refresh_interval': 60
            },
            
            # Security Configuration
            'security': {
                'enable_authentication': False,
                'session_timeout_minutes': 120,
                'max_login_attempts': 3,
                'password_min_length': 8,
                'require_https': False,
                'enable_audit_logging': True,
                'data_encryption_enabled': False,
                'backup_encryption_enabled': False
            },
            
            # Data Retention
            'retention': {
                'analysis_history_days': 90,
                'log_entries_days': 30,
                'trend_data_days': 365,
                'system_metrics_days': 30,
                'user_feedback_days': 180,
                'export_files_days': 7,
                'temp_files_hours': 24,
                'auto_cleanup_enabled': True,
                'cleanup_schedule': '0 2 * * *'  # Daily at 2 AM
            },
            
            # Integration Configuration
            'integrations': {
                'enable_api': False,
                'api_port': 8080,
                'api_host': '127.0.0.1',
                'api_rate_limit': 100,
                'enable_webhooks': False,
                'webhook_timeout_seconds': 30,
                'enable_file_monitoring': False,
                'file_monitor_interval': 60,
                'enable_database_sync': False
            }
        }
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize rules
        self._init_config_rules()
        
        logger.info("Configuration manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and database"""
        
        config = self.default_config.copy()
        
        # Load from file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.suffix.lower() == '.yaml':
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                
                # Merge with default config
                config = self._merge_configs(config, file_config)
                
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        
        # Load from database (overrides file config)
        try:
            db_config = self._load_db_config()
            config = self._merge_configs(config, db_config)
        except Exception as e:
            logger.error(f"Failed to load database config: {e}")
        
        return config
    
    def _load_db_config(self) -> Dict[str, Any]:
        """Load configuration from database"""
        
        config = {}
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute('SELECT key, value, value_type FROM system_config')
            
            for row in cursor.fetchall():
                key, value, value_type = row['key'], row['value'], row['value_type']
                
                # Convert value based on type
                if value_type == 'int':
                    parsed_value = int(value)
                elif value_type == 'float':
                    parsed_value = float(value)
                elif value_type == 'bool':
                    parsed_value = value.lower() in ('true', '1', 'yes', 'on')
                elif value_type == 'json':
                    parsed_value = json.loads(value)
                elif value_type == 'list':
                    parsed_value = json.loads(value)
                else:
                    parsed_value = value
                
                # Set nested config value
                self._set_nested_config(config, key, parsed_value)
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries"""
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _set_nested_config(self, config: Dict[str, Any], key: str, value: Any):
        """Set nested configuration value using dot notation"""
        
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _init_config_rules(self):
        """Initialize configuration rules from current config"""
        
        self.config_rules = {}
        
        # Analysis rules
        self.config_rules.update({
            'analysis.confidence_threshold': ConfigRule(
                name='Confidence Threshold',
                category='analysis',
                value=self.config['analysis']['confidence_threshold'],
                value_type='float',
                description='Minimum confidence score for accepting categorization results',
                default_value=0.6,
                min_value=0.0,
                max_value=1.0
            ),
            'analysis.learning_enabled': ConfigRule(
                name='Learning Enabled',
                category='analysis',
                value=self.config['analysis']['learning_enabled'],
                value_type='bool',
                description='Enable automatic pattern learning from user feedback',
                default_value=True
            ),
            'analysis.anomaly_detection_threshold': ConfigRule(
                name='Anomaly Detection Threshold',
                category='analysis',
                value=self.config['analysis']['anomaly_detection_threshold'],
                value_type='float',
                description='Standard deviation threshold for anomaly detection',
                default_value=2.0,
                min_value=1.0,
                max_value=5.0
            )
        })
        
        # Performance rules
        self.config_rules.update({
            'performance.max_concurrent_analyses': ConfigRule(
                name='Max Concurrent Analyses',
                category='performance',
                value=self.config['performance']['max_concurrent_analyses'],
                value_type='int',
                description='Maximum number of concurrent analysis processes',
                default_value=4,
                min_value=1,
                max_value=16,
                requires_restart=True
            ),
            'performance.processing_timeout_seconds': ConfigRule(
                name='Processing Timeout',
                category='performance',
                value=self.config['performance']['processing_timeout_seconds'],
                value_type='int',
                description='Timeout for analysis processing in seconds',
                default_value=300,
                min_value=60,
                max_value=3600
            )
        })
        
        # Alert rules
        self.config_rules.update({
            'alerts.degradation_threshold_percent': ConfigRule(
                name='Degradation Threshold',
                category='alerts',
                value=self.config['alerts']['degradation_threshold_percent'],
                value_type='float',
                description='Percentage degradation threshold for alerts',
                default_value=20.0,
                min_value=5.0,
                max_value=100.0
            ),
            'alerts.max_alerts_per_hour': ConfigRule(
                name='Max Alerts Per Hour',
                category='alerts',
                value=self.config['alerts']['max_alerts_per_hour'],
                value_type='int',
                description='Maximum number of alerts to send per hour',
                default_value=10,
                min_value=1,
                max_value=100
            )
        })
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        
        keys = key.split('.')
        current = self.config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set_config(self, key: str, value: Any, save_to_db: bool = True, 
                  modified_by: str = 'system') -> bool:
        """Set configuration value"""
        
        try:
            # Validate against rule if exists
            if key in self.config_rules:
                rule = self.config_rules[key]
                if not self._validate_config_value(rule, value):
                    logger.error(f"Invalid value for config {key}: {value}")
                    return False
            
            # Set in memory
            self._set_nested_config(self.config, key, value)
            
            # Save to database if requested
            if save_to_db:
                self._save_config_to_db(key, value, modified_by)
            
            # Update rule
            if key in self.config_rules:
                self.config_rules[key].value = value
                self.config_rules[key].last_modified = datetime.now()
                self.config_rules[key].modified_by = modified_by
            
            logger.info(f"Configuration updated: {key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False
    
    def _validate_config_value(self, rule: ConfigRule, value: Any) -> bool:
        """Validate configuration value against rule"""
        
        # Type validation
        if rule.value_type == 'int':
            if not isinstance(value, int):
                return False
        elif rule.value_type == 'float':
            if not isinstance(value, (int, float)):
                return False
        elif rule.value_type == 'bool':
            if not isinstance(value, bool):
                return False
        elif rule.value_type == 'string':
            if not isinstance(value, str):
                return False
        elif rule.value_type in ['json', 'list']:
            # JSON and list values are more flexible
            pass
        
        # Range validation
        if rule.min_value is not None and value < rule.min_value:
            return False
        if rule.max_value is not None and value > rule.max_value:
            return False
        
        # Allowed values validation
        if rule.allowed_values is not None and value not in rule.allowed_values:
            return False
        
        return True
    
    def _save_config_to_db(self, key: str, value: Any, modified_by: str):
        """Save configuration to database"""
        
        # Determine value type
        if isinstance(value, bool):
            value_type = 'bool'
            value_str = str(value).lower()
        elif isinstance(value, int):
            value_type = 'int'
            value_str = str(value)
        elif isinstance(value, float):
            value_type = 'float'
            value_str = str(value)
        elif isinstance(value, (dict, list)):
            value_type = 'json'
            value_str = json.dumps(value)
        else:
            value_type = 'string'
            value_str = str(value)
        
        # Get description from rule
        description = ""
        if key in self.config_rules:
            description = self.config_rules[key].description
        
        # Save to database
        self.db_manager.set_config(key, value_str, value_type, description)
    
    def get_config_rules(self, category: Optional[str] = None) -> List[ConfigRule]:
        """Get configuration rules, optionally filtered by category"""
        
        rules = list(self.config_rules.values())
        
        if category:
            rules = [rule for rule in rules if rule.category == category]
        
        return sorted(rules, key=lambda r: r.name)
    
    def get_categories(self) -> List[str]:
        """Get all configuration categories"""
        
        categories = set(rule.category for rule in self.config_rules.values())
        return sorted(categories)
    
    def reset_config(self, key: str) -> bool:
        """Reset configuration to default value"""
        
        if key in self.config_rules:
            rule = self.config_rules[key]
            return self.set_config(key, rule.default_value, modified_by='system')
        
        return False
    
    def reset_category(self, category: str) -> bool:
        """Reset all configuration in a category to defaults"""
        
        success = True
        
        for key, rule in self.config_rules.items():
            if rule.category == category:
                if not self.reset_config(key):
                    success = False
        
        return success
    
    def save_config_file(self, backup_existing: bool = True) -> bool:
        """Save current configuration to file"""
        
        try:
            # Backup existing file
            if backup_existing and self.config_file.exists():
                backup_path = self.config_file.with_suffix(f'.backup.{int(datetime.now().timestamp())}')
                self.config_file.rename(backup_path)
            
            # Save configuration
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.suffix.lower() == '.yaml':
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")
            return False
    
    def export_config(self, output_path: str, format_type: str = 'json') -> bool:
        """Export configuration to file"""
        
        try:
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'configuration': self.config,
                'rules': {key: asdict(rule) for key, rule in self.config_rules.items()}
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if format_type.lower() == 'yaml':
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Configuration exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export config: {e}")
            return False
    
    def import_config(self, input_path: str, merge: bool = True) -> bool:
        """Import configuration from file"""
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                if input_path.endswith('.yaml'):
                    import_data = yaml.safe_load(f)
                else:
                    import_data = json.load(f)
            
            imported_config = import_data.get('configuration', {})
            
            if merge:
                # Merge with existing config
                self.config = self._merge_configs(self.config, imported_config)
            else:
                # Replace config
                self.config = imported_config
            
            # Update rules
            self._init_config_rules()
            
            logger.info(f"Configuration imported from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import config: {e}")
            return False
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate current configuration"""
        
        validation_errors = {}
        
        for key, rule in self.config_rules.items():
            errors = []
            
            current_value = self.get_config(key)
            
            if not self._validate_config_value(rule, current_value):
                errors.append(f"Invalid value: {current_value}")
            
            if errors:
                validation_errors[key] = errors
        
        return validation_errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        
        summary = {
            'total_rules': len(self.config_rules),
            'categories': self.get_categories(),
            'last_modified': None,
            'requires_restart': [],
            'validation_errors': self.validate_config()
        }
        
        # Find last modified
        last_modified_times = [rule.last_modified for rule in self.config_rules.values() if rule.last_modified]
        if last_modified_times:
            summary['last_modified'] = max(last_modified_times).isoformat()
        
        # Find rules requiring restart
        summary['requires_restart'] = [
            key for key, rule in self.config_rules.items() 
            if rule.requires_restart and rule.last_modified
        ]
        
        return summary


# Global configuration manager
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager 