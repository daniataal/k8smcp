import logging
import json
import time
import anthropic
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class ClaudeAnalyzer:
    """
    Handles integration with Claude API for analyzing Kubernetes resources and logs.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        
        if api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info("Claude API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude API client: {e}")
    
    def is_available(self) -> bool:
        """Check if Claude API is available for use"""
        return self.client is not None
    
    def _send_request(self, messages: List[Dict[str, str]], max_tokens: int = 4000) -> Optional[str]:
        """
        Send a request to Claude API.
        
        Args:
            messages: List of message dictionaries to send
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Claude's response text or None if the request fails
        """
        if not self.is_available():
            return None
            
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=max_tokens,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return None
    
    def analyze_logs(self, logs: str) -> Optional[Dict[str, Any]]:
        """
        Analyze Kubernetes pod logs for issues and patterns.
        
        Args:
            logs: The raw log content to analyze
            
        Returns:
            Dictionary containing the analysis results or None if failed
        """
        if not logs:
            return {"error": "No logs provided"}
            
        # Truncate logs if they're too long (API limit)
        if len(logs) > 100000:
            logs = logs[-100000:]
            
        prompt = [
            {"role": "user", "content": f"""
Please analyze these Kubernetes pod logs and provide a clear summary. Look for:

1. Error patterns or exceptions
2. Performance issues (slowdowns, timeouts)
3. Connection problems
4. Resource constraints (OOM, CPU throttling)
5. Security concerns
6. Startup/initialization issues

Then provide:
- A brief summary of what this service appears to be doing
- Major issues detected (ordered by severity)
- Root cause analysis for identified issues
- Specific recommendations to fix the problems

Here are the logs:
```
{logs}
```

Format your response as JSON with the following structure:
{{
  "summary": "Brief description of service and logs",
  "issues": [
    {{ "severity": "HIGH|MEDIUM|LOW", "issue": "Description", "pattern": "Log pattern" }}
  ],
  "root_causes": [
    {{ "issue": "Issue reference", "cause": "Likely root cause", "certainty": "High|Medium|Low" }}
  ],
  "recommendations": [
    {{ "action": "Recommended action", "explanation": "Why this would help" }}
  ]
}}
"""
            }
        ]
        
        response = self._send_request(prompt)
        if not response:
            return None
            
        try:
            # Extract the JSON from the response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON block found, try to parse the whole response
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse Claude's JSON response: {e}")
            return {"raw_analysis": response}
    
    def analyze_node_health(self, pressures: Dict[str, bool], 
                           resource_usage: Dict[str, Any],
                           conditions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyze node health based on pressure conditions and resource usage.
        
        Args:
            pressures: Dictionary of pressure conditions
            resource_usage: Dictionary of resource usage metrics
            conditions: List of node conditions
            
        Returns:
            Dictionary containing analysis results or None if failed
        """
        prompt = [
            {"role": "user", "content": f"""
Analyze this Kubernetes node's health status and provide recommendations.

Node pressure conditions:
{json.dumps(pressures, indent=2)}

Node resource usage:
{json.dumps(resource_usage, indent=2)}

Node conditions:
{json.dumps(conditions, indent=2)}

Format your response as JSON with the following structure:
{{
  "status_summary": "Brief description of node health",
  "issues_detected": [
    {{ "issue": "Description", "severity": "HIGH|MEDIUM|LOW", "impact": "How it affects the cluster" }}
  ],
  "possible_causes": [
    "Cause 1", "Cause 2"
  ],
  "recommendations": [
    {{ "action": "Recommended action", "explanation": "Why this would help", "priority": "HIGH|MEDIUM|LOW" }}
  ]
}}
"""
            }
        ]
        
        response = self._send_request(prompt)
        if not response:
            return None
            
        try:
            # Extract the JSON from the response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON block found, try to parse the whole response
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse Claude's JSON response: {e}")
            return {"raw_analysis": response}
    
    def analyze_cluster_health(self, issues: List[Dict[str, Any]], 
                              warnings: List[Dict[str, Any]],
                              component: Optional[str] = None,
                              name: Optional[str] = None,
                              namespace: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze cluster health based on discovered issues and warnings.
        
        Args:
            issues: List of issues detected
            warnings: List of warnings detected
            component: Specific component being analyzed (optional)
            name: Name of the resource being analyzed (optional)
            namespace: Namespace of the resource (optional)
            
        Returns:
            Dictionary containing analysis and recommendations
        """
        scope = f"component {component}" if component else "cluster"
        if name:
            scope += f" '{name}'"
        if namespace:
            scope += f" in namespace '{namespace}'"
            
        prompt = [
            {"role": "user", "content": f"""
Analyze the health of this Kubernetes {scope} and provide recommendations.

Issues detected:
{json.dumps(issues, indent=2)}

Warnings detected:
{json.dumps(warnings, indent=2)}

You are an expert Kubernetes troubleshooter. Format your response as JSON with the following structure:
{{
  "overall_health": "HEALTHY|DEGRADED|CRITICAL",
  "summary": "Brief assessment of the cluster's health",
  "critical_issues": [
    {{ "issue": "Description", "impact": "How it affects the cluster", "components_affected": ["list", "of", "components"] }}
  ],
  "correlations": [
    {{ "pattern": "Correlated issue pattern", "explanation": "Why these issues might be related" }}
  ],
  "recommendations": [
    {{ 
      "action": "Specific action to take", 
      "command": "kubectl command if applicable",
      "explanation": "Why this would help",
      "priority": "HIGH|MEDIUM|LOW"
    }}
  ]
}}
"""
            }
        ]
        
        response = self._send_request(prompt)
        if not response:
            return None
            
        try:
            # Extract the JSON from the response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON block found, try to parse the whole response
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse Claude's JSON response: {e}")
            return {"raw_analysis": response}
    
    def analyze_resource(self, resource_type: str, 
                        resource_data: Dict[str, Any],
                        events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyze a specific Kubernetes resource.
        
        Args:
            resource_type: Type of resource (pod, deployment, etc.)
            resource_data: Resource data
            events: Events related to the resource
            
        Returns:
            Dictionary containing analysis results
        """
        prompt = [
            {"role": "user", "content": f"""
Analyze this Kubernetes {resource_type} and its events. Provide a detailed assessment and recommendations.

Resource data:
{json.dumps(resource_data, indent=2)}

Events:
{json.dumps(events, indent=2)}

You are a Kubernetes expert. Format your response as JSON with the following structure:
{{
  "status_summary": "Brief description of resource status",
  "health": "HEALTHY|DEGRADED|CRITICAL",
  "issues": [
    {{ "issue": "Description", "severity": "HIGH|MEDIUM|LOW", "evidence": "What indicates this issue" }}
  ],
  "insights": [
    "Insight 1", "Insight 2"
  ],
  "recommendations": [
    {{ 
      "action": "Recommended action", 
      "command": "kubectl command if applicable",
      "explanation": "Why this would help" 
    }}
  ]
}}
"""
            }
        ]
        
        response = self._send_request(prompt)
        if not response:
            return None
            
        try:
            # Extract the JSON from the response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON block found, try to parse the whole response
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse Claude's JSON response: {e}")
            return {"raw_analysis": response}
    
    def recommend_actions(self, resource_type: str,
                         resource_data: Dict[str, Any],
                         events: List[Dict[str, Any]],
                         logs: Dict[str, str],
                         issue: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Get recommendations for fixing issues with a resource.
        
        Args:
            resource_type: Type of resource (pod, deployment, etc.)
            resource_data: Resource data
            events: Events related to the resource
            logs: Container logs
            issue: Description of the issue (if provided)
            
        Returns:
            Dictionary containing recommendations
        """
        issue_desc = f" with issue: {issue}" if issue else ""
        
        # Prepare logs for inclusion
        logs_text = ""
        for container, log in logs.items():
            # Truncate logs if they're too long
            if len(log) > 10000:
                log = log[-10000:]
            logs_text += f"\nContainer {container} logs:\n{log}\n"
            
        prompt = [
            {"role": "user", "content": f"""
As a Kubernetes expert, recommend actions to fix this {resource_type}{issue_desc}.

Resource data:
{json.dumps(resource_data, indent=2)}

Events:
{json.dumps(events, indent=2)}

{logs_text}

Provide a detailed troubleshooting plan. Format your response as JSON with the following structure:
{{
  "problem_analysis": "Detailed analysis of what's wrong",
  "root_cause": "Likely root cause of the issue",
  "recommended_actions": [
    {{
      "action": "Specific action to take",
      "command": "kubectl command to execute",
      "expected_outcome": "What this action will accomplish",
      "priority": "HIGH|MEDIUM|LOW"
    }}
  ],
  "verification_steps": [
    "How to verify the issue is fixed"
  ],
  "preventative_measures": [
    "How to prevent this issue in the future"
  ]
}}
"""
            }
        ]
        
        response = self._send_request(prompt, max_tokens=8000)
        if not response:
            return None
            
        try:
            # Extract the JSON from the response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON block found, try to parse the whole response
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse Claude's JSON response: {e}")
            return {"raw_analysis": response}
    
    def analyze_prometheus_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze Prometheus query results"""
        prompt = [
            {"role": "user", "content": f"""
Analyze these Prometheus query results and provide insights.

Query results:
{json.dumps(data, indent=2)}

You are a Kubernetes monitoring expert. Format your response as JSON with the following structure:
{{
  "summary": "Brief summary of the metrics data",
  "anomalies": [
    {{ "metric": "Metric name", "observation": "What's unusual", "significance": "Why it matters" }}
  ],
  "trends": [
    {{ "pattern": "Observed trend", "analysis": "What this indicates" }}
  ],
  "insights": [
    "Insight 1", "Insight 2"
  ],
  "recommendations": [
    {{ "action": "Recommended action", "reasoning": "Why this would help" }}
  ]
}}
"""
            }
        ]
        
        response = self._send_request(prompt)
        if not response:
            return None
            
        try:
            # Extract the JSON from the response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON block found, try to parse the whole response
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse Claude's JSON response: {e}")
            return {"raw_analysis": response}
    
    def analyze_elasticsearch_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze Elasticsearch query results"""
        prompt = [
            {"role": "user", "content": f"""
Analyze these Elasticsearch query results and provide insights.

Query results:
{json.dumps(data, indent=2)}

You are a Kubernetes logging expert. Format your response as JSON with the following structure:
{{
  "summary": "Brief summary of the log data",
  "patterns": [
    {{ "pattern": "Log pattern observed", "frequency": "How often it occurs", "significance": "Why it matters" }}
  ],
  "anomalies": [
    {{ "description": "Unusual behavior", "evidence": "Supporting log entries", "severity": "HIGH|MEDIUM|LOW" }}
  ],
  "insights": [
    "Insight 1", "Insight 2"
  ],
  "recommendations": [
    {{ "action": "Recommended action", "reasoning": "Why this would help" }}
  ]
}}
"""
            }
        ]
        
        response = self._send_request(prompt)
        if not response:
            return None
            
        try:
            # Extract the JSON from the response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON block found, try to parse the whole response
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse Claude's JSON response: {e}")
            return {"raw_analysis": response}
