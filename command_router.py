import logging
import re
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

class CommandRouter:
    """
    Routes commands to their handler functions based on pattern matching.
    """
    def __init__(self, handler_obj):
        self.handler_obj = handler_obj
        self.routes = {}
    
    def register(self, pattern: str, handler_func: Callable):
        """
        Register a handler function for a command pattern.
        
        Args:
            pattern: Command pattern to match (can contain wildcards)
            handler_func: Function to call when pattern matches
        """
        # Convert simple pattern with * to regex
        regex_pattern = pattern.replace("*", ".*")
        self.routes[regex_pattern] = handler_func
        logger.debug(f"Registered handler for pattern: {pattern}")
    
    def route(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a command to its appropriate handler.
        
        Args:
            command: The command string to route
            params: Parameters to pass to the handler function
            
        Returns:
            The result from the handler function
        
        Raises:
            ValueError: If no matching handler is found
        """
        command = command.lower().strip()
        
        for pattern, handler in self.routes.items():
            if re.fullmatch(pattern, command, re.IGNORECASE):
                logger.debug(f"Routing command '{command}' to handler for pattern '{pattern}'")
                return handler(params)
        
        logger.warning(f"No handler found for command: {command}")
        return {
            "status": "error",
            "message": f"Unknown command: {command}. Type 'help' to see available commands."
        }
