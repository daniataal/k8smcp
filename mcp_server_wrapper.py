import logging
import importlib
import sys

logger = logging.getLogger(__name__)

class MCPServerWrapper:
    """
    Wrapper around different fastmcp implementations to provide a unified interface.
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = None
        self.handlers = {}
        
        # Try to import fastmcp and initialize the server
        self._initialize_server()
        
    def _initialize_server(self):
        """Initialize the server based on available fastmcp implementations"""
        # Try different methods of importing fastmcp
        try:
            # Try method 1: Direct import
            import fastmcp
            
            # Check if fastmcp has a Server class
            if hasattr(fastmcp, 'Server'):
                self.server = fastmcp.Server(self.host, self.port)
                self._register_method = self._register_method_direct
                self._start_method = self._start_method_direct
                logger.info("Using fastmcp.Server")
                return
                
            # Check if fastmcp has a create_server function
            if hasattr(fastmcp, 'create_server'):
                self.server = fastmcp.create_server(host=self.host, port=self.port)
                self._register_method = self._register_method_decorator
                self._start_method = self._start_method_direct
                logger.info("Using fastmcp.create_server")
                return
                
            # Try method 2: Import from fastmcp.mcp
            if hasattr(fastmcp, 'mcp'):
                if hasattr(fastmcp.mcp, 'Server'):
                    self.server = fastmcp.mcp.Server(self.host, self.port)
                    self._register_method = self._register_method_mcp
                    self._start_method = self._start_method_mcp
                    logger.info("Using fastmcp.mcp.Server")
                    return
            
            # Try a direct import of mcp
            try:
                from fastmcp import mcp
                self.server = mcp.Server(self.host, self.port)
                self._register_method = self._register_method_mcp
                self._start_method = self._start_method_mcp
                logger.info("Using imported mcp.Server")
                return
            except (ImportError, AttributeError):
                pass
                
            # Fall back to a very simplified MCP server
            logger.warning("No suitable fastmcp implementation found. Using fallback implementation.")
            self._create_fallback_server()
            
        except ImportError:
            logger.error("Failed to import fastmcp package. Verify it's installed correctly.")
            raise
    
    def _create_fallback_server(self):
        """Create a very simple fallback MCP server implementation"""
        import socket
        import threading
        import json
        
        class SimpleMCPServer:
            def __init__(self, host, port):
                self.host = host
                self.port = port
                self.handlers = {}
                self.running = False
                
            def add_handler(self, command, handler_func):
                self.handlers[command] = handler_func
                
            def serve_forever(self):
                self.running = True
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((self.host, self.port))
                server_socket.listen(5)
                
                logger.info(f"Fallback MCP server listening on {self.host}:{self.port}")
                
                try:
                    while self.running:
                        client_socket, address = server_socket.accept()
                        client_thread = threading.Thread(
                            target=self._handle_client, 
                            args=(client_socket, address)
                        )
                        client_thread.daemon = True
                        client_thread.start()
                finally:
                    server_socket.close()
                    
            def _handle_client(self, client_socket, address):
                try:
                    data = b""
                    while True:
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    
                    if data:
                        try:
                            request = json.loads(data.decode('utf-8'))
                            namespace = request.get('namespace')
                            command = request.get('command')
                            params = request.get('params', {})
                            
                            if namespace in self.handlers:
                                handler = self.handlers[namespace]
                                result = handler(command, params)
                                response = {
                                    'status': 'success',
                                    'result': result
                                }
                            else:
                                response = {
                                    'status': 'error',
                                    'message': f"Unknown namespace: {namespace}"
                                }
                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                        
                        client_socket.sendall(json.dumps(response).encode('utf-8'))
                finally:
                    client_socket.close()
        
        self.server = SimpleMCPServer(self.host, self.port)
        self._register_method = self._register_method_fallback
        self._start_method = self._start_method_fallback
        
    def _register_method_direct(self, namespace, handler_func):
        """Register handler using direct method"""
        self.server.register_handler(namespace, handler_func)
        
    def _register_method_decorator(self, namespace, handler_func):
        """Register handler using decorator method"""
        @self.server.handler(namespace)
        def wrapper(command, params):
            return handler_func(command, params)
            
    def _register_method_mcp(self, namespace, handler_func):
        """Register handler using mcp method"""
        self.server.add_handler(namespace, handler_func)
        
    def _register_method_fallback(self, namespace, handler_func):
        """Register handler using fallback method"""
        self.server.add_handler(namespace, handler_func)
        
    def _start_method_direct(self):
        """Start server using direct method"""
        self.server.start()
        
    def _start_method_mcp(self):
        """Start server using mcp method"""
        self.server.serve_forever()
        
    def _start_method_fallback(self):
        """Start server using fallback method"""
        self.server.serve_forever()
        
    def register_handler(self, namespace, handler_func):
        """Register a handler function for a namespace"""
        self._register_method(namespace, handler_func)
        
    def start(self):
        """Start the server"""
        self._start_method()
