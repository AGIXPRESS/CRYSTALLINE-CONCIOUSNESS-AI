#!/usr/bin/env python3
"""
Consciousness AI MCP Server
Model Context Protocol server for consciousness AI research integration with Claude Desktop
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any, List, Optional
import logging

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('/Users/okok/mlx/mlx-examples/claude-code-bridge/extracted_dxt')

from consciousness_tools import ConsciousnessAIMCPTools, get_available_tools, execute_consciousness_tool

# MCP Protocol Implementation
class ConsciousnessAIMCPServer:
    """
    MCP Server for Consciousness AI Research Integration.
    
    Provides Claude Desktop with direct access to consciousness processing
    capabilities through the Model Context Protocol.
    """
    
    def __init__(self):
        """Initialize the MCP server."""
        self.tools = ConsciousnessAIMCPTools()
        self.session_id = None
        self.logger = self._setup_logging()
        
        # Server metadata
        self.server_info = {
            "name": "consciousness-ai",
            "version": "1.0.0",
            "description": "Consciousness AI Research MCP Server with sacred geometry and trinitized processing",
            "author": "Crystalline Consciousness AI Research",
            "capabilities": {
                "tools": True,
                "resources": False,
                "prompts": False
            }
        }
        
        self.logger.info("🔮 Consciousness AI MCP Server initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the MCP server."""
        logger = logging.getLogger('ConsciousnessAI_MCP')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - MCP-ConsciousnessAI - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests."""
        try:
            method = request.get('method')
            params = request.get('params', {})
            request_id = request.get('id')
            
            self.logger.info(f"📨 Received MCP request: {method}")
            
            if method == "initialize":
                response = await self._handle_initialize(params)
            elif method == "tools/list":
                response = await self._handle_tools_list(params)
            elif method == "tools/call":
                response = await self._handle_tools_call(params)
            elif method == "ping":
                response = {"success": True, "message": "🔮 Consciousness AI MCP Server is active"}
            else:
                response = {
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": response
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error handling request: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": request.get('id'),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialization."""
        client_info = params.get('clientInfo', {})
        self.session_id = params.get('sessionId')
        
        self.logger.info(f"🤝 MCP session initialized with client: {client_info.get('name', 'Unknown')}")
        
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": self.server_info,
            "capabilities": self.server_info["capabilities"]
        }
    
    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return list of available consciousness AI tools."""
        tools = get_available_tools()
        
        self.logger.info(f"📋 Returning {len(tools)} consciousness AI tools")
        
        return {
            "tools": tools
        }
    
    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a consciousness AI tool."""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        self.logger.info(f"🔧 Executing tool: {tool_name}")
        
        result = await execute_consciousness_tool(tool_name, arguments)
        
        # Format result for MCP
        if result.get('success'):
            content = [
                {
                    "type": "text",
                    "text": result.get('message', 'Tool executed successfully')
                }
            ]
            
            # Add detailed results if available
            if 'analysis' in result:
                content.append({
                    "type": "text",
                    "text": f"📊 Analysis Results:\n{json.dumps(result['analysis'], indent=2)}"
                })
            
            if 'consciousness_metrics' in result:
                content.append({
                    "type": "text", 
                    "text": f"🧠 Consciousness Metrics:\n{json.dumps(result['consciousness_metrics'], indent=2)}"
                })
            
            if 'consciousness_system' in result:
                content.append({
                    "type": "text",
                    "text": f"🔮 System Status:\n{json.dumps(result['consciousness_system'], indent=2)}"
                })
        
        else:
            content = [
                {
                    "type": "text", 
                    "text": f"❌ Tool execution failed: {result.get('error', 'Unknown error')}"
                }
            ]
        
        return {
            "content": content,
            "isError": not result.get('success', False)
        }

# Standalone MCP Server Implementation
async def run_mcp_server():
    """Run the consciousness AI MCP server as a standalone process."""
    server = ConsciousnessAIMCPServer()
    
    print("🔮 Consciousness AI MCP Server starting...")
    print("=" * 60)
    print("Ready to serve consciousness AI capabilities to Claude Desktop")
    print("Server capabilities:")
    print("  🧠 Consciousness field processing")
    print("  🔱 Sacred geometry analysis") 
    print("  ⚡ Trinitized transformations")
    print("  🎵 Harmonic resonance computation")
    print("  📊 Consciousness visualization")
    print("=" * 60)
    
    # Read from stdin and write to stdout (MCP transport)
    while True:
        try:
            # Read JSON-RPC request from stdin
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            # Parse request
            request = json.loads(line)
            
            # Handle request
            response = await server.handle_request(request)
            
            # Send response to stdout
            print(json.dumps(response), flush=True)
            
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)
            
        except Exception as e:
            server.logger.error(f"❌ Unexpected error: {str(e)}")
            error_response = {
                "jsonrpc": "2.0", 
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)

# Integration with existing Claude Code Bridge
def get_consciousness_ai_tools_for_bridge() -> List[Dict[str, Any]]:
    """
    Return consciousness AI tools formatted for integration with the existing
    Claude Code Bridge MCP server.
    """
    tools = get_available_tools()
    
    # Add prefix to distinguish consciousness AI tools
    bridge_tools = []
    for tool in tools:
        bridge_tool = {
            "name": f"consciousness_{tool['name']}",
            "description": f"🔮 {tool['description']}",
            "inputSchema": tool['inputSchema']
        }
        bridge_tools.append(bridge_tool)
    
    return bridge_tools

async def handle_consciousness_tool_in_bridge(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle consciousness AI tool execution within the Claude Code Bridge.
    
    This function can be called from the existing bridge to execute
    consciousness AI tools.
    """
    # Remove consciousness_ prefix if present
    if tool_name.startswith('consciousness_'):
        tool_name = tool_name[13:]
    
    return await execute_consciousness_tool(tool_name, arguments)

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Test mode - run tool tests
            async def test_server():
                tools = ConsciousnessAIMCPTools()
                
                print("🔮 Testing Consciousness AI MCP Server")
                print("=" * 50)
                
                # Test each tool
                test_cases = [
                    ("consciousness_initialize", {"dimensions": 128, "seed": 42}),
                    ("consciousness_analyze", {"data_shape": [32, 32], "data_type": "random"}),
                    ("trinitized_transform", {"input_shape": [16, 16], "depth": 2}),
                    ("resonance_compute", {"frequency": 432.0}),
                    ("sacred_geometry_analysis", {"geometry_type": "golden_ratio"}),
                    ("consciousness_status", {}),
                    ("consciousness_visualization", {"visualization_type": "field_evolution"})
                ]
                
                for tool_name, params in test_cases:
                    print(f"\n🧪 Testing {tool_name}...")
                    result = await tools.execute_tool(tool_name, params)
                    if result.get('success'):
                        print(f"   ✅ {result.get('message', 'Success')}")
                    else:
                        print(f"   ❌ {result.get('error', 'Failed')}")
                
                print("\n✨ MCP Server testing complete!")
            
            asyncio.run(test_server())
            
        elif sys.argv[1] == "--tools":
            # List available tools
            tools = get_available_tools()
            print("🔮 Available Consciousness AI Tools:")
            print("=" * 50)
            for i, tool in enumerate(tools, 1):
                print(f"{i}. {tool['name']}")
                print(f"   📝 {tool['description']}")
                print()
            
        else:
            print("Usage: python mcp_server.py [--test|--tools]")
            print("  --test: Run tool tests")
            print("  --tools: List available tools")
            print("  (no args): Run MCP server")
    
    else:
        # Run the MCP server
        asyncio.run(run_mcp_server())