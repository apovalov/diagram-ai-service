from __future__ import annotations

__all__ = [
    "intent_prompt",
    "diagram_analysis_prompt",
    "diagram_critique_prompt",
    "diagram_adjustment_prompt",
]


def intent_prompt(message: str) -> str:
    """Generate intent classification prompt for assistant agent."""
    # escape potential injection by wrapping in triple quotes and escaping triple quotes
    safe_message = message.replace('"""', r"\"\"\"").replace("\\", "\\\\")
    return f'''
You are an intelligent assistant. Your job is to determine the user's intent from their message.

Message: """{safe_message}"""

Possible intents are:
- "generate_diagram": The user wants to generate a diagram.
- "clarification": The user is asking for more information or clarification.
- "greeting": The user is just saying hello.
- "unknown": The user's intent is unclear.

Please respond with a JSON object containing the user's intent and any relevant entities.
For example:
{{
    "intent": "generate_diagram",
    "description": "Create a diagram of a web application."
}}
'''


def diagram_analysis_prompt(description: str) -> str:
    """Generate diagram analysis prompt for diagram agent."""
    # escape potential injection by wrapping in triple quotes and escaping triple quotes
    safe_description = description.replace('"""', r"\"\"\"").replace("\\", "\\\\")
    return f'''
You are a diagram architecture expert. Analyze the user's natural language description and break it down into specific components, relationships, and groupings needed for a technical diagram.

Description: """{safe_description}"""

Available component types:
- Compute: ec2, lambda, service, microservice, web_server
- Database: rds, dynamodb, database
- Network & Load Balancing: elb, alb, nlb, api_gateway, apigateway, gateway
- Storage: s3
- Integration & Messaging: sqs, sns, queue
- Management & Monitoring: cloudwatch, monitoring
- Security: iam, cognito, auth_service
- Analytics: kinesis
- Developer Tools: codebuild, codepipeline

Please identify:
1. All nodes/components mentioned (give each a unique id)
2. Their types (use the available types listed above)
3. Any grouping/clustering requirements
4. Connections and relationships between components (using the unique ids)
5. Any specific labeling requirements

For microservices, use "service" or "microservice" type, or specific service types like "auth_service", "payment_service", "order_service".
For Application Load Balancer, use "alb" type.
For API Gateway, use "api_gateway" type.
For SQS queues, use "sqs" type.
For CloudWatch monitoring, use "cloudwatch" type.

Respond in structured JSON format like this example:

{{
    "title": "Application Diagram",
    "nodes": [
        {{"id": "alb", "type": "alb", "label": "Application Load Balancer"}},
        {{"id": "web1", "type": "ec2", "label": "Web Server 1"}},
        {{"id": "web2", "type": "ec2", "label": "Web Server 2"}},
        {{"id": "db", "type": "rds", "label": "Database"}},
        {{"id": "api_gw", "type": "api_gateway", "label": "API Gateway"}},
        {{"id": "auth_svc", "type": "auth_service", "label": "Authentication Service"}},
        {{"id": "queue", "type": "sqs", "label": "Message Queue"}},
        {{"id": "monitoring", "type": "cloudwatch", "label": "CloudWatch"}}
    ],
    "clusters": [
        {{"label": "Web Tier", "nodes": ["web1", "web2"]}},
        {{"label": "Microservices", "nodes": ["auth_svc"]}}
    ],
    "connections": [
        {{"source": "alb", "target": "web1"}},
        {{"source": "alb", "target": "web2"}},
        {{"source": "web1", "target": "db"}},
        {{"source": "web2", "target": "db"}},
        {{"source": "api_gw", "target": "auth_svc"}},
        {{"source": "auth_svc", "target": "queue"}}
    ]
}}
'''


def diagram_critique_prompt(description: str) -> str:
    return f'''
You are a senior systems architect. You will be given:
- A user's description of the desired diagram
- The current rendered diagram image
- The current structured analysis used to produce the diagram

Task:
1) Judge whether the current diagram fully satisfies the user's description.
  a) If there is a node which is not connected to any other node, it is a sign that something is missing, not always true though.
  b) If there is a node connected to all other nodes the graph gets unreadable and we need to think about how to limit the number of connections.
2) If the diagram does satisfy the user's description, set done=true. If not, set done=false and provide a concise critique of what is missing or incorrect.

Respond strictly as JSON:
{{
  "done": <true|false>,
  "critique": "<brief explanation if not done>"
}}

User description: """{description}"""
'''


def diagram_adjustment_prompt(description: str, critique: str) -> str:
    return f'''
You are a senior systems architect. The current diagram does not satisfy the request.
User description: """{description}"""
Critique: """{critique}"""

Adjust the structured diagram analysis so that the next render addresses the critique. Return only valid JSON matching the DiagramAnalysis schema.
'''
