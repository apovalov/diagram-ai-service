# prompts.py
from __future__ import annotations

from app.core.constants import IntentType

__all__ = [
    "intent_prompt",
    "diagram_analysis_prompt",
    "diagram_critique_prompt",
    "diagram_adjustment_prompt",
]


def _escape_triple_quoted(text: str) -> str:
    """Escape input for safe insertion inside a Python f-string triple-quoted block.
    Order matters: escape backslashes first, then triple quotes.
    """
    return text.replace("\\", "\\\\").replace('"""', r"\"\"\"")


def intent_prompt(message: str) -> str:
    """Generate intent classification prompt for assistant agent."""
    safe_message = _escape_triple_quoted(message or "")
    return f'''
You are an intelligent assistant. Determine the user's intent from their message.

Message: """{safe_message}"""

Possible intents are:
- "{IntentType.GENERATE_DIAGRAM.value}": The user wants to generate a diagram.
- "{IntentType.CLARIFICATION.value}": The user is asking for more information or clarification.
- "{IntentType.GREETING.value}": The user is just saying hello.
- "{IntentType.UNKNOWN.value}": The user's intent is unclear.

Respond ONLY with a JSON object:
{{
  "intent": "<one of: {IntentType.GENERATE_DIAGRAM.value} | {IntentType.CLARIFICATION.value} | {IntentType.GREETING.value} | {IntentType.UNKNOWN.value}>",
  "description": "<optional short paraphrase if intent is {IntentType.GENERATE_DIAGRAM.value}>"
}}
'''


def diagram_analysis_prompt(description: str) -> str:
    """Generate diagram analysis prompt for diagram agent (Core v1 canonical set)."""
    safe_description = _escape_triple_quoted(description or "")
    return f'''
You are a diagram architecture expert. Analyze the user's natural-language description
and produce a precise plan for a technical diagram using ONLY the canonical component types
listed below.

Description: """{safe_description}"""

Canonical component types (use these exact values in "type"):
- Compute: ec2, lambda, service
- Database & Storage: rds, dynamodb, s3
- Networking & Routing: alb, api_gateway, vpc, internet_gateway
- Integration: sqs, sns
- Observability & Identity: cloudwatch, cognito

Clustering Guidelines:
- Each node can belong to ONLY ONE cluster - no overlapping clusters allowed
- When choosing between multiple possible groupings, prioritize FUNCTIONAL groupings over infrastructure groupings
- Examples of functional groupings: "Web Tier", "Microservices", "Data Layer", "Processing Pipeline"
- Examples of infrastructure groupings: "VPC", "Availability Zone", "Network Segment"
- If a description mentions both (e.g., "microservices in a VPC"), choose the functional grouping ("Microservices")
- Infrastructure components that don't fit functional groups should remain unclustered

Rules:
- Do not invent components not present or clearly implied.
- If the user mentions an unsupported AWS service, choose the closest canonical type
- Every node must have a unique "id" and a human-readable "label"
- "connections" must reference existing node ids
- Prefer simple, direct edges unless the text implies a specific flow
- Respond as VALID JSON only (no comments, no trailing commas)

Schema:
{{
  "title": "<short diagram title>",
  "nodes": [
    {{"id": "<unique_id>", "type": "<canonical_type>", "label": "<display label>"}}
  ],
  "clusters": [
    {{"label": "<group name>", "nodes": ["<id1>", "<id2>"]}}
  ],
  "connections": [
    {{"source": "<id>", "target": "<id>", "label": "<edge label, optional>"}}
  ]
}}

Example (functional grouping preferred):
{{
  "title": "Microservices Architecture",
  "nodes": [
    {{"id": "api_gw", "type": "api_gateway", "label": "API Gateway"}},
    {{"id": "auth_svc", "type": "service", "label": "Auth Service"}},
    {{"id": "payment_svc", "type": "service", "label": "Payment Service"}},
    {{"id": "order_svc", "type": "service", "label": "Order Service"}},
    {{"id": "queue", "type": "sqs", "label": "Message Queue"}},
    {{"id": "db", "type": "rds", "label": "Database"}},
    {{"id": "cognito", "type": "cognito", "label": "User Pool"}}
  ],
  "clusters": [
    {{"label": "Microservices", "nodes": ["auth_svc", "payment_svc", "order_svc"]}}
  ],
  "connections": [
    {{"source": "cognito", "target": "api_gw"}},
    {{"source": "api_gw", "target": "auth_svc"}},
    {{"source": "api_gw", "target": "payment_svc"}},
    {{"source": "api_gw", "target": "order_svc"}},
    {{"source": "order_svc", "target": "queue"}},
    {{"source": "queue", "target": "payment_svc"}},
    {{"source": "auth_svc", "target": "db"}},
    {{"source": "payment_svc", "target": "db"}},
    {{"source": "order_svc", "target": "db"}}
  ]
}}
'''


def diagram_critique_prompt(description: str) -> str:
    safe_description = _escape_triple_quoted(description or "")
    return f'''
You are a senior systems architect. You will be given:
- A user's description of the desired diagram
- The current rendered diagram image
- The current structured analysis used to produce the diagram

Task:
1) Judge whether the current diagram fully satisfies the user's description.
  a) If there is a node which is not connected to any other node, it is a sign that something is missing (not always, but often).
  b) If one node is connected to nearly all nodes, the graph may be unreadable; consider re-routing via fewer hubs.
2) If the diagram does satisfy the user's description, set done=true. If not, set done=false and provide a concise critique of what is missing or incorrect.

Respond strictly as JSON:
{{
  "done": <true|false>,
  "critique": "<brief explanation if not done>"
}}

User description: """{safe_description}"""
'''


def diagram_adjustment_prompt(description: str, critique: str) -> str:
    safe_description = _escape_triple_quoted(description or "")
    safe_critique = _escape_triple_quoted(critique or "")
    return f'''
You are a senior systems architect. The current diagram does not satisfy the request.
User description: """{safe_description}"""
Critique: """{safe_critique}"""

Adjust the structured diagram analysis so that the next render addresses the critique. Return only valid JSON matching the DiagramAnalysis schema.
'''