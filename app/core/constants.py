"""Constants and enums for the diagram AI service."""

from enum import Enum
from typing import TypedDict

__all__ = [
    "IntentType",
    "GREETING_PHRASES",
    "DIAGRAM_PHRASES",
    "HELP_PHRASES",
    "AWS_COMPONENTS",
]


class ComponentConfig(TypedDict):
    """Configuration for AWS component detection."""

    keywords: list[str]
    type: str
    label: str


class IntentType(Enum):
    """Intent types for user messages."""

    GENERATE_DIAGRAM = "generate_diagram"
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    HELP = "help"
    GENERAL = "general"
    UNKNOWN = "unknown"


# Phrase constants for intent detection
GREETING_PHRASES = ["hello", "hi", "hey"]

DIAGRAM_PHRASES = ["create", "generate", "diagram", "draw", "build"]

HELP_PHRASES = ["help", "how", "what", "explain"]

# AWS component detection keywords
AWS_COMPONENTS: dict[str, ComponentConfig] = {
    "alb": {
        "keywords": ["alb", "load balancer", "application load balancer"],
        "type": "alb",
        "label": "Application Load Balancer",
    },
    "api_gateway": {
        "keywords": ["api gateway", "apigateway", "gateway"],
        "type": "api_gateway",
        "label": "API Gateway",
    },
    "ec2": {
        "keywords": ["ec2", "server", "service", "microservice", "web"],
        "type": "ec2",
        "label": "Web / Service",
    },
    "lambda": {"keywords": ["lambda"], "type": "lambda", "label": "Lambda Function"},
    "rds": {
        "keywords": ["rds", "postgres", "mysql", "database", "db"],
        "type": "rds",
        "label": "Database",
    },
    "s3": {
        "keywords": ["s3", "bucket", "object storage"],
        "type": "s3",
        "label": "S3 Bucket",
    },
    "sqs": {"keywords": ["sqs", "queue"], "type": "sqs", "label": "Queue"},
    "sns": {"keywords": ["sns"], "type": "sns", "label": "SNS Topic"},
    "cloudwatch": {
        "keywords": ["cloudwatch", "monitoring", "metrics"],
        "type": "cloudwatch",
        "label": "CloudWatch",
    },
    "cognito": {"keywords": ["cognito", "auth"], "type": "cognito", "label": "Cognito"},
}
