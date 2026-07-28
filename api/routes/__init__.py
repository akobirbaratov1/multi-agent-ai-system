from api.routes.chat import router as chat_router
from api.routes.crm import router as crm_router
from api.routes.knowledge import router as knowledge_router
from api.routes.monitoring import router as monitoring_router
from api.routes.operator import router as operator_router
from api.routes.webhooks import router as webhooks_router

__all__ = ["chat_router", "knowledge_router", "crm_router", "monitoring_router", "operator_router", "webhooks_router"]
