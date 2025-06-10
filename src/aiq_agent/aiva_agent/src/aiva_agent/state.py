from contextvars import ContextVar

ctx_routing_level: ContextVar[int] = ContextVar("routing_level", default=0)
