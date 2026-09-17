"""Implements the shared PipelineService contract (plugin_gateway.proto).

This is what the REST gateway calls to discover stage names and check
liveness — separate from VisionService, whose RPCs are specific to
face-landmark processing.
"""

import logging

from compiler_generated.plugin_gateway_pb2 import (
    GetStageNamesResponse,
    HealthCheckResponse,
    HealthStatus,
    Stage,
)
from compiler_generated.plugin_gateway_pb2_grpc import PipelineServiceServicer

logger = logging.getLogger(__name__)

# STAGE_NAME = "Abandoned Detection"
STAGE_DISPLAY_NAME = "Abandoned Detection"


class PipelineServiceImpl(PipelineServiceServicer):
    def __init__(self, vision_servicer) -> None:
        self._vision_servicer = vision_servicer

    def GetStageNames(self, request, context):
        stage_name = self._vision_servicer.__class__.__name__
        return GetStageNamesResponse(
            stages=[Stage(name=stage_name, display_name=STAGE_DISPLAY_NAME)]
        )

    def HealthCheck(self, request, context):
        model_loaded = self._vision_servicer is not None
        if model_loaded:
            return HealthCheckResponse(
                status=HealthStatus.HEALTH_STATUS_SERVING,
                message="abandoned-classifier model loaded",
            )
        return HealthCheckResponse(
            status=HealthStatus.HEALTH_STATUS_NOT_SERVING,
            message="abandoned-classifier model not loaded",
        )
