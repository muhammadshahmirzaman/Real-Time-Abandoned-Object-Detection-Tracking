import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_generated"))

import time
import cv2
import numpy as np

from compiler_generated.simple_object_detector_pb2 import (
    DetectionObject,
    DetectionResponse,
    HealthCheckResponse,
    NamesResponse    
)
from compiler_generated.simple_object_detector_pb2_grpc import ObjectDetectionServicer


class NewImplObjectDetectionServicer(
    ObjectDetectionServicer
):

    def Forward(self, request, context):

        start = time.time()

        # Decode image
        np_arr = np.frombuffer(request.frame, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return DetectionResponse(
                message="invalid image"
            )

        h, w, _ = image.shape

        # Dummy lightweight detection
        detections = [
            DetectionObject(
                x1=w * 0.25,
                y1=h * 0.25,
                x2=w * 0.75,
                y2=h * 0.75,
                class_name="mock_object",
                confidence=0.95,
                track_id=1,
                attributes={
                    "source": "mock_detector"
                }
            )
        ]

        inference_time = int((time.time() - start) * 1000)

        return DetectionResponse(
            objects=detections,
            message="success",
            inference_time_ms=inference_time,
            model_version="v1",
            frame_id=request.frame_id
        )

    def HealthCheck(self, request, context):

        return HealthCheckResponse(
            status="healthy",
            service="simple_object_detector",
            timestamp=int(time.time() * 1000),
            model_loaded=True,
            version="v1"
        )
    def GetNames(self, request, context):
        return NamesResponse(
            names={1: "mock_object"}
        )
