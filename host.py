import os
import sys
from concurrent import futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_generated"))

import grpc

from abandoned_inference import AbandonedDetectionServicer
from compiler_generated.abandoned_detection_pb2_grpc import add_AbandonedDetectionServiceServicer_to_server
from interceptors.auth import AuthInterceptor
from compiler_generated.plugin_gateway_pb2_grpc import add_PipelineServiceServicer_to_server
from utils.pipeline_service import PipelineServiceImpl


MAX_MESSAGE_MB = int(os.getenv("MAX_MESSAGE_MB", "64"))


def serve():
    max_bytes = MAX_MESSAGE_MB * 1024 * 1024

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        interceptors=[
            AuthInterceptor()
        ],
        options=[
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ],
    )

    servicer = AbandonedDetectionServicer()

    add_AbandonedDetectionServiceServicer_to_server(
        servicer,
        server
    )

    add_PipelineServiceServicer_to_server(
        PipelineServiceImpl(servicer),
        server
    )

    server.add_insecure_port("0.0.0.0:50051")

    server.start()

    print("🚀 gRPC Abandoned Inference Server Running on :50051")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
