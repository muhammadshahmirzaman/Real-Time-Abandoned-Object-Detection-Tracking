import sys
from pathlib import Path

from concurrent import futures
import grpc
from compiler_generated.simple_object_detector_pb2_grpc import add_ObjectDetectionServicer_to_server
from inference import NewImplObjectDetectionServicer
from interceptors.auth import AuthInterceptor
def serve():

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        interceptors=[
            AuthInterceptor()  # Uncomment to enable authentication
        ],
        options=[
        ("grpc.max_send_message_length", 20 * 1024 * 1024),
        ("grpc.max_receive_message_length", 20 * 1024 * 1024),
    ],
    )

    add_ObjectDetectionServicer_to_server(
        NewImplObjectDetectionServicer(),# Rule:only this line is changed according to use cases
        server
    )

    server.add_insecure_port("0.0.0.0:50051")

    server.start()

    print("🚀 gRPC Object Detection Server Running on :50051")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()