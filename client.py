import argparse
import grpc
import voice_pb2
import voice_pb2_grpc


def parse_args():
    parser = argparse.ArgumentParser(description="Piper TTS gRPC client")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument(
        "-H", "--host",
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "-p", "--port",
        default=50051,
        type=int,
        help="Server port (default: 50051)",
    )
    return parser.parse_args()


def run():
    args = parse_args()
    with grpc.insecure_channel(f"{args.host}:{args.port}") as channel:
        stub = voice_pb2_grpc.VoiceServiceStub(channel)
        response = stub.SpeakLocally(voice_pb2.SpeechRequest(text=args.text))
        print(f"Status: {response.message}")


if __name__ == "__main__":
    run()
