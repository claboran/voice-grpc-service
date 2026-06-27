import argparse
import grpc
import io
import logging
import queue
import subprocess
import threading
import wave
from concurrent import futures
from pathlib import Path

from piper.voice import PiperVoice

import voice_pb2
import voice_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ESPEAK_DATA_DIR = Path("./piper/espeak-ng-data")


def _load_voice(model_path: str, use_cuda: bool) -> PiperVoice:
    logger.info("Loading %s (cuda=%s)...", model_path, use_cuda)
    if use_cuda:
        try:
            voice = PiperVoice.load(
                model_path,
                use_cuda=True,
                espeak_data_dir=ESPEAK_DATA_DIR,
            )
            logger.info("Voice loaded on GPU (CUDAExecutionProvider).")
            return voice
        except Exception as exc:
            logger.warning("CUDA load failed (%s), falling back to CPU.", exc)
    voice = PiperVoice.load(
        model_path,
        use_cuda=False,
        espeak_data_dir=ESPEAK_DATA_DIR,
    )
    logger.info("Voice loaded on CPU.")
    return voice


class _Job:
    __slots__ = ("text", "result", "done")

    def __init__(self, text: str):
        self.text = text
        self.result: voice_pb2.SpeechStatus | None = None
        self.done = threading.Event()


class VoiceService(voice_pb2_grpc.VoiceServiceServicer):
    def __init__(self, voice: PiperVoice, aplay_device: str, max_queued: int = 3):
        self._voice = voice
        self._aplay_cmd = ["aplay", "-D", aplay_device, "-t", "wav", "-q"]
        # Bounded queue — worker thread is the sole owner of the CUDA context.
        self._q: queue.Queue[_Job] = queue.Queue(maxsize=max_queued)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            job = self._q.get()
            try:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wav_file:
                    self._voice.synthesize_wav(job.text, wav_file)

                aplay = subprocess.Popen(self._aplay_cmd, stdin=subprocess.PIPE)
                aplay.communicate(input=buf.getvalue())

                if aplay.returncode != 0:
                    job.result = voice_pb2.SpeechStatus(
                        success=False,
                        message=f"aplay exited with code {aplay.returncode}",
                    )
                else:
                    job.result = voice_pb2.SpeechStatus(success=True, message="Playback complete.")

            except Exception as exc:
                logger.error("Playback failed: %s", exc)
                job.result = voice_pb2.SpeechStatus(success=False, message=str(exc))
            finally:
                job.done.set()

    def SpeakLocally(self, request, context):
        logger.info("Speaking: '%s'", request.text)
        job = _Job(request.text)
        try:
            self._q.put_nowait(job)
        except queue.Full:
            logger.warning("Queue full, rejecting request.")
            return voice_pb2.SpeechStatus(success=False, message="Server busy, try again later.")
        job.done.wait()
        return job.result


def parse_args():
    parser = argparse.ArgumentParser(description="Piper TTS gRPC server")
    parser.add_argument(
        "-m", "--model",
        default="en_US-lessac-high.onnx",
        help="Path to the Piper ONNX voice model (default: en_US-lessac-high.onnx)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA and run on CPU only",
    )
    parser.add_argument(
        "-d", "--device",
        default="plughw:0,0",
        help="ALSA device for aplay (default: plughw:0,0)",
    )
    parser.add_argument(
        "-p", "--port",
        default=50051,
        type=int,
        help="gRPC port (default: 50051)",
    )
    return parser.parse_args()


def serve():
    args = parse_args()

    voice = _load_voice(args.model, use_cuda=not args.no_cuda)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    voice_pb2_grpc.add_VoiceServiceServicer_to_server(
        VoiceService(voice, args.device), server
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    logger.info(
        "Voice gRPC server listening on port %d  model=%s  cuda=%s",
        args.port, args.model, not args.no_cuda,
    )
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
