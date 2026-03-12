#!/usr/bin/env python3
"""Test client: send WAV file as raw PCM to moonshine-server, print JSON responses."""
import socket
import struct
import sys
import threading
import wave

def recv_thread(sock):
    buf = b""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                print(line.decode())
        except OSError:
            break
    print("[CLIENT] Server closed connection")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.wav> [host] [port]")
        sys.exit(1)

    wav_path = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 9900

    # Read WAV and convert to f32le
    with wave.open(wav_path, "rb") as wf:
        assert wf.getsampwidth() == 2, "Only 16-bit WAV supported"
        assert wf.getnchannels() == 1, "Only mono WAV supported"
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    # Convert int16 -> float32
    samples_i16 = struct.unpack(f"<{len(frames)//2}h", frames)
    samples_f32 = struct.pack(f"<{len(samples_i16)}f",
                              *[s / 32768.0 for s in samples_i16])

    print(f"[CLIENT] {wav_path}: {len(samples_i16)} samples, {sr}Hz, "
          f"{len(samples_i16)/sr:.1f}s")

    # Resample to 16kHz if needed (simple skip/repeat, not ideal)
    if sr != 16000:
        print(f"[CLIENT] WARNING: sample rate {sr}Hz, server expects 16kHz. "
              "Moonshine will resample internally.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"[CLIENT] Connected to {host}:{port}")

    # Start receiver thread
    t = threading.Thread(target=recv_thread, args=(sock,), daemon=True)
    t.start()

    # Send audio in 100ms chunks (6400 bytes = 1600 float32 samples)
    chunk_bytes = 1600 * 4  # 100ms @ 16kHz f32le
    import time
    for i in range(0, len(samples_f32), chunk_bytes):
        chunk = samples_f32[i:i+chunk_bytes]
        # Pad last chunk if needed
        if len(chunk) < chunk_bytes:
            chunk += b"\x00" * (chunk_bytes - len(chunk))
        sock.sendall(chunk)
        time.sleep(0.1)  # Simulate real-time streaming

    # Wait for final results
    print("[CLIENT] Audio sent, waiting for final results...")
    time.sleep(2)
    sock.close()
    print("[CLIENT] Done")

if __name__ == "__main__":
    main()
