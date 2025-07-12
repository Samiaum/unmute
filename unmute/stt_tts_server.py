from __future__ import annotations

import asyncio
import io
from typing import List

import numpy as np
import sphn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME
from unmute.stt.speech_to_text import (
    SpeechToText,
    STTMarkerMessage,
    STTWordMessage,
)
from unmute.tts.text_to_speech import (
    TextToSpeech,
    TTSAudioMessage,
    TTSClientEosMessage,
)

app = FastAPI()


async def _transcribe_audio(data: np.ndarray) -> str:
    stt = SpeechToText()
    await stt.start_up()
    for i in range(0, len(data), SAMPLES_PER_FRAME):
        chunk = data[i : i + SAMPLES_PER_FRAME]
        await stt.send_audio(chunk)
    await stt.send_marker(0)
    for _ in range(25):
        await stt.send_audio(np.zeros(SAMPLES_PER_FRAME, dtype=np.int16))

    words: List[str] = []
    async for msg in stt:
        if isinstance(msg, STTWordMessage):
            words.append(msg.text)
        elif isinstance(msg, STTMarkerMessage):
            break
    await stt.shutdown()
    return " ".join(words)


@app.post("/stt")
async def stt_http(file: UploadFile = File(...)):
    data, _ = sphn.read(file.file, sample_rate=SAMPLE_RATE)
    data = data[0]
    text = await _transcribe_audio(data)
    return {"text": text}


@app.websocket("/stt/ws")
async def stt_ws(websocket: WebSocket):
    await websocket.accept()
    buffer = bytearray()
    try:
        while True:
            chunk = await websocket.receive_bytes()
            buffer.extend(chunk)
    except WebSocketDisconnect:
        pass
    audio = np.frombuffer(buffer, dtype=np.int16)
    text = await _transcribe_audio(audio)
    await websocket.send_text(text)
    await websocket.close()


@app.post("/tts")
async def tts_http(text: str):
    tts = TextToSpeech()
    await tts.start_up()
    await tts.send(text)
    await tts.send(TTSClientEosMessage())
    audio_chunks: List[np.ndarray] = []
    async for msg in tts:
        if isinstance(msg, TTSAudioMessage):
            audio_chunks.append(np.array(msg.pcm, dtype=np.float32))
    await tts.shutdown()
    audio = np.concatenate(audio_chunks)
    return Response(content=audio.tobytes(), media_type="application/octet-stream")

