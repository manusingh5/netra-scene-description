# app.py
"""
NETRA Backend Server - With Real Pipeline + TTS
Supports up to 500MB video uploads
"""

import os
import asyncio
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Pipeline imports
from pipeline import (
    SequentialInference,
    DescriptorFusion,
    SceneSummarizer,
    QARetriever
)
from utils.tts import generate_audio, unload_tts

# ============================================================
# CONFIGURATION
# ============================================================

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

tasks = {}

# ============================================================
# SIZE LIMIT MIDDLEWARE
# ============================================================

class MaxSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and "/upload" in request.url.path:
            content_length = request.headers.get("Content-Length", "0")
            try:
                if int(content_length) > MAX_UPLOAD_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": f"File too large ({int(content_length)/(1024*1024):.1f}MB). Maximum: {MAX_UPLOAD_SIZE // (1024*1024)}MB"
                        }
                    )
            except:
                pass
        return await call_next(request)

app = FastAPI(title="NETRA Backend")

app.add_middleware(MaxSizeMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REAL PIPELINE PROCESSING
# ============================================================

async def process_video_async(task_id: str, video_path: str):
    """Background task: Real pipeline processing"""
    try:
        # Step 1: Sequential Inference (Florence → YOLO → OCR)
        tasks[task_id]["progress"] = 10
        tasks[task_id]["stage"] = "Extracting frames & running inference..."
        
        inference = SequentialInference()
        descriptors = await asyncio.to_thread(inference.run, video_path, 0.5)
        
        tasks[task_id]["progress"] = 40
        tasks[task_id]["stage"] = "Inference complete. Fusing descriptors..."
        
        # Free perception models (save RAM for LLM)
        inference.unload_perception_models()

        # Step 2: Descriptor Fusion
        tasks[task_id]["progress"] = 50
        fusion = DescriptorFusion()
        fused_descriptors = await asyncio.to_thread(fusion.fuse, descriptors)

        # Store for QA
        tasks[task_id]["descriptors"] = descriptors
        tasks[task_id]["fused_descriptors"] = fused_descriptors

        # Step 3: Scene Summarization (Phi-3)
        tasks[task_id]["progress"] = 70
        tasks[task_id]["stage"] = "Generating narration..."
        summarizer = SceneSummarizer()
        narration = await asyncio.to_thread(summarizer.generate_narration, fused_descriptors)
        tasks[task_id]["narration_text"] = narration

        # Step 4: TTS Audio Generation (Pyttsx3)
        tasks[task_id]["progress"] = 85
        tasks[task_id]["stage"] = "Synthesizing audio..."
        audio_filename = f"{task_id}_narration.wav"
        audio_path = os.path.join("outputs", audio_filename)

        audio_result = await asyncio.to_thread(generate_audio, narration, audio_path, "pyttsx3")

        if audio_result:
            tasks[task_id]["audio_path"] = audio_result

        # Unload LLM
        summarizer.unload_llm()

        # Done
        tasks[task_id]["progress"] = 100
        tasks[task_id]["stage"] = "Completed"
        tasks[task_id]["status"] = "completed"

        print(f"[DEBUG] Task {task_id} completed!")

    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["stage"] = str(e)
        print(f"[ERROR] Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "NETRA Backend Running",
        "version": "2.0",
        "max_upload_size_mb": MAX_UPLOAD_SIZE // (1024*1024),
        "status": "healthy"
    }

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    """Handle video upload (up to 500MB)"""
    try:
        print(f"[DEBUG] File upload started: {video.filename}")

        if not video.filename:
            return JSONResponse(status_code=400, content={"error": "No filename"})

        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv')
        if not video.filename.lower().endswith(valid_extensions):
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid file type. Use: {', '.join(valid_extensions)}"}
            )

        # Read file content
        content = await asyncio.to_thread(lambda: video.file.read())
        file_size_mb = len(content) / (1024 * 1024)

        print(f"[DEBUG] File uploaded: {video.filename} ({file_size_mb:.2f} MB)")

        if len(content) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {MAX_UPLOAD_SIZE // (1024*1024)}MB"
                }
            )

        task_id = str(uuid.uuid4())[:8]
        file_path = f"uploads/{task_id}_{video.filename}"

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        print(f"[DEBUG] Video saved to: {file_path}")

        tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "stage": "initializing",
            "file_path": file_path,
            "file_size_mb": round(file_size_mb, 2),
            "created_at": datetime.now().isoformat(),
            "descriptors": None,
            "fused_descriptors": None,
            "audio_path": None,
            "narration_text": None
        }

        # Start async pipeline (pass video_path!)
        asyncio.create_task(process_video_async(task_id, file_path))

        return {
            "task_id": task_id,
            "message": f"Video uploaded successfully ({file_size_mb:.2f} MB)",
            "status": "processing",
            "file_size_mb": file_size_mb
        }

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    task = tasks[task_id]
    return {
        "status": task["status"],
        "progress": task["progress"],
        "stage": task["stage"],
        "task_id": task_id
    }

@app.get("/audio/{task_id}")
async def get_audio(task_id: str):
    if task_id not in tasks:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    task = tasks[task_id]

    if task["status"] != "completed" or not task["audio_path"]:
        return JSONResponse(status_code=404, content={"error": "Audio not ready"})

    audio_path = task["audio_path"]

    if not os.path.exists(audio_path):
        return JSONResponse(status_code=404, content={"error": f"Audio file not found: {audio_path}"})

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"{task_id}_narration.wav"
    )

@app.post("/ask/{task_id}")
async def ask_question(task_id: str, question: str = Form(...)):
    """FIXED: Now accepts form data (question: str) instead of JSON body"""
    if task_id not in tasks:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    task = tasks[task_id]

    if task["status"] != "completed":
        return JSONResponse(status_code=400, content={"error": "Video not yet processed"})

    fused_descriptors = task.get("fused_descriptors")
    if not fused_descriptors:
        return JSONResponse(status_code=500, content={"error": "No scene data available"})

    # Use real QA retriever
    qa = QARetriever()
    qa.set_context(fused_descriptors, task.get("narration_text", ""))
    answer = await asyncio.to_thread(qa.answer, question)

    return {
        "question": question,
        "answer": answer,
        "task_id": task_id
    }

# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NETRA Backend Starting...")
    print(f"MAX_UPLOAD_SIZE: {MAX_UPLOAD_SIZE // (1024*1024)} MB")
    print("Pipeline: Real (Inference → Fusion → Summarization → TTS)")
    print("==================================================")
    print("Server: http://127.0.0.1:8000")
    print("=" * 60)

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info", timeout_keep_alive=300)