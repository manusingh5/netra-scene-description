# ui.py
"""
NETRA Frontend UI
Gradio interface for video upload, audio playback, and voice chatbot
"""

import gradio as gr
import httpx
import os
import time
from io import BytesIO

BACKEND_URL = "http://127.0.0.1:8000"

def process_video_frontend(video_file):
    """Upload video to backend and poll for completion"""
    if not video_file:
        return "❌ Please upload a video first.", None, ""
    
    try:
        if isinstance(video_file, str):
            file_path = video_file
        elif hasattr(video_file, 'name'):
            file_path = video_file.name
        else:
            return "❌ Invalid video format.", None, ""
        
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}", None, ""
        
        print(f"[DEBUG] Video file path: {file_path}")
        file_size_kb = os.path.getsize(file_path) / 1024
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"[DEBUG] File size: {file_size_mb:.2f} MB")
        
        # Increase timeout for large files
        with httpx.Client(timeout=1800.0) as client:  # 30 minute timeout
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            
            print(f"[DEBUG] Reading file complete, {len(file_bytes) / (1024*1024):.2f} MB")
            
            mime_type = "video/mp4"
            filename = os.path.basename(file_path)
            
            files = {
                "video": (
                    filename,
                    BytesIO(file_bytes),
                    mime_type
                )
            }
            
            print(f"[DEBUG] Sending request to {BACKEND_URL}/upload")
            response = client.post(
                f"{BACKEND_URL}/upload",
                files=files,
                headers={"Accept": "application/json"}
            )
            
            print(f"[DEBUG] Upload response status: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = f"❌ Upload failed ({response.status_code}): {response.text[:500]}"
                print(error_msg)
                return error_msg, None, ""
            
            json_response = response.json()
            task_id = json_response.get("task_id")
            print(f"[DEBUG] Task ID received: {task_id}")
            
            for attempt in range(1800):
                time.sleep(1)
                status_res = client.get(f"{BACKEND_URL}/status/{task_id}")
                status_data = status_res.json()
                
                status = status_data.get("status")
                progress = status_data.get("progress", 0)
                stage = status_data.get("stage", "")
                
                print(f"[DEBUG] Attempt {attempt+1}: status={status}, progress={progress}%")
                
                if status == "completed":
                    print(f"[DEBUG] Downloading audio from backend...")
                    audio_response = client.get(f"{BACKEND_URL}/audio/{task_id}")
                    
                    if audio_response.status_code == 200:
                        audio_filename = f"{task_id}_narration.wav"
                        os.makedirs("outputs", exist_ok=True)
                        audio_local_path = os.path.join("outputs", audio_filename)
                        
                        with open(audio_local_path, "wb") as f:
                            f.write(audio_response.content)
                        
                        print(f"[DEBUG] Audio saved to: {audio_local_path}")
                        return f"✅ {stage} (Complete! {file_size_mb:.1f}MB)", audio_local_path, task_id
                    else:
                        print(f"[DEBUG] Audio download failed: {audio_response.status_code}")
                        return f"✅ {stage} (Audio download failed)", None, task_id
                
                elif status == "error":
                    return f"❌ Error: {status_data.get('stage', 'Unknown error')}", None, ""
            
            return "❌ Timeout: Processing took too long", None, ""
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] {error_details}")
        return f"❌ Connection error: {str(e)}\n{error_details}", None, ""

def chat_with_bot(question_text, task_id):
    if not task_id:
        return "❌ Please process a video first."
    if not question_text or question_text.strip() == "":
        return "❌ Please ask a question."
    
    try:
        with httpx.Client(timeout=30.0) as client:
            data = {"question": question_text}
            response = client.post(f"{BACKEND_URL}/ask/{task_id}", data=data)
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer received.")
                return f"💬 Answer: {answer}"
            else:
                return f"❌ Error: {response.text}"
                
    except Exception as e:
        return f"❌ Connection error: {str(e)}"

def transcribe_and_ask(audio_file, task_id):
    if not task_id:
        return "❌ Please process a video first."
    if not audio_file:
        return "❌ Please record your question via microphone."
    
    mock_transcription = "What objects are visible in the video?"
    answer = chat_with_bot(mock_transcription, task_id)
    
    return f"🎤 You asked: {mock_transcription}\n\n{answer}"

# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="NETRA - Scene Description System") as demo:
    
    current_task_id = gr.State("")
    
    gr.Markdown("# 🎥 NETRA — Scene Description System")
    gr.Markdown("### Offline AI-Powered Video Analysis | Supports videos up to 500MB")
    
    with gr.Tab("📹 Video Upload & Description"):
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="Upload Video (Up to 500MB supported)",
                    height=400
                )
                process_btn = gr.Button("▶️ Analyze Video", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="Processing Status",
                    lines=3,
                    interactive=False
                )
                audio_output = gr.Audio(label="🎧 Audio Narration", interactive=False)
    
    with gr.Tab("🎤 Voice Chatbot"):
        gr.Markdown("### Ask questions about the video using your voice")
        with gr.Row():
            mic_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Ask a Question (Voice)"
            )
        
        ask_btn = gr.Button("💬 Ask", variant="primary")
        
        with gr.Row():
            transcription_output = gr.Textbox(
                label="You Asked (Transcribed)",
                lines=4,
                interactive=False
            )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## NETRA — Neural Environment Tracking And Real-time Analysis
        
        **NETRA** is an offline AI-powered scene description system.
        
        ### Features:
        - 📹 **Video Upload**: Up to 500MB videos supported
        - 🧠 **AI Analysis**: Object detection, OCR, scene description
        - 🎧 **Audio Narration**: Scene description converted to speech
        - 🎤 **Voice Chatbot**: Ask questions using voice
        
        ### Tech Stack:
        - **Frontend**: Gradio
        - **Backend**: FastAPI
        - **Models**: Florence-2, YOLOv8, PaddleOCR, Phi-3, Piper TTS, Whisper STT
        """)
    
    process_btn.click(
        fn=process_video_frontend,
        inputs=[video_input],
        outputs=[status_output, audio_output, current_task_id]
    )
    
    ask_btn.click(
        fn=transcribe_and_ask,
        inputs=[mic_input, current_task_id],
        outputs=[transcription_output]
    )

if __name__ == "__main__":
    print("=" * 60)
    print("NETRA Frontend UI Starting...")
    print("Make sure Backend (app.py) is running on port 8000")
    print("URL: http://localhost:7860")
    print("=" * 60)
    demo.launch(server_port=7860, show_error=True)