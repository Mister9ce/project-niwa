import asyncio
import json
import base64
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware  # <--- CRITICAL IMPORT
from fastapi.staticfiles import StaticFiles
from app.forensic_engine import ForensicEngine
from app.config import settings # Import settings
from app.schemas import ForensicResponse # Add this import

logger = logging.getLogger("NiwaApp")

app = FastAPI(title="Project Niwa Backend")

# Create a single global instance of ForensicEngine
global_forensic_engine = ForensicEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Mount static files if you use them
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass

@app.post("/upload_and_audit")
async def upload_and_audit(file: UploadFile = File(...)):
    logger.info(f"Received file upload: {file.filename}, content type: {file.content_type}")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files are allowed."
        )

    try:
        file_content = await file.read()
        frame_id = f"upload_{file.filename}_{int(asyncio.get_event_loop().time() * 1000)}"
        
        # Use the global engine instance
        global_forensic_engine.frame_buffer[frame_id] = file_content
        
        asyncio.create_task(global_forensic_engine.run_deep_audit_manual(frame_id))
        
        return {"message": f"Deep audit started for uploaded file: {file.filename} (ID: {frame_id})"}
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {e}"
        )

@app.websocket("/ws/spatial-stream") 
async def forensic_stream(websocket: WebSocket):
    logger.info("[Niwa] Incoming Connection Request...")
    
    # Accept the handshake immediately to prevent 403
    await websocket.accept()
    logger.info("[Niwa] Connection Accepted.")
    
    # Use the global engine instance
    engine = global_forensic_engine
    
    try:
        # Initialize the Dual-Model Engine
        await engine.connect()
        logger.info("[Niwa] Engine Connected to Gemini.")

        async def receive_from_client():
            while True:
                try:
                    data = await websocket.receive_text()
                    
                    if data == "INTERRUPT":
                        await engine.live_agent.interrupt() # Call interrupt on the live_agent
                        continue

                    msg = json.loads(data)
                    action = msg.get("action", "LIVE_FRAME") # Default to live frame processing

                    if "image_data" not in msg:
                        continue

                    image_bytes = base64.b64decode(msg["image_data"])
                    frame_id = msg.get("frame_id", f"auto_{int(asyncio.get_event_loop().time() * 1000)}")

                    if settings.NIWA_MODE == "DEEP_AUDIT_ONLY":
                        if action != "DEEP_AUDIT":
                            logger.warning(f"[Niwa] NIWA_MODE is DEEP_AUDIT_ONLY, ignoring action: {action}")
                            continue
                        logger.info(f"[Niwa] Manual Deep Audit triggered for frame {frame_id}.")
                        # Store frame in shared buffer before triggering deep audit
                        engine.frame_buffer[frame_id] = image_bytes
                        await engine.run_deep_audit_manual(str(frame_id))
                    elif settings.NIWA_MODE == "LIVE_ONLY":
                        if action == "DEEP_AUDIT":
                            logger.warning(f"[Niwa] NIWA_MODE is LIVE_ONLY, ignoring deep audit request for frame: {frame_id}")
                            continue
                        await engine.send_frame(image_bytes, str(frame_id))
                    else: # settings.NIWA_MODE == "BOTH" or invalid setting
                        if action == "DEEP_AUDIT":
                            logger.info(f"[Niwa] Manual Deep Audit triggered for frame {frame_id}.")
                            # Store frame in shared buffer before triggering deep audit
                            engine.frame_buffer[frame_id] = image_bytes
                            await engine.run_deep_audit_manual(str(frame_id))
                        else: # Default action is LIVE_FRAME
                            await engine.send_frame(image_bytes, str(frame_id))
                    
                except json.JSONDecodeError:
                    logger.warning(f"[Niwa] Invalid JSON received from client. Skipping message.")
                    continue
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    logger.error(f"[Niwa] Input Error: {e}", exc_info=True) # Added exc_info
                    continue

        async def send_to_client():
            async for analysis in engine.receive():
                logger.info(f"[Niwa] Sending analysis to frontend via WebSocket: {analysis}")
                await websocket.send_json(analysis)

        # Run loops concurrently
        task_receive = asyncio.create_task(receive_from_client())
        task_send = asyncio.create_task(send_to_client())

        done, pending = await asyncio.wait(
            [task_receive, task_send],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            try:
                task.result()  # This will re-raise the exception if one occurred
            except WebSocketDisconnect:
                pass # Expected
            except Exception as e:
                logger.error(f"Task finished with unexpected exception: {e}", exc_info=True) # Added exc_info

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info("[Niwa] Client Disconnected")
    except Exception as e:
        logger.error(f"[Niwa] Critical Error: {e}", exc_info=True) # Added exc_info
    finally:
        await engine.close()
        try:
            await websocket.close()
        except Exception as e: # Added e for logging
            logger.error(f"Error closing websocket: {e}", exc_info=True) # Added logging

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)