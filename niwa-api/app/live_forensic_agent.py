import json
import base64
import asyncio
import logging
import traceback
import time
from typing import List, Dict, Optional, Any

from websockets import connect

from app.config import settings
from app.schemas import ForensicResponse

LIVE_RESPONSE_TIMEOUT = 5 # seconds

logger = logging.getLogger("NiwaEngine")

class LiveForensicAgent:
    def __init__(self, deep_audit_queue: asyncio.Queue, frame_buffer: Dict[str, bytes], run_deep_audit_callback):
        self.api_key = settings.GOOGLE_API_KEY
        self.live_uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={self.api_key}"
        
        self.ws = None
        self.json_buffer = "" # Used for accumulating JSON chunks
        self.last_message_time = time.time() # Track last message received
        self.frame_buffer = frame_buffer # Reference to shared frame buffer
        self.current_live_frame_id: Optional[str] = None # Tracks the ID of the frame currently being streamed
        self.deep_audit_queue = deep_audit_queue # For yielding results to the orchestrator
        self.run_deep_audit_callback = run_deep_audit_callback # Callback to trigger deep audit in orchestrator

    async def connect(self):
        """Establishes WebSocket connection to Gemini Live with protocol setup."""
        try:
            # Re-add ping_interval and ping_timeout for robustness
            self.ws = await connect(self.live_uri, 
                                    ping_interval=30, 
                                    ping_timeout=40,
                                    additional_headers={"Content-Type": "application/json"})
            
            setup_msg = {
                "setup": {
                    "model": f"models/{settings.LIVE_MODEL}",
                    "generation_config": {
                        "response_modalities": ["AUDIO"], 
                        "temperature": 0.2
                    },
                    "system_instruction": {
                        "parts": [{"text": settings.LIVE_SYSTEM_INSTRUCTION}]
                    },
                    "outputAudioTranscription": {} 
                }
            }
            logger.info(f"Connecting with setup: {json.dumps(setup_msg)}")
            await self.ws.send(json.dumps(setup_msg))
            
            ack = await self.ws.recv()
            logger.info(f"Live API Connected. ACK: {len(ack)} bytes")
            
        except Exception as e:
            logger.error(f"Live Connection Failed: {e}", exc_info=True)
            raise e

    async def send_frame(self, frame_data: bytes, frame_id: str):
        """Sends a single frame to the Live Model (WebSocket)."""
        if not self.ws: return

        # Store the raw frame data in the buffer
        self.frame_buffer[frame_id] = frame_data
        self.current_live_frame_id = frame_id # Update pointer for escalation
        
        b64_img = base64.b64encode(frame_data).decode('utf-8')
        
        msg = {
            "clientContent": {
                "turns": [{
                    "role": "user",
                    "parts": [
                        {"text": f"FRAME_ID: {frame_id}"},
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64_img}}
                    ]
                }],
                "turnComplete": True 
            }
        }
        
        logger.info(f"Sending frame {frame_id} to Live Model. Message size: {len(json.dumps(msg))} bytes")
        logger.info(f"Sent to Gemini WebSocket: {json.dumps(msg)[:500]}...")
        try:
            await self.ws.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"Send Frame Error: {e}")

    async def interrupt(self):
        if self.ws:
            msg = { "clientContent": { "turns": [{"role": "user", "parts": [{"text": "Stop generation."}]}], "turnComplete": True } }
            try:
                await self.ws.send(json.dumps(msg))
                logger.info("Sent interrupt signal to live model.")
            except Exception as e:
                logger.warning(f"Failed to send interrupt signal: {e}")

    async def close(self):
        if self.ws: await self.ws.close()

    async def receive(self):
        """Yields JSON results from the Live Orchestrator Stream."""
        if not self.ws: return
        
        # Continuously check for results from the live stream
        while True:
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=0.1) 
                self.last_message_time = time.time() # Update last message time
                logger.info(f"Received message from WebSocket. Length: {len(message)}")
                logger.info(f"Raw message received from WebSocket: {message[:500]}...")
            except asyncio.TimeoutError:
                # If no message within timeout, check if buffer needs clearing
                if self.json_buffer and (time.time() - self.last_message_time > LIVE_RESPONSE_TIMEOUT):
                    logger.warning(f"[Niwa] Live JSON buffer timed out. Clearing buffer ({len(self.json_buffer)} chars). Content: {self.json_buffer[:200]}...")
                    self.json_buffer = ""
                await asyncio.sleep(0.01) # Short sleep to prevent busy-waiting
                continue
            except Exception as e:
                logger.error(f"[Niwa] Live Stream Receive Error: {e}", exc_info=True)
                break

            try:
                response = json.loads(message)
                server_content = response.get("serverContent", {})
                
                # Extract all textual content regardless of its source (outputTranscription or modelTurn)
                current_text_chunk = ""
                if server_content.get("outputTranscription", {}).get("text"):
                    current_text_chunk = server_content["outputTranscription"]["text"]
                else:
                    model_turn_parts = server_content.get("modelTurn", {}).get("parts", [])
                    for part in model_turn_parts:
                        if part.get("text"):
                            current_text_chunk += part["text"]

                if current_text_chunk:
                    self.json_buffer += current_text_chunk
                    logger.debug(f"Current JSON buffer state: {self.json_buffer[:500]}...") # Debug log for buffer

                   
                    # Keep trying to extract JSON blocks until no more are found
                    while True:
                        json_start_tag = "```json"
                        json_end_tag = "```"

                        start_idx = self.json_buffer.find(json_start_tag)
                        
                        if start_idx == -1: # No JSON block found yet
                            # Check if the buffer is just conversational text and needs clearing
                            if len(self.json_buffer) > 1000: # Arbitrary large size
                                logger.warning(f"[Niwa] Live JSON buffer growing with conversational text. Clearing. Content: {self.json_buffer[:200]}...")
                                self.json_buffer = ""
                            break # No JSON block, wait for more messages

                        # Found start tag, now look for end tag
                        end_idx = self.json_buffer.find(json_end_tag, start_idx + len(json_start_tag))

                        if end_idx == -1: # Start tag found, but end tag not yet. Keep buffering.
                            break 
                        
                        # Full JSON block found
                        json_str = self.json_buffer[start_idx + len(json_start_tag) : end_idx].strip()
                        
                        try:
                            data = json.loads(json_str)
                            logger.info(f"[Niwa] Successfully parsed JSON from LIVE_MODEL: {data}")
                            data["source_model"] = "LIVE_HUD" 
                            
                            # --- ESCALATION TRIGGER from Live Agent ---
                            if data.get("request_deep_audit"): 
                                logger.info(f"[Live Agent] Live Model requested deep audit. Triggering via callback.")
                                if self.current_live_frame_id:
                                    await self.deep_audit_queue.put(ForensicResponse(
                                        frame_id=self.current_live_frame_id,
                                        source_model="LIVE_HUD",
                                        status="ESCALATING",
                                        audit_notes=f"⚠️ Live Model requested Deep Audit for frame {self.current_live_frame_id}",
                                        verdict="SUSPICIOUS",
                                        anomalies=[]
                                    ).model_dump())
                                    await self.run_deep_audit_callback(self.current_live_frame_id)
                                else:
                                    logger.warning("Live Model requested deep audit but no current frame ID available.")
                            else:
                                # Normal Live Update
                                forensic_response = ForensicResponse(
                                    frame_id="LIVE",
                                    source_model="LIVE_HUD",
                                    verdict=data.get("verdict", "AUTHENTIC"),
                                    anomalies=data.get("anomalies", []),
                                    audit_notes=data.get("notes", "")
                                )
                                logger.info(f"Live Agent yielding ForensicResponse: {forensic_response.verdict}")
                                await self.deep_audit_queue.put(forensic_response.model_dump())
                            
                            # Remove processed JSON block from buffer and continue checking for more
                            self.json_buffer = self.json_buffer[end_idx + len(json_end_tag):].lstrip()
                            logger.debug(f"Buffer after processing JSON: {self.json_buffer[:200]}...")

                        except json.JSONDecodeError:
                            logger.warning(f"[Niwa] Incomplete or malformed JSON found within '```json' block. Keeping in buffer. Current JSON buffer: {self.json_buffer[:500]}...")
                            # Don't clear buffer here, it's an incomplete JSON block that might complete in next message
                            break # Wait for more chunks to complete this JSON
                        except Exception as e:
                            logger.error(f"[Niwa] Error processing extracted JSON: {e}", exc_info=True)
                            self.json_buffer = "" # Clear buffer on unrecoverable error
                            break # Exit inner loop

                else: # No new text chunk in this message
                    logger.debug(f"[Niwa] LIVE_MODEL message with no text content. Full message: {response}")
                    # No text, so no potential JSON to parse. Buffer might still contain partial JSON.

            except Exception as e:
                logger.error(f"[Niwa] Receive Error: {e}", exc_info=True)
                self.json_buffer = "" # Clear buffer on outer error
                continue
