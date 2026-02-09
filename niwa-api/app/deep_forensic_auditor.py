import json
import base64
import asyncio
import logging
import traceback
import io
import time
from datetime import datetime, timezone 
from typing import List, Dict, Optional, Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageChops
import exifread
import imagehash

from google import genai
from google.genai import types

from app.config import settings
from app.schemas import ForensicResponse, Anomaly

logger = logging.getLogger("NiwaEngine")


def _process_noise_analysis(image_bytes: bytes) -> Dict[str, Any]:
    """Calculates Laplacian variance to detect smoothing/blurring/noise inconsistencies."""
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "error", "message": "Failed to decode image bytes."}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Heuristic normalization (0.0 to 1.0 approx)
        normalized_score = min(laplacian_var / 1000.0, 1.0)
        
        return {
            "status": "success",
            "noise_inconsistency_score": normalized_score,
            "raw_variance": laplacian_var, 
            "message": "Analysis complete. Low variance (< 0.1) often implies unnatural smoothing/AI generation."
        }
    except Exception as e:
        return {"status": "error", "message": f"Noise analysis failed: {str(e)}"}

def _process_crop(image_bytes: bytes, bbox: List[float]) -> tuple[Optional[bytes], Dict[str, Any]]:
    """
    Crops image. Returns tuple: (cropped_bytes, result_metadata).
    Includes validation for normalized coordinates vs pixel coordinates.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None: 
            return None, {"status": "error", "message": "Failed to decode source image."}

        h, w, _ = img.shape
        
        # CRITICAL FIX: Validation for coordinate systems
        # If model hallucinates and sends integers (e.g. 500) instead of floats (0.5), catch it.
        if any(coord > 1.0 for coord in bbox):
            return None, {
                "status": "error", 
                "message": f"Invalid coordinates {bbox}. You sent Pixel coordinates (> 1.0). You MUST send Normalized coordinates (0.0 to 1.0). Please retry."
            }

        ymin, xmin, ymax, xmax = bbox
        
        # Convert Normalized -> Pixel
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        # Bounds checking
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None, {
                "status": "error", 
                "message": f"Calculated crop dimensions are zero or negative ({x1},{y1} to {x2},{y2}). Check your bbox coordinates."
            }

        cropped_img = img[y1:y2, x1:x2]
        
        # Re-encode to JPEG for storage/further processing
        success, buffer = cv2.imencode('.jpeg', cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success: 
            return None, {"status": "error", "message": "Failed to encode cropped region."}
             
        return buffer.tobytes(), {"status": "success", "message": "Image cropped successfully. Result stored in memory."}

    except Exception as e:
        return None, {"status": "error", "message": f"Cropping exception: {str(e)}"}

def _process_exif(image_bytes: bytes) -> Dict[str, Any]:
    """Extracts EXIF data using ExifRead."""
    try:
        f = io.BytesIO(image_bytes)
        tags = exifread.process_file(f)
        if not tags:
            return {"status": "success", "metadata": {}, "message": "No EXIF metadata found (common in generated or stripped images)."}

        # Filter out binary/thumbnail data to keep JSON light
        metadata_dict = {
            tag: str(value) for tag, value in tags.items() 
            if tag not in ('JPEGThumbnail', 'TIFFThumbnail')
        }
        return {"status": "success", "metadata": metadata_dict}
    except Exception as e:
        return {"status": "error", "message": f"EXIF extraction error: {str(e)}"}

def _process_ela(image_bytes: bytes, quality: int) -> Dict[str, Any]:
    """Generates Error Level Analysis (ELA) map."""
    try:
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Save compressed version to buffer
        buffer = io.BytesIO()
        original_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed_image = Image.open(buffer)

        # Calculate difference
        ela_diff = ImageChops.difference(original_image, recompressed_image)
        
        # Maximize contrast
        extrema = ela_diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255 / max_diff if max_diff > 0 else 1
        ela_diff = ela_diff.point(lambda i: i * scale)

        # Return visual representation (Base64)
        output_buffer = io.BytesIO()
        ela_diff.save(output_buffer, format="PNG")
        ela_b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

        return {
            "status": "success",
            "ela_image_b64": ela_b64,
            "message": "ELA map generated. High contrast or rainbowing in uniform areas suggests manipulation."
        }
    except Exception as e:
        return {"status": "error", "message": f"ELA failed: {str(e)}"}

def _process_phash(image_bytes: bytes, hash_type: str) -> Dict[str, Any]:
    """Calculates perceptual hashes."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if hash_type == "phash": 
            h = str(imagehash.phash(image))
        elif hash_type == "dhash": 
            h = str(imagehash.dhash(image))
        else: 
            h = str(imagehash.ahash(image))
        
        return {"status": "success", "hash_type": hash_type, "hash_value": h}
    except Exception as e:
        return {"status": "error", "message": f"Hash calculation failed: {str(e)}"}

def _get_current_datetime_tool() -> Dict[str, Any]:
    """
    Returns the current UTC date and time.
    Models can use this tool to get real-world time context for their analysis,
    e.g., to identify if a date in EXIF metadata is in the past, present, or future.
    """
    try:
        current_utc_time = datetime.now(timezone.utc).isoformat()
        return {"status": "success", "current_utc_datetime": current_utc_time}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get current datetime: {str(e)}"}


# ==========================================
# DEEP FORENSIC AUDITOR
# ==========================================

class DeepForensicAuditor:
    def __init__(self, deep_audit_queue: asyncio.Queue, frame_buffer: Dict[str, bytes]):
        self.api_key = settings.GOOGLE_API_KEY
        self.deep_audit_queue = deep_audit_queue
        self.frame_buffer = frame_buffer # Reference to shared frame buffer
        self.genai_client = genai.Client(api_key=self.api_key)

        # --- DEEP AUDIT TOOL CONFIGURATION ---
        tools_list = [
            {
                "name": "_noise_analysis_tool",
                "description": "Analyzes an image for noise consistency. Returns a score and variance.",
                "parameters": {
                    "type": "object",
                    "properties": {"frame_id": {"type": "string", "description": "ID of the frame to analyze."}},
                    "required": ["frame_id"]
                }
            },
            {
                "name": "_crop_image_tool",
                "description": "Crops an image. Returns a NEW frame_id for the cropped area. Does NOT return visual data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frame_id": {"type": "string"},
                        "bbox": {
                            "type": "array", 
                            "items": {"type": "number"}, 
                            "description": "Normalized coordinates [ymin, xmin, ymax, xmax] (0.0 to 1.0). Do not use pixels."
                        },
                        "new_frame_id": {"type": "string", "description": "ID to assign to the new crop."}
                    },
                    "required": ["frame_id", "bbox", "new_frame_id"]
                }
            },
            {
                "name": "_extract_exif_metadata_tool",
                "description": "Extracts EXIF metadata tags.",
                "parameters": {
                    "type": "object", 
                    "properties": {"frame_id": {"type": "string"}}, 
                    "required": ["frame_id"]
                }
            },
            {
                "name": "_perform_ela_tool",
                "description": "Performs Error Level Analysis. Returns a base64 string of the ELA visualization.",
                "parameters": {
                    "type": "object", 
                    "properties": {"frame_id": {"type": "string"}, "quality": {"type": "integer", "default": 90}}, 
                    "required": ["frame_id"]
                }
            },
            {
                "name": "_calculate_perceptual_hash_tool",
                "description": "Calculates pHash/dHash/aHash.",
                "parameters": {
                    "type": "object", 
                    "properties": {"frame_id": {"type": "string"}, "hash_type": {"type": "string", "enum": ["phash", "ahash", "dhash"]}}, 
                    "required": ["frame_id"]
                }
            },
            { # New Tool Declaration
                "name": "_get_current_datetime_tool",
                "description": "Returns the current UTC date and time in ISO 8601 format. Use this to get real-world time context.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        ]

        self.auditor_generation_config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=tools_list)],
            temperature=0.4, # Balanced for reasoning
        )

    async def run_deep_audit(self, frame_id: str, areas_of_interest: Optional[List[Dict]] = None):
        """Public entry point to trigger the independent Deep Audit Agent."""
        if frame_id not in self.frame_buffer:
            logger.error(f"Deep Audit request for {frame_id} failed: Frame not in buffer.")
            return
        
        # Retrieve bytes (thread-safe copy)
        frame_bytes = self.frame_buffer[frame_id]
        
        # Spawn background task so the websocket loop isn't blocked
        asyncio.create_task(self._run_deep_audit_internal(frame_bytes, frame_id, areas_of_interest))

    async def _run_deep_audit_internal(self, initial_frame_bytes: bytes, initial_frame_id: str, areas_of_interest: list):
        """
        The "Deep Thinking" Agent Loop.
        Uses the high-intelligence model to call tools, analyze results, and form a verdict.
        """
        logger.info(f"[[DEEP AUDIT STARTED: {initial_frame_id}]]")
        
        # Ensure frame is in buffer (redundant safety)
        self.frame_buffer[initial_frame_id] = initial_frame_bytes

        # Initial System Prompt
        prompt = f"""
        You are an expert Forensic Image Analyst.
        Target Frame ID: "{initial_frame_id}".
        
        **Objective**: Determine if the image is AUTHENTIC, SUSPICIOUS, or SYNTHETIC_CONFIRMED.
        
        **IMPORTANT**: Before any other analysis, you MUST call the `_get_current_datetime_tool()` to establish the current date and time context for your analysis, especially when evaluating timestamps in EXIF data or other image metadata.

        **Workflow**:
        1. Use `_noise_analysis_tool` and `_extract_exif_metadata_tool` on the main frame.
        2. If metadata is suspicious (Adobe, missing, future dates) or noise is low (<0.1), investigate further.
        3. Use `_crop_image_tool` to isolate specific suspicious regions (faces, text, background).
           - NOTE: You must provide NORMALIZED coordinates (0.0 to 1.0) for crops.
           - Analyze the *returned* `new_frame_id` with noise analysis or ELA.
        4. Use `_perform_ela_tool` if you suspect splicing or modification.
        
        **Output Format**:
        Return a JSON object:
        {{
            "verdict": "AUTHENTIC" | "SUSPICIOUS" | "SYNTHETIC_CONFIRMED",
            "anomalies": [
                {{
                    "box_2d": [ymin, xmin, ymax, xmax], // Integers 0-1000
                    "label": "short description",
                    "confidence": 0.0-1.0,
                    "reasoning": "detailed explanation"
                }}
            ],
            "audit_notes": "Summary of your findings."
        }}
        """

        # Initialize History with the Image + Prompt
        history = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=prompt),
                    # Provide visual context to the model once at the start
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=initial_frame_bytes))
                ]
            )
        ]

        MAX_TURNS = 12
        turn_count = 0

        try:
            while turn_count < MAX_TURNS:
                turn_count += 1
                
                # 1. Model Generation Step
                response = await self.genai_client.aio.models.generate_content(
                    model=f"models/{settings.AUDITOR_MODEL}",
                    contents=history,
                    config=self.auditor_generation_config
                )
                
                # Log the whole raw response object
                logger.info(f"Full Raw Deep Audit Agent Response Object: {response}")

                # Append model response to history
                history.append(response.candidates[0].content)
                candidate = response.candidates[0]

                # 2. Tool Execution Logic
                tool_calls = []
                for part in candidate.content.parts:
                    if part.function_call:
                        tool_calls.append(part.function_call)

                if tool_calls:
                    # The model wants to run tools
                    for tc in tool_calls:
                        fname = tc.name
                        fargs = dict(tc.args)
                        logger.info(f"[Deep Audit Tool] {fname} called with {fargs}")

                        # Execute Tool locally
                        try:
                            # 2a. Resolve Frame ID or call generic tool
                            if fname == "_get_current_datetime_tool": # Handle new tool
                                result_data = _get_current_datetime_tool()
                            else: # Existing image processing tools require frame_id
                                target_id = fargs.get("frame_id")
                                if not target_id or target_id not in self.frame_buffer:
                                    result_data = {"status": "error", "message": f"Frame ID '{target_id}' not found in memory buffer."}
                                else:
                                    target_bytes = self.frame_buffer[target_id]
                                    
                                    # 2b. Dispatch to Helper Functions
                                    if fname == "_noise_analysis_tool":
                                        result_data = _process_noise_analysis(target_bytes)
                                    
                                    elif fname == "_extract_exif_metadata_tool":
                                        result_data = _process_exif(target_bytes)
                                    
                                    elif fname == "_perform_ela_tool":
                                        qual = int(fargs.get("quality", 90))
                                        result_data = _process_ela(target_bytes, qual)
                                    
                                    elif fname == "_calculate_perceptual_hash_tool":
                                        htype = fargs.get("hash_type", "phash")
                                        result_data = _process_phash(target_bytes, htype)
                                    
                                    elif fname == "_crop_image_tool":
                                        bbox = fargs.get("bbox")
                                        new_id = fargs.get("new_frame_id")
                                        # Perform Crop
                                        cropped_bytes, status = _process_crop(target_bytes, bbox)
                                        result_data = status
                                        if cropped_bytes:
                                            # Save crop to buffer so next turn can reference 'new_id'
                                            self.frame_buffer[new_id] = cropped_bytes
                                            result_data["new_frame_id"] = new_id
                                            result_data["note"] = "Crop created. Call other tools using new_frame_id."
                                    else:
                                        result_data = {"error": f"Unknown tool: {fname}"}

                        except Exception as tool_err:
                            logger.error(f"Tool Execution Error: {tool_err}")
                            result_data = {"status": "error", "message": str(tool_err)}

                        # 2c. Feed Result back to Model
                        history.append(types.Content(
                            role="function",
                            parts=[types.Part.from_function_response(name=fname, response=result_data)]
                        ))
                else: # No tools called in this turn
                    # Concatenate all text parts from the response for the final verdict
                    final_text_parts = []
                    for part in candidate.content.parts:
                        if part.text:
                            final_text_parts.append(part.text)
                    final_text = " ".join(final_text_parts) if final_text_parts else None

                    if final_text is None:
                        logger.error(f"Deep Audit: Model finished without text response or tool call for frame {initial_frame_id}. Full response: {response}")
                        await self.deep_audit_queue.put(ForensicResponse(
                            frame_id=initial_frame_id,
                            source_model="DEEP_AUDIT",
                            verdict="ERROR",
                            anomalies=[],
                            audit_notes=f"Model did not return a final text response. Full response: {response}"
                        ).model_dump())
                        break # Exit loop as no further processing is possible
                    
                    logger.info(f"[[DEEP AUDIT VERDICT]]: {final_text}")
                    
                    try:
                        # Clean Markdown
                        clean_json = final_text.replace("```json", "").replace("```", "").strip()
                        if "{" not in clean_json: 
                            raise ValueError("No JSON found")
                        
                        start = clean_json.find("{")
                        end = clean_json.rfind("}") + 1
                        data = json.loads(clean_json[start:end])
                        
                        # Validate/Convert Schema
                        anomalies_list = []
                        for a in data.get("anomalies", []):
                            box = a.get("box_2d", [0,0,0,0])
                            # Fix: Ensure integers 0-1000
                            if any(isinstance(x, float) for x in box):
                                box = [int(x * 1000) for x in box]
                            
                            anomalies_list.append(Anomaly(
                                box_2d=box,
                                label=a.get("label", "Unknown"),
                                confidence=float(a.get("confidence", 0.0)),
                                reasoning=a.get("reasoning", "")
                            ))

                        response_obj = ForensicResponse(
                            frame_id=initial_frame_id,
                            source_model="DEEP_AUDIT",
                            verdict=data.get("verdict", "SUSPICIOUS"),
                            anomalies=anomalies_list,
                            audit_notes=data.get("audit_notes", "Analysis completed.")
                        )
                        
                        # Push to Queue for Frontend
                        await self.deep_audit_queue.put(response_obj.model_dump())
                        logger.info(f"Deep audit result (dict) put into queue: {response_obj.model_dump()}")
                        return # Exit Loop

                    except Exception as e:
                        logger.error(f"JSON Parse Error in Audit: {e}")
                        # Provide fallback response
                        await self.deep_audit_queue.put(ForensicResponse(
                            frame_id=initial_frame_id,
                            source_model="DEEP_AUDIT",
                            verdict="SUSPICIOUS",
                            anomalies=[],
                            audit_notes=f"Audit completed but response was malformed: {final_text[:100]}..."
                        ).model_dump())
                        return

        except Exception as e:
            logger.error(f"Critical Deep Audit Failure: {e}")
            traceback.print_exc()