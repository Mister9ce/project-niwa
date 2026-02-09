import asyncio
import logging
from typing import List, Dict, Optional, Any, AsyncGenerator

from app.config import settings
from app.schemas import ForensicResponse
from app.deep_forensic_auditor import DeepForensicAuditor
from app.live_forensic_agent import LiveForensicAgent

logger = logging.getLogger("NiwaEngine")

class ForensicEngine:
    def __init__(self):
        self.deep_audit_queue = asyncio.Queue()
        self.frame_buffer: Dict[str, bytes] = {} # Stores RAW BYTES. Key = frame_id.
        self.current_live_frame_id: Optional[str] = None # Tracks the ID of the frame currently being streamed

        self.deep_auditor = DeepForensicAuditor(
            deep_audit_queue=self.deep_audit_queue,
            frame_buffer=self.frame_buffer
        )
        self.live_agent = LiveForensicAgent(
            deep_audit_queue=self.deep_audit_queue, # Live agent will put messages here for orchestrator to yield
            frame_buffer=self.frame_buffer,
            run_deep_audit_callback=self._trigger_deep_audit_from_live # Callback for escalation
        )
    
    async def connect(self):
        """Establishes WebSocket connection for the Live Forensic Agent."""
        await self.live_agent.connect()

    async def send_frame(self, frame_data: bytes, frame_id: str):
        """Sends frame to Live Forensic Agent."""
        self.current_live_frame_id = frame_id # Update orchestrator's pointer
        await self.live_agent.send_frame(frame_data, frame_id)

    async def _trigger_deep_audit_from_live(self, frame_id: str, areas_of_interest: Optional[List[Dict]] = None):
        """
        Callback to trigger a deep audit, called by the LiveForensicAgent.
        This allows the orchestrator to manage the deep audit process.
        """
        await self.deep_auditor.run_deep_audit(frame_id, areas_of_interest)

    async def run_deep_audit_manual(self, frame_id: str, areas_of_interest: Optional[List[Dict]] = None):
        """Public entry point for manual Deep Audit requests (e.g., from an upload)."""
        await self.deep_auditor.run_deep_audit(frame_id, areas_of_interest)

    async def receive(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main loop for receiving and yielding results from both agents.
        This method acts as a multiplexer for the frontend.
        """
        # Start the live agent's internal receive loop as a background task
        live_agent_receive_task = asyncio.create_task(self.live_agent.receive())

        try:
            while True:
                # Wait for results from either agent via the shared queue
                result = await self.deep_audit_queue.get()
                logger.info(f"ForensicEngine received item from queue: {result}")
                yield result
        finally:
            # Ensure the live agent's receive task is cancelled when this receive loop stops
            live_agent_receive_task.cancel()
            try:
                await live_agent_receive_task
            except asyncio.CancelledError:
                pass
    
    async def close(self):
        """Closes connections for both agents."""
        await self.live_agent.close()