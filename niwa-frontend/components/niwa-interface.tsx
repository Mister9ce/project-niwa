'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';

// --- Constants ---
const FRAME_RATE = 0.5; // FPS for sending frames to backend
const WS_URL = `ws://127.0.0.1:8000/ws/spatial-stream`; // Define WebSocket URL

// Define ForensicResponse and Anomaly types for frontend consistency
interface Anomaly {
    box_2d: number[]; // [ymin, xmin, ymax, xmax] normalized 0-1000
    label: string;
    confidence: number;
    reasoning: string;
}

interface ForensicResponse {
    frame_id: string;
    source_model: string;
    verdict: 'AUTHENTIC' | 'SUSPICIOUS' | 'SYNTHETIC_CONFIRMED' | 'ERROR' | 'ESCALATING';
    anomalies: Anomaly[];
    audit_notes: string;
    request_deep_audit?: boolean;
    status?: string; // For escalation status from live agent
}

// --- Main Component ---
export default function NiwaInterface() {
    // --- State Management ---
    const [isStreaming, setIsStreaming] = useState(false);
    const [isLoading, setIsLoading] = useState(false); // New state for general loading
    const [log, setLog] = useState<{ time: string; msg: string; type?: 'success' | 'error' | 'audit' }[]>([]);
    const [verdict, setVerdict] = useState<{ text: string; type: 'authentic' | 'suspicious' | 'synthetic' | '' }>({ text: '', type: '' });
    const [forensicResults, setForensicResults] = useState<ForensicResponse | null>(null); // New state for full results
    const [hoveredAnomaly, setHoveredAnomaly] = useState<Anomaly | null>(null); // New state for hover effect
    const [uploadedImageForAuditSrc, setUploadedImageForAuditSrc] = useState<string | null>(null); // To display uploaded image

    // --- Refs for direct DOM/Object access ---
    const videoRef = useRef<HTMLVideoElement>(null);
    const imageDisplayRef = useRef<HTMLImageElement>(null); // Ref for uploaded image display
    const hudCanvasRef = useRef<HTMLCanvasElement>(null);
    const logRef = useRef<HTMLDivElement>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const websocketRef = useRef<WebSocket | null>(null);
    const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // --- Logging Utility ---
    const addLog = useCallback((msg: string, type?: 'success' | 'error' | 'audit') => {
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        setLog(prevLogs => [...prevLogs, { time, msg, type }]);
    }, []);

    // Auto-scroll log
    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [log]);

    // --- Core Functions ---
    const drawHudElements = useCallback(() => {
        const canvas = hudCanvasRef.current;
        const video = videoRef.current;
        const image = imageDisplayRef.current; // Use image ref for audit mode
        const targetElement = isStreaming ? video : image; // Dynamically choose source

        if (!canvas || !targetElement) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const naturalWidth = isStreaming ? video?.videoWidth : image?.naturalWidth;
        const naturalHeight = isStreaming ? video?.videoHeight : image?.naturalHeight;

        // Ensure canvas matches resolution of the displayed media
        const w = naturalWidth || targetElement.clientWidth;
        const h = naturalHeight || targetElement.clientHeight;
        
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!forensicResults) return;

        // Scale functions (normalize 0-1000 to current width/height)
        const scaleX = (val: number) => (val / 1000) * w;
        const scaleY = (val: number) => (val / 1000) * h;

        // Draw the bracket for the verdict (overall frame, if needed)
        const drawVerdictBracket = (color: string) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 5;
            ctx.lineCap = 'square';
            const offset = 10;
            const lineLen = Math.min(w, h) * 0.05;

            ctx.beginPath();
            // Top-left
            ctx.moveTo(offset, offset + lineLen); ctx.lineTo(offset, offset); ctx.lineTo(offset + lineLen, offset);
            // Top-right
            ctx.moveTo(w - offset - lineLen, offset); ctx.lineTo(w - offset, offset); ctx.lineTo(w - offset, offset + lineLen);
            // Bottom-right
            ctx.moveTo(w - offset, h - offset - lineLen); ctx.lineTo(w - offset, h - offset); ctx.lineTo(w - offset - lineLen, h - offset);
            // Bottom-left
            ctx.moveTo(offset + lineLen, h - offset); ctx.lineTo(offset, h - offset); ctx.lineTo(offset, h - offset - lineLen);
            ctx.stroke();
        };

        let verdictColor = "#00ff00"; // AUTHENTIC
        if (forensicResults.verdict === "SUSPICIOUS") verdictColor = "#ffff00";
        else if (forensicResults.verdict === "SYNTHETIC_CONFIRMED" || forensicResults.verdict === "ERROR") verdictColor = "#ff0000";
        else if (forensicResults.verdict === "ESCALATING") verdictColor = "#00bfff"; // Light blue for escalating

        drawVerdictBracket(verdictColor);
        
        // Bounding boxes are now handled by absolutely positioned HTML elements
        // The HUD will primarily draw the overall verdict bracket and possibly the main verdict text
        // (Verdict text is already handled by a div overlay in JSX)
    }, [forensicResults, isStreaming]);

    // Effect to redraw HUD when forensicResults or streaming state changes
    useEffect(() => {
        drawHudElements();
    }, [forensicResults, isStreaming, drawHudElements]);
    
    // stopStream must be declared before ensureWebSocketConnected because ensureWebSocketConnected uses it.
    const stopStream = useCallback(() => {
        setIsStreaming(false);
        setIsLoading(false); // Stop loading on termination
        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current);
            frameIntervalRef.current = null;
        }
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(track => track.stop());
            mediaStreamRef.current = null;
        }
        if (websocketRef.current) {
            websocketRef.current.close();
            websocketRef.current = null;
        }
        if (hudCanvasRef.current) {
            const ctx = hudCanvasRef.current.getContext('2d');
            ctx?.clearRect(0, 0, hudCanvasRef.current.width, hudCanvasRef.current.height);
        }
        setVerdict({ text: '', type: '' });
        setForensicResults(null); // Clear all results
        setUploadedImageForAuditSrc(null); // Clear uploaded image
        addLog("System: Standby Mode.");
    }, [addLog, setVerdict]);

    // Reusable WebSocket Connection Logic
    const ensureWebSocketConnected = useCallback((): Promise<WebSocket> => {
        return new Promise((resolve, reject) => {
            if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
                resolve(websocketRef.current);
                return;
            }

            if (websocketRef.current && websocketRef.current.readyState === WebSocket.CONNECTING) {
                // If already connecting, wait for it to open
                const originalOnOpen = websocketRef.current.onopen;
                websocketRef.current.onopen = (event) => {
                    originalOnOpen?.(event);
                    resolve(websocketRef.current!);
                };
                const originalOnError = websocketRef.current.onerror;
                websocketRef.current.onerror = (event) => {
                    originalOnError?.(event);
                    reject(event);
                };
                return;
            }

            addLog(`System: Dialing Neural Uplink (${WS_URL})...`);
            websocketRef.current = new WebSocket(WS_URL);

            websocketRef.current.onopen = () => {
                addLog("Uplink Established.", 'success');
                resolve(websocketRef.current!);
            };

            websocketRef.current.onmessage = (event) => {
                try {
                    const response: ForensicResponse = JSON.parse(event.data);
                    console.log("Received ForensicResponse:", response); // Log the full object
                    setForensicResults(response);
                    setIsLoading(false); // Stop loading when first response comes in (or after escalation)

                    if (response.audit_notes) {
                        addLog(`${response.source_model}: ${response.audit_notes}`, response.verdict === 'SYNTHETIC_CONFIRMED' || response.verdict === 'ERROR' ? 'error' : response.verdict === 'AUTHENTIC' ? 'success' : 'audit');
                    } else if (response.status === 'ESCALATING') {
                         addLog(`LIVE_HUD: ${response.audit_notes}`, 'audit');
                    } else {
                        addLog(`Received response from ${response.source_model} with verdict: ${response.verdict}`, 'audit');
                    }

                    // Update verdict text for HUD overlay
                    let type: 'authentic' | 'suspicious' | 'synthetic' | '' = '';
                    if (response.verdict === 'AUTHENTIC') type = 'authentic';
                    else if (response.verdict === 'SUSPICIOUS') type = 'suspicious';
                    else if (response.verdict === 'SYNTHETIC_CONFIRMED' || response.verdict === 'ERROR') type = 'synthetic';
                    else if (response.verdict === 'ESCALATING') type = 'suspicious'; // Treat escalating as suspicious for color
                    setVerdict({ text: response.verdict, type });

                } catch (e) {
                    console.error("Failed to parse or draw HUD:", e);
                    addLog("Error: Received malformed data packet.", 'error');
                    setIsLoading(false);
                }
            };

            websocketRef.current.onclose = () => {
                addLog("Uplink Terminated.", 'error');
                stopStream(); // Call stopStream to clean up everything
            };

            websocketRef.current.onerror = (err) => {
                addLog("Connection Error. Check console.", 'error');
                console.error("WebSocket Error:", err);
                setIsLoading(false);
                reject(err);
            };
        });
    }, [addLog, setForensicResults, setIsLoading, setVerdict, stopStream]);

    const startStream = useCallback(async (sourceType: 'camera' | 'screen') => {
        if (isStreaming || isLoading) {
            addLog("System: Session already active or loading.", 'error');
            return;
        }
        setIsLoading(true); // Start loading
        setForensicResults(null); // Clear previous results
        setUploadedImageForAuditSrc(null); // Clear uploaded image
        addLog(`System: Initializing session with ${sourceType.toUpperCase()} source...`);

        try {
            // 1. Get Media Stream
            if (sourceType === 'camera') {
                mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false
                });
            } else {
                mediaStreamRef.current = await navigator.mediaDevices.getDisplayMedia({
                    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false
                });
            }

            if (videoRef.current) {
                videoRef.current.srcObject = mediaStreamRef.current;
                videoRef.current.play().catch(e => console.error("Video play failed:", e));
            }
            addLog("System: Optical sensor active.");

            // 2. Ensure WebSocket Connected
            await ensureWebSocketConnected();
            setIsStreaming(true); // Only set streaming to true if WS and media stream are active
            setIsLoading(false);

        } catch (err: unknown) {
            const error = err as Error;
            addLog(`Init Failed: ${error.name}: ${error.message}`, 'error');
            stopStream();
            setIsLoading(false);
        }
    }, [isStreaming, isLoading, addLog, ensureWebSocketConnected, stopStream]);

    const runDeepAudit = useCallback(() => {
        if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) {
            addLog("System: Cannot request Deep Audit without an active WebSocket connection. Please 'Init System' first.", 'error');
            return;
        }
        addLog("System: Requesting Manual Deep Scan...", 'audit');
        setIsLoading(true); // Start loading for deep audit

        const video = videoRef.current;
        if (!video || video.videoWidth === 0) {
            addLog("System: Video feed not available for audit.", 'error');
            setIsLoading(false);
            return;
        }

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx?.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        const base64Image = tempCanvas.toDataURL('image/jpeg', 0.95).split(',')[1];

        // Frame ID for live stream is used as the current target for deep audit
        websocketRef.current.send(JSON.stringify({
            action: "DEEP_AUDIT",
            frame_id: Date.now().toString(), // Use a new ID or the last sent live frame ID if preferred
            image_data: base64Image
        }));
    }, [addLog]); // Removed isStreaming from deps as it's not strictly required here if WS is active.

    const uploadAndAudit = useCallback(async () => {
        if (isLoading) {
            addLog("System: Already processing an audit.", 'error');
            return;
        }
        if (isStreaming) {
            addLog("System: Please terminate live stream before uploading for audit.", 'error');
            return;
        }

        if (fileInputRef.current && fileInputRef.current.files && fileInputRef.current.files.length > 0) {
            const file = fileInputRef.current.files[0];
            addLog(`System: Uploading file "${file.name}" for Deep Audit...`, 'audit');
            setIsLoading(true); // Start loading

            const reader = new FileReader();
            reader.onloadend = async () => {
                const base64Image = reader.result as string;
                setUploadedImageForAuditSrc(base64Image); // Display uploaded image
                setForensicResults(null); // Clear previous results

                try {
                    // Ensure WebSocket is connected before sending HTTP upload
                    await ensureWebSocketConnected();

                    const formData = new FormData();
                    formData.append('file', file);

                    const response = await fetch(`${process.env.NEXT_PUBLIC_NIWA_API_BASE_URL}/upload_and_audit`, {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    addLog(`Upload successful: ${data.message}`, 'success');
                    // Results will come via WebSocket onmessage handler (already set up by ensureWebSocketConnected)
                } catch (error: any) {
                    console.error("Upload failed:", error);
                    addLog(`Upload failed: ${error.message}`, 'error');
                    setIsLoading(false);
                } finally {
                    if (fileInputRef.current) {
                        fileInputRef.current.value = ''; // Clear file input
                    }
                }
            };
            reader.readAsDataURL(file); // Read file as base64
        } else {
            addLog("System: No file selected for upload.", 'error');
        }
    }, [isLoading, isStreaming, addLog, ensureWebSocketConnected, setForensicResults, setUploadedImageForAuditSrc]);

    // Effect for starting and stopping the frame sending interval
    useEffect(() => {
        if (isStreaming && !uploadedImageForAuditSrc) { // Only send frames if streaming and not displaying an uploaded image
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');

            frameIntervalRef.current = setInterval(() => {
                const video = videoRef.current;
                const ws = websocketRef.current;

                if (ws && ws.readyState === WebSocket.OPEN && video && video.videoWidth > 0) {
                    tempCanvas.width = video.videoWidth;
                    tempCanvas.height = video.videoHeight;
                    tempCtx?.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
                    const base64Image = tempCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];
                    const currentFrameId = Date.now().toString(); // Generate unique frame_id

                    ws.send(JSON.stringify({
                        action: "LIVE_FRAME",
                        frame_id: currentFrameId,
                        image_data: base64Image
                    }));
                }
            }, 1000 / FRAME_RATE);
        }
        // Cleanup function
        return () => {
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current);
            }
        };
    }, [isStreaming, uploadedImageForAuditSrc]);

    // Effect for cleaning up on component unmount
    useEffect(() => {
        return () => stopStream();
    }, [stopStream]);

    // --- Render ---
    // Calculate display size for image/video and canvas
    const displayWidth = uploadedImageForAuditSrc ? 800 : (videoRef.current?.videoWidth || 800);
    const displayHeight = uploadedImageForAuditSrc ? (imageDisplayRef.current?.naturalHeight && imageDisplayRef.current?.naturalWidth ? (imageDisplayRef.current.naturalHeight / imageDisplayRef.current.naturalWidth) * 800 : 600) : (videoRef.current?.videoHeight || 600);


    return (
        <div className="niwa-container">
            {/* Title */}
            <h2 className="niwa-title">
                PROJECT NIWA <span className="niwa-version">// V.3.0 ALPHA</span>
            </h2>

            <div style={{ display: 'flex', gap: '20px' }}>
                {/* Main Content Area (Video/Image + HUD) */}
                <div style={{ position: 'relative', width: displayWidth, height: displayHeight, border: '1px solid #00f3ff' }}>
                    {isLoading && (
                        <div style={{
                            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                            display: 'flex', justifyContent: 'center', alignItems: 'center',
                            color: '#00f3ff', fontSize: '2em', zIndex: 100
                        }}>
                            <div className="spinner"></div> {/* Basic spinner, add CSS for animation */}
                            <p style={{marginLeft: '10px'}}>ANALYZING...</p>
                        </div>
                    )}
                    {uploadedImageForAuditSrc ? (
                        <img ref={imageDisplayRef} src={uploadedImageForAuditSrc} alt="Uploaded for Audit" style={{ width: '100%', height: '100%', objectFit: 'contain' }} onLoad={drawHudElements}/>
                    ) : (
                        <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                    )}
                    <canvas ref={hudCanvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
                    
                    {/* Verdict Overlay */}
                    <div className={`niwa-verdict niwa-verdict-${verdict.type}`}>
                        {verdict.text}
                    </div>

                    {/* Anomaly Bounding Boxes (Interactive) */}
                    {forensicResults?.anomalies && (
                        forensicResults.anomalies.map((anomaly, index) => {
                            // Calculate pixel coordinates from normalized 0-1000 values
                            const [ymin, xmin, ymax, xmax] = anomaly.box_2d;
                            const x = (xmin / 1000) * (uploadedImageForAuditSrc ? (imageDisplayRef.current?.clientWidth || 0) : (videoRef.current?.videoWidth || 0));
                            const y = (ymin / 1000) * (uploadedImageForAuditSrc ? (imageDisplayRef.current?.clientHeight || 0) : (videoRef.current?.videoHeight || 0));
                            const width = ((xmax - xmin) / 1000) * (uploadedImageForAuditSrc ? (imageDisplayRef.current?.clientWidth || 0) : (videoRef.current?.videoWidth || 0));
                            const height = ((ymax - ymin) / 1000) * (uploadedImageForAuditSrc ? (imageDisplayRef.current?.clientHeight || 0) : (videoRef.current?.videoHeight || 0));
                            
                            const borderColor = "#ff00ff"; // Consistent anomaly color

                            return (
                                <div
                                    key={index}
                                    style={{
                                        position: 'absolute',
                                        top: y,
                                        left: x,
                                        width: width,
                                        height: height,
                                        border: `2px solid ${borderColor}`,
                                        boxShadow: `0 0 5px ${borderColor}`,
                                        pointerEvents: 'auto', // Make div clickable/hoverable
                                        cursor: 'help',
                                        zIndex: 50,
                                        opacity: 0.8,
                                    }}
                                    onMouseEnter={() => setHoveredAnomaly(anomaly)}
                                    onMouseLeave={() => setHoveredAnomaly(null)}
                                >
                                    {hoveredAnomaly === anomaly && (
                                        <div style={{
                                            position: 'absolute',
                                            top: '-25px', // Position above the box
                                            left: '0px',
                                            backgroundColor: borderColor,
                                            color: '#000',
                                            padding: '2px 5px',
                                            fontSize: '0.8em',
                                            whiteSpace: 'nowrap',
                                            pointerEvents: 'none', // Don't block mouse events
                                            zIndex: 51,
                                        }}>
                                            {anomaly.label.toUpperCase()}
                                        </div>
                                    )}
                                </div>
                            );
                        })
                    )}
                </div>

                {/* Side Panel for Audit Notes */}
                <div style={{ width: '400px', backgroundColor: '#1a1a1a', padding: '15px', border: '1px solid #00f3ff', overflowY: 'auto' }}>
                    <h3 style={{ color: '#00f3ff', marginBottom: '10px' }}>AUDIT NOTES</h3>
                    {forensicResults?.audit_notes ? (
                        <p style={{ color: '#fff', fontSize: '0.9em', lineHeight: '1.4' }}>{forensicResults.audit_notes}</p>
                    ) : (
                        <p style={{ color: '#888', fontSize: '0.9em' }}>No detailed audit notes available.</p>
                    )}
                     {forensicResults?.anomalies && forensicResults.anomalies.length > 0 && (
                        <div style={{ marginTop: '20px' }}>
                            <h4 style={{ color: '#00f3ff', marginBottom: '5px' }}>ANOMALY DETAILS</h4>
                            {forensicResults.anomalies.map((anom, index) => (
                                <div key={index} style={{ marginBottom: '10px', paddingBottom: '10px', borderBottom: '1px dashed #333' }}>
                                    <p style={{ color: '#ff00ff', fontWeight: 'bold' }}>{anom.label.toUpperCase()}</p>
                                    <p style={{ color: '#ccc', fontSize: '0.85em' }}>Confidence: {(anom.confidence * 100).toFixed(0)}%</p>
                                    <p style={{ color: '#eee', fontSize: '0.85em' }}>{anom.reasoning}</p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Controls (Existing) */}
            <div className="niwa-controls">
                <button
                    onClick={() => startStream('camera')}
                    disabled={isStreaming || isLoading}
                    className="niwa-btn"
                >
                    Init System (Camera)
                </button>
                <button
                    onClick={() => startStream('screen')}
                    disabled={isStreaming || isLoading}
                    className="niwa-btn"
                >
                    Init System (Screen)
                </button>
                <button
                    onClick={stopStream}
                    disabled={!isStreaming && !uploadedImageForAuditSrc && !isLoading}
                    className="niwa-btn niwa-btn-destructive"
                >
                    Terminate
                </button>
                <button
                    onClick={runDeepAudit}
                    disabled={!isStreaming || isLoading}
                    className="niwa-btn"
                >
                    Deep Audit (Live)
                </button>
                <div className="niwa-upload-section">
                    <input
                        type="file"
                        ref={fileInputRef}
                        accept="image/*"
                        className="niwa-file-input"
                    />
                    <button
                        onClick={uploadAndAudit}
                        disabled={isStreaming || isLoading}
                        className="niwa-btn"
                    >
                        Upload Image for Audit
                    </button>
                </div>
            </div>

            {/* Log Panel (Existing) */}
            <div ref={logRef} className="niwa-log">
                {log.map((entry, index) => (
                    <div key={index} className={`niwa-log-entry ${entry.type ? `niwa-log-${entry.type}` : ''}`}>
                        <span className="niwa-log-time">[{entry.time}]</span> {entry.msg}
                    </div>
                ))}
            </div>
            {/* Basic CSS for spinner - typically would be in a CSS file */}
            <style jsx>{`
                .spinner {
                    border: 4px solid rgba(0, 243, 255, 0.1);
                    border-left-color: #00f3ff;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    to {
                        transform: rotate(360deg);
                    }
                }
            `}</style>
        </div>
    );
}