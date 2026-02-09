# Project Niwa: Agentic Deepfake Forensics

![Project Niwa Banner](https://i.imgur.com/Y7zlBK4.jpeg)

**More than a detector. An autonomous digital investigator. Powered by Google Gemini 3.**

### 📺 [WATCH THE DEMO VIDEO HERE](https://youtu.be/jF2TXpnpVTo)

## What Exactly is Project Niwa?

Project Niwa is a real-time forensic scanner designed to restore trust in digital media. Inspired by the need for "Zero-Trust" media verification, Niwa combines low-latency biometric scanning with high-level semantic reasoning to detect deepfakes that traditional tools miss.

Unlike standard detectors that rely on invisible watermarks (which can be stripped) or black-box classifiers, Niwa acts as an agentic investigator. It uses a tiered architecture to analyze video streams for **temporal artifacts** (jitter, glitching) and **logical inconsistencies** (physics, lighting, metadata).

## What can Niwa do?

Niwa utilizes a "Dual-Agent" architecture powered by the Gemini 3 family.

### 1. The Live Sentinel (Gemini 2.5 Flash)
* **Real-Time Triage:** Monitors video streams via WebSockets with sub-100ms latency.
* **Temporal Artifact Detection:** Identifies micro-jitters, texture washing, and lip-sync failures in raw pixels.
* **Biometric Stability Check:** Flags "gliding" faces or unnatural blinking patterns.

### 2. The Deep Forensic Auditor (Gemini 3 Preview)
* **Agentic Tool Use:** autonomously executes Python forensic scripts (ELA, Noise Analysis, EXIF Extraction) when a threat is escalated.
* **Semantic Reasoning:** identifying logical impossibilities, such as a receipt timestamp that contradicts the sunlight in the background.
* **Explainable Verdicts:** Provides a detailed "Case Report" explaining *why* a frame is fake, rather than just a probability score.

## Unique Innovation: Semantic Logic Overlays
Most AI detectors fail when the image quality is perfect. Niwa succeeds by looking for truth, not just pixels.

### Semantic Anomaly Detection (Industry First):
Unlike traditional detectors that rely solely on heatmaps, Niwa uses Gemini 3's reasoning engine to identify logical and semantic impossibilities.

**Example:** Detecting that a receipt’s timestamp (11:27:00) is identical for the start and end of a service, or that a Zip Code corresponds to Florida while the Area Code is from Colorado.

**The Result:** Niwa overlays precise bounding boxes on these logical fallacies, catching high-quality deepfakes that are visually flawless but factually impossible.

### Key Features
*   **Real-time Biometric Triage (Live Agent):** A sub-100ms "beat cop" that scans live video streams (camera or screen share) using Gemini Flash, instantly flagging micro-expressions and glitch artifacts.
*   **Agentic Forensic Auditing (Deep Audit):** An autonomous "expert detective" that doesn't just guess—it investigates. It iteratively runs tools (ELA, noise analysis, cropping) to build a verified case file.
*   **Intuitive "Terminator" HUD:** A visual-first overlay system that draws bounding boxes around anomalies in real-time, providing immediate visual feedback to the user.
*   **Tool-Augmented Verification:** Automatically extracts EXIF metadata, calculates perceptual hashes, and performs Error Level Analysis (ELA) to provide mathematical proof of manipulation.
*   **Hybrid Architecture:** Seamlessly routes tasks between the ultra-fast Gemini 2.5 Flash (for streaming) and the reasoning-heavy Gemini 3 Preview (for deep auditing).

## Architecture & Logic

The system is built on a **FastAPI** backend and **Next.js** frontend. It solves the "Latency vs. Intelligence" trade-off by routing frames based on threat levels.

![Architecture Diagram](https://i.imgur.com/Y7zlBK4.jpeg)

## Safety and Privacy Acknowledgement

The idea of real-time biometric scanning requires strict ethical boundaries.

**What does Niwa have access to?**
* Niwa analyzes only the active video stream or uploaded file provided by the user.
* It does NOT record or store video feeds. All processing happens in ephemeral memory buffers.
* Once the session is closed, all forensic data is wiped.

**How is the data processed?**
* Frames are sent to Google Gemini via encrypted WebSocket channels.
* No facial recognition databases are used. Niwa looks for *authenticity artifacts*, not *identity*. It does not know who you are, only if you are real.

## How to Run Niwa

For the best experience, we recommend running the backend on a machine with decent internet bandwidth for the WebSocket stream.

### 1. Clone the Repository
```bash
git clone [https://github.com/Mister9ce/project](https://github.com/Mister9ce/project-niwa)
cd project-niwa

```

### 2. Backend Setup (FastAPI)

```bash
cd niwa-api

# Install uv if you haven't already:
# curl -LsSf https://astral.sh/uv/install.sh | sh
# or
# pip install uv

uv venv
source .venv/bin/activate # On Windows, use `.venv\Scripts\activate`
uv pip install -r requirements.txt

```

**Create a `.env` file in the `niwa-api` folder:**

```env
GOOGLE_API_KEY=your_gemini_api_key_here

```

**Run the Server:**

```bash
# On Windows
.venv\Scripts\activate
uvicorn app.main:app --reload

# On Linux/macOS
source .venv/bin/activate
uvicorn app.main:app --reload


```

### 3. Frontend Setup (Next.js)

Open a new terminal:

```bash
cd niwa-frontend
npm install
npm run dev

```

Navigate to `http://localhost:3000` to access the Niwa HUD.

## API Usage

Niwa relies entirely on the **Google Gemini API**.

* **Live Stream:** Uses `gemini-2.5-flash-native-audio-preview` for high-throughput heuristic scanning.
* **Deep Audit:** Uses `gemini-3-flash-preview` (or Pro) for complex reasoning and tool orchestration.

## About

Submitted to the **Google Gemini 3 Hackathon (2026)**.
Project Niwa aims to build a scalable trust layer for the internet age.

**Created by:** [Emmanuel Kissi]
**License:** MIT License
