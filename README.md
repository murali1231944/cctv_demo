# Multi-Camera Person Re-Identification (Re-ID) System

A multi-camera person re-identification system built from scratch in Python with a Streamlit frontend. The system ingests and tracks people across multiple camera feeds, indexes their appearance embeddings into Qdrant, maintains cross-camera identity mappings in `resources.json`, and allows querying identities by screenshot with green bounding-box visual video evidence.

---

## 1. System Architecture

```
[Video Files] (Camera_1.mp4, Camera_2.mp4)
      │
      ▼
[YOLOv8 Detection] (Human class 0, Confidence & Area filtering)
      │
      ▼
[ByteTrack Multi-Object Tracker] (Camera-specific track IDs: Camera_1_X, Camera_2_Y)
      │
      ▼
[Frame-Level Metadata & Timeline Timestamps] (i / fps, [x1, y1, x2, y2])
      │
      ▼
[OSNet Feature Extractor] (256x128 input, 512-dim L2-normalized appearance embeddings)
      │
      ▼
[Qdrant Vector Store] (Cosine similarity, 512-dim vectors, deterministic track UUIDs)
      │
      ├────────────────────────────────────────┬────────────────────────────────────────┐
      ▼                                        ▼                                        ▼
[Identity Registry]                    [Query by Screenshot]                    [Evidence Generator]
(resources.json mapping)               (Candidate retrieval)                   (Playable MP4 with green
• Camera 1 manual assignment           • Cosine similarity search               bounding boxes & timestamps)
• Camera 2 automated match/new rule    • Ambiguity margin check
• Idempotent duplicate prevention      • Unknown person rejection
```

---

## 2. Project Directory Structure

```
├── app.py                            # Streamlit frontend (Query by Image & Register Identities)
├── requirements.txt                  # Pinned dependencies
├── README.md                         # Project documentation and runbook
├── .gitignore                        # Git ignore configuration
│
├── configs/
│   ├── __init__.py
│   └── config.py                     # Central configuration, paths, thresholds, and dimensions
│
├── models/
│   ├── yolov8n.pt                    # Pretrained YOLOv8 detection & tracking weights
│   └── osnet_x1_0_market1501.pth     # Pretrained OSNet 512-dim feature extraction weights
│
├── data/
│   ├── input_videos/
│   │   ├── Camera_1.mp4              # Source video 1 (1280x720, 30.01 fps, 3532 frames)
│   │   └── Camera_2.mp4              # Source video 2 (1280x720, 30.01 fps, 3684 frames)
│   ├── crops/                        # Stored representative person crops (e.g. Camera_1_2.jpg)
│   └── evidence/                     # Generated green bounding-box MP4 evidence clips and snapshots
│
├── storage/
│   ├── resources.json                # Persistent cross-camera identity registry
│   ├── qdrant/                       # Embedded local Qdrant vector database
│   └── metadata/
│       ├── Camera_1_tracks.json      # Frame-level bounding boxes and timestamps for Camera 1 (72 tracks)
│       └── Camera_2_tracks.json      # Frame-level bounding boxes and timestamps for Camera 2 (71 tracks)
│
├── src/
│   ├── __init__.py
│   ├── detector.py                   # YOLO person detector
│   ├── tracker.py                    # ByteTrack tracker assigning camera-specific IDs
│   ├── embeddings.py                 # OSNet feature extraction (512-dim, L2-normalized)
│   ├── video_processor.py            # Complete video ingestion pipeline
│   ├── qdrant_store.py               # Qdrant vector store management
│   ├── identity_registry.py          # resources.json manager
│   ├── matcher.py                    # Query by image matcher and decision logic
│   ├── evidence_generator.py         # Green bounding-box video generator
│   ├── utils.py                      # Decoders, timestamps, atomic file I/O, annotation
│   └── models/
│       ├── __init__.py
│       └── osnet.py                  # PyTorch implementation of the OSNet Re-ID architecture
│
├── scripts/
│   └── ingest_all.py                 # Pipeline script to ingest both complete videos into Qdrant
│
└── tests/
    ├── test_milestone1.py            # Milestone 1 suite (Directories, models, basic inference)
    ├── test_milestone2.py            # Milestone 2 suite (Single video ingestion mechanics)
    ├── test_milestone3.py            # Milestone 3 suite (Qdrant storage & idempotency)
    ├── test_milestone4.py            # Milestone 4 suite (Full video ingestion verification)
    ├── validate_thresholds.py        # Empirical threshold & ambiguity margin validation
    ├── test_evidence_and_query.py    # Query by image and video evidence tests
    └── test_e2e_full.py              # Full 16-point specification verification suite
```

---

## 3. Installation & Setup (Windows PowerShell)

### Prerequisites
- Python 3.10 to 3.13 installed on Windows.
- OpenCV dependencies (included with `opencv-python`).

### Setup Commands
Open PowerShell in the project directory:

```powershell
# 1. Create Python virtual environment
python -m venv --system-site-packages .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
pip install -r requirements.txt
```

---

## 4. Model Weights & Data Preparation

The required model weights are located in the `models/` directory:
- **`models/yolov8n.pt`**: YOLOv8 nano model (downloaded automatically by Ultralytics or available from GitHub releases).
- **`models/osnet_x1_0_market1501.pth`**: OSNet x1.0 trained on Market-1501 (downloaded from Hugging Face).

Input videos are stored in `data/input_videos/`:
- `data/input_videos/Camera_1.mp4`
- `data/input_videos/Camera_2.mp4`

---

## 5. Ingestion Pipeline Execution

Both complete videos must be ingested into Qdrant prior to identity registration and querying.

To run the complete ingestion pipeline:

```powershell
python scripts/ingest_all.py
```

### Ingestion Results
- **Camera 1**: 3,532 frames processed → **72 valid tracks**.
- **Camera 2**: 3,684 frames processed → **71 valid tracks**.
- **Total Tracks Indexed in Qdrant**: **143 tracks**.
- Every track is stored with its 512-dimensional OSNet embedding, start/end timestamps, camera ID, representative crop, and full frame-level detections.

---

## 6. Running the Streamlit Application

Start the Streamlit application using PowerShell:

```powershell
streamlit run app.py
```

The application opens automatically in your browser at `http://localhost:8501`.

---

## 7. Streamlit User Guide

The application provides two workflows in the sidebar:

### Option A: Query by Image
1. Upload a screenshot in JPG or PNG format (or use the sample screenshot checkbox).
2. If the screenshot contains multiple people, the system automatically detects and crops candidate people and allows you to select the target person.
3. Adjust the **Cosine Similarity Threshold** (default: `0.65`).
4. Click **Search Person in Qdrant**.
5. The system displays:
   - **Matching Decision**: `CONFIRMED MATCH`, `AMBIGUOUS CANDIDATE`, or `UNKNOWN`.
   - **Metrics**: Matched Identity Label, Camera ID, Track ID, and Appearance Timestamps (`Start -> End`).
   - **Playable Video Evidence**: Video clip showing the matched person highlighted with a **vibrant green bounding box** and timestamp badge on every observed frame.
   - **Keyframe Snapshots**: Visual snapshots showing the person's appearances across time.
   - **Candidate Comparison Table**: Top 5 hits across both cameras with similarity scores.

### Option B: Upload Video / Register Identities
1. **Pipeline Status**: Displays total tracks stored in Qdrant (72 for Camera 1, 71 for Camera 2).
2. **Representative Gallery**:
   - Select **Camera 1** or **Camera 2**.
   - Browse the representative images, track IDs, and timeline intervals.
3. **Camera 1 Registration**:
   - Inspect a clearly visible person crop.
   - Assign an identity label (e.g. `Person_A`, `Person_B`).
   - Click **Register Track** to save atomically to `resources.json`.
4. **Camera 2 Registration**:
   - Click **Match / Register Track**.
   - If the track matches an existing identity with similarity >= 0.65 and margin >= 0.05, it appends to that identity.
   - If no confident match exists, it automatically creates a new unique identity (e.g. `Person_B`).
5. **Persistent Registry Viewer**:
   - View all registered identities and their cross-camera references.
   - Clear or reset the registry anytime without deleting ingested videos or Qdrant data.

---

## 8. Empirical Threshold Validation & Matching Accuracy

A threshold validation study was conducted using actual video frames:

| Metric | Positive Pairs (Same Person) | Negative Pairs (Different People in Same Frame) |
|---|---|---|
| **Pair Count** | 10 | 336 |
| **Mean Cosine Similarity** | `0.7355` | `0.5719` |
| **Std Dev** | `0.1292` | `0.0906` |
| **Min / Max** | `0.5027` / `0.9338` | `0.3671` / `0.9339` |

### Threshold Evaluation Curve
- **0.50**: False Accept Rate = 82.1% (Too loose)
- **0.55**: False Accept Rate = 55.7% (High false positives)
- **0.60**: False Accept Rate = 31.8% (Moderate false positives)
- **0.65**: True Accept Rate = 80.0%, False Accept Rate = 16.1% (**Recommended Operating Point**)
- **0.70**: True Accept Rate = 50.0%, False Accept Rate = 7.7% (Overly strict across viewpoints)

### Ambiguity Margin Rule
When the top candidate score exceeds 0.65, but the difference between top-1 and top-2 is `< 0.05`, the system classifies the result as **AMBIGUOUS** rather than confirming an unverified identity.

---

## 9. Running Automated Tests

Run the test suites from PowerShell:

```powershell
# Milestone 1: Environment, models, and embeddings
python tests/test_milestone1.py

# Milestone 2: Single-video ingestion mechanics
python tests/test_milestone2.py

# Milestone 3: Qdrant vector storage and idempotency
python tests/test_milestone3.py

# Milestone 4: Full video ingestion verification (143 tracks)
python tests/test_milestone4.py

# Empirical threshold validation
python tests/validate_thresholds.py

# Query by image and video evidence generation
python tests/test_evidence_and_query.py

# Complete 16-point specification integration test suite
python tests/test_e2e_full.py
```

---

## 10. Backup and Reset Instructions

### Backup
- Back up `storage/resources.json` to preserve cross-camera identity mappings.
- Back up `storage/qdrant/` to preserve vector embeddings.
- Back up `storage/metadata/` to preserve frame-level bounding boxes and timestamps.

### Reset Demo Data Safely
- To reset only registered identities: Click **Reset resources.json** in the Streamlit UI or delete `storage/resources.json`.
- To completely reset ingestion data without touching source videos or code:
  ```powershell
  Remove-Item -Recurse -Force storage\qdrant\*, storage\metadata\*, data\crops\*, data\evidence\*
  ```
  Then re-run `python scripts/ingest_all.py`.
