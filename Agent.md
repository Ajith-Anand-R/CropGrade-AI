# CropGrade AI: AI Agent Working Instructions & Roadmap

As the AI agent (Antigravity) working on CropGrade AI, you must follow the rules, guidelines, and directory structures outlined below to ensure the codebase remains clean, well-documented, and aligned with the project vision.

---

## 🛑 Critical Rules for the AI Agent
1. **Graphify Synchronization**:
   - Every time a change is made in the codebase, you **must** update the `graphify` knowledge graph.
   - When a new task is received, do not read the whole codebase unless absolutely necessary. Instead, read the `graphify-out/` outputs to understand the current structure and file relationships.
2. **No Placeholders**:
   - Always write complete, functional code. If UI designs or assets are needed, use generative tools or real CSS/HTML styling rather than placeholder text/boxes.
3. **Aesthetic Excellence**:
   - All frontend designs must look modern, responsive, and premium. Use curated HSL palettes (no default web colors), modern typography (e.g., Google Fonts Inter/Outfit), smooth gradients, and subtle micro-interactions.

---

## 📁 Codebase Directory Structure
Organize all files according to the following component mapping:

* **`/Dataset`**: Data collection, preprocessing, and augmentation scripts.
  * Contains dataset configurations, Roboflow upload scripts, and local image preprocessing helper scripts.
* **`/Models`**: Model architecture, training, and evaluation scripts.
  * YOLOv8 training configs, Google Colab notebooks (references), hyperparameter tuning logs, and local model inference testers.
* **`/Backend`**: FastAPI application implementation.
  * FastAPI endpoints, YOLOv8 inference wrapper, size estimation ratio calculator, and shelf life rule-based logic.
* **`/Frontend`**: Responsive mobile web interface.
  * HTML/JS or React application designed for mobile screens with native camera access.
* **`/Docs`**: Technical blueprint documents, user guides, and research logs.
  * Project blueprints, user feedback documents, and performance metrics reports.

---

## 📝 Agent Implementation Checklist

### 📋 Phase 1: Setup & Data Pipelines (Dataset & Models)
- [ ] Research and download suitable datasets from Roboflow/Kaggle (mandi tomatoes, defect datasets).
- [ ] Create dataset helper scripts in `Dataset/` to clean and structure the data.
- [ ] Write YOLOv8 training configuration files in `Models/`.
- [ ] Document training steps and Colab setup in `Models/README.md`.

### 📋 Phase 2: Core ML Inference (Models)
- [ ] Write a Python class to load YOLOv8 and run inference on raw images.
- [ ] Implement bounding box relative size calculation (using standard reference object).
- [ ] Code the rule-based shelf-life estimation algorithm.

### 📋 Phase 3: Backend API (Backend)
- [ ] Bootstrap the FastAPI application in `Backend/main.py`.
- [ ] Define API request/response Pydantic models (image upload -> tomato detection details + batch summary JSON).
- [ ] Integrate the YOLOv8 model inference wrapper into the FastAPI endpoint.
- [ ] Add exception handling and loggers.

### 📋 Phase 4: Mobile Web Interface (Frontend)
- [ ] Create a mobile-first UI with HTML5/CSS and Javascript in `Frontend/`.
- [ ] Add camera capture input (`accept="image/*" capture="camera"`).
- [ ] Write fetch calls to POST the captured image to the FastAPI backend.
- [ ] Design visual cards and interactive reports to show count, sizes, grades, defects, and shelf life.

---

## 🚀 How to Run Graphify on the Codebase
Whenever changes are made, run:
```powershell
# In PowerShell:
& (Get-Content graphify-out\.graphify_python) -m graphify
# Or run with the workspace path:
/graphify .
```
Verify that `graphify-out/graph.json` and `graphify-out/GRAPH_REPORT.md` are updated.
