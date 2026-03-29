# IndoHurb

A plant species recognition and insights platform combining fast API backend, deep learning model inference, and a modern frontend.

<!-- ## 🚀 Quick Start

1. Backend
   - `cd API`
   - `pip install -r requirements.txt`
   - `uvicorn app:app --reload`

2. Frontend
   - `cd frontend`
   - `npm install`
   - `npm run dev`

3. Data processing & model training are in `Data_operations` and `src/indoherb`. -->

## 🖼️ Results

<table>
  <tr>
    <td><img width="1920" height="2127" alt="screencapture-localhost-5173-2026-03-29-20_52_57" src="https://github.com/user-attachments/assets/137fe438-9074-4a22-bd92-049808de0091" /></td>
    <td><img width="1920" height="2127" alt="screencapture-localhost-5173-2026-03-29-20_53_33" src="https://github.com/user-attachments/assets/eac4fe22-03e4-4ad8-94c6-7fe8a5dbbffd" /></td>
  </tr>
</table>


## 📁 Project Structure

- `API/` - FastAPI routes, model inference, Grad-CAM utilities.
- `Data_operations/` - loader, split, and transformation scripts.
- `frontend/` - web UI (Vite/React).
- `Results/` - outputs, demo screenshots.


## 🧩 Usage

- Upload leaf images in UI to predict plant type.
- See Grad-CAM heatmap overlays in `Results/` for explainability.

<!-- ## 📌 Notes

- Adjust model paths in `API/model.py` as needed (e.g., `models/resnet34_best.pth`).
- `Results/Web1.png` and `Results/Web2.png` are included and shown above. -->
