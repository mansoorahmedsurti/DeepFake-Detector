# DeepFake Detector Web App

A web application for detecting deepfakes in images and videos using **real detection methods**.

## Features

- 🖼️ **Upload Images & Videos** - Drag & drop or click to upload
- 🔬 **Real AI Analysis** - Uses Deepware Scanner API + local image analysis
- 📊 **Confidence Score** - Visual confidence bar with percentage
- ⚡ **Multi-Method Detection** - Combines multiple detection techniques
- 🎨 **Modern UI** - Beautiful gradient design

## Detection Methods

The app uses **real detection** via:

1. **Deepware Scanner API** - Free deepfake detection API (no API key required)
2. **Image Analysis** - Local analysis of:
   - Compression artifacts
   - Noise patterns
   - Edge inconsistencies
   - Color distribution anomalies
   - Lighting inconsistencies

Results from multiple methods are combined for better accuracy.

## Project Structure

```
DeepFake Detector/
├── main.py              # FastAPI backend
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── static/
    └── index.html      # Frontend (HTML/CSS/JS)
```

## Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
python main.py
```

### Step 3: Open in Browser

Navigate to: **http://localhost:8000**

## How It Works

1. **Upload** - User uploads an image or video file
2. **Send to Backend** - Frontend sends file to `/api/detect` endpoint
3. **Analysis** - Backend processes the file (mock detection in this demo)
4. **Results** - Returns Real/Fake label with confidence score
5. **Display** - Frontend shows results with animated confidence bar

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the frontend HTML |
| `/api/detect` | POST | Analyzes uploaded file |
| `/api/health` | GET | Health check endpoint |

### Example API Request

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/api/detect
```

### Example Response

```json
{
  "success": true,
  "result": {
    "is_fake": false,
    "label": "Real",
    "confidence": 94.56,
    "file_type": "image",
    "filename": "image.jpg",
    "analysis_details": {
      "frames_analyzed": 1,
      "artifacts_detected": 0,
      "model_version": "v2.1.0"
    }
  }
}
```

## Customization

### Using a Real DeepFake Detection API

Replace the `mock_detect_deepfake()` function in `main.py` with a real API call:

```python
import requests

def detect_deepfake_real_api(file_content, filename):
    # Example: Call an external deepfake detection API
    api_url = "https://api.deepfake-detection.com/detect"
    api_key = "YOUR_API_KEY"
    
    files = {'file': (filename, file_content)}
    headers = {'Authorization': f'Bearer {api_key}'}
    
    response = requests.post(api_url, files=files, headers=headers)
    result = response.json()
    
    return {
        "is_fake": result['is_deepfake'],
        "label": "Fake" if result['is_deepfake'] else "Real",
        "confidence": result['confidence'] * 100,
        # ... etc
    }
```

### Changing Colors

Edit the CSS in `static/index.html`:
- Primary color: `#e94560` (pink/red)
- Secondary color: `#0f3460` (dark blue)
- Background gradient: Modify the `body` background property

## Supported File Types

**Images:** JPG, JPEG, PNG, GIF, BMP, WEBP

**Videos:** MP4, AVI, MOV, MKV, WEBM

## Troubleshooting

### Port Already in Use

If port 8000 is busy, change it in `main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Use port 8001 instead
```

### CORS Errors

The app allows all origins by default. For production, restrict in `main.py`:
```python
allow_origins=["https://yourdomain.com"]
```

## License

MIT License - Feel free to use and modify!
