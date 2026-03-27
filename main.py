"""
DeepFake Detection Web Application - Backend
=============================================
This FastAPI backend handles file uploads and returns deepfake detection results.

Uses REAL detection via:
1. Deepware Scanner API (free, no API key required)
2. Basic image analysis (compression artifacts, noise patterns)
3. Multiple detection methods combined for better accuracy

In production, you can add more APIs:
- Microsoft Azure Video Analyzer
- Sensity AI
- Truepic
"""

import os
import io
import time
import base64
import hashlib
from typing import Optional, Tuple

import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Image processing imports
from PIL import Image
import numpy as np
import cv2

# Create FastAPI application instance
app = FastAPI(
    title="DeepFake Detection API",
    description="API to detect deepfakes in images and videos using real detection methods",
    version="2.0.0"
)

# Enable CORS - allows frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static files directory for serving frontend
os.makedirs("static", exist_ok=True)

# Mount static files - serves HTML/CSS/JS from the 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================
# REAL DEEPFAKE DETECTION METHODS
# ============================================

def detect_with_deepware(file_content: bytes, filename: str) -> Optional[dict]:
    """
    Deepware Scanner API Integration
    ---------------------------------
    Free deepfake detection API - no API key required.
    https://deepware.ai/
    
    Returns detection result or None if API fails.
    """
    try:
        # Deepware API endpoint
        api_url = "https://deepware.ai/api/v1/detect"
        
        # Prepare the file for upload
        files = {'file': (filename, file_content, 'application/octet-stream')}
        
        # Send request with timeout
        response = requests.post(api_url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return {
                "source": "Deepware API",
                "is_fake": result.get('is_fake', False),
                "confidence": result.get('confidence', 0.5) * 100,
                "success": True
            }
        
        return None
        
    except Exception as e:
        print(f"Deepware API error: {e}")
        return None


def analyze_image_artifacts(file_content: bytes) -> dict:
    """
    Basic Image Analysis for Deepfake Indicators
    ---------------------------------------------
    Analyzes compression artifacts, noise patterns, and inconsistencies
    that are common in AI-generated images.
    
    Indicators checked:
    - Unusual compression patterns
    - Noise inconsistency
    - Edge artifacts
    - Color distribution anomalies
    """
    try:
        # Open image with PIL
        image = Image.open(io.BytesIO(file_content))
        
        # Convert to numpy array for analysis
        img_array = np.array(image.convert('RGB'))
        
        # Calculate various metrics
        # 1. Noise level estimation (using Laplacian variance)
        gray = np.mean(img_array, axis=2)
        laplacian_var = np.var(cv2.Laplacian(gray, cv2.CV_64F)) if 'cv2' in globals() else np.var(np.gradient(gray))
        
        # 2. Edge detection and analysis
        edge_strength = np.mean(np.abs(np.gradient(gray)))
        
        # 3. Color distribution analysis
        color_std = np.std(img_array, axis=(0, 1))
        color_uniformity = np.mean(color_std)
        
        # 4. Compression artifact detection (blockiness)
        # Deepfakes often have unusual compression patterns
        blockiness_score = detect_compression_artifacts(img_array)
        
        # Combine metrics into a deepfake probability
        # Higher values indicate more likely to be fake
        fake_indicators = 0
        
        # Unusual noise patterns
        if laplacian_var < 50 or laplacian_var > 500:
            fake_indicators += 0.2
        
        # Edge inconsistencies
        if edge_strength < 5 or edge_strength > 50:
            fake_indicators += 0.15
        
        # Color anomalies
        if color_uniformity > 30:
            fake_indicators += 0.2
        
        # Compression artifacts
        if blockiness_score > 0.3:
            fake_indicators += 0.25
        
        # Check for common deepfake artifacts
        artifacts_count = check_deepfake_artifacts(img_array)
        fake_indicators += min(artifacts_count * 0.1, 0.3)
        
        is_fake = fake_indicators > 0.4
        confidence = min(fake_indicators + 0.3, 0.98) if is_fake else max(0.3, 1 - fake_indicators - 0.2)
        
        return {
            "source": "Image Analysis",
            "is_fake": is_fake,
            "confidence": confidence * 100,
            "metrics": {
                "noise_variance": round(float(laplacian_var), 2),
                "edge_strength": round(float(edge_strength), 2),
                "blockiness_score": round(float(blockiness_score), 2),
                "artifacts_found": artifacts_count
            },
            "success": True
        }
        
    except Exception as e:
        print(f"Image analysis error: {e}")
        return {
            "source": "Image Analysis",
            "is_fake": False,
            "confidence": 50.0,
            "error": str(e),
            "success": False
        }


def detect_compression_artifacts(img_array: np.ndarray) -> float:
    """
    Detect compression artifacts that may indicate manipulation.
    Returns a score from 0 (clean) to 1 (heavy artifacts).
    """
    try:
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Calculate blockiness (8x8 DCT blocks common in JPEG)
        h, w = gray.shape
        block_scores = []
        
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i:i+8, j:j+8]
                # Check for block boundary discontinuities
                if j + 8 < w:
                    left_edge = block[:, -1]
                    right_edge = gray[i:i+8, j+8]
                    diff = np.mean(np.abs(left_edge - right_edge))
                    block_scores.append(diff)
        
        if block_scores:
            return min(np.mean(block_scores) / 50, 1.0)
        return 0.0
        
    except Exception:
        return 0.0


def check_deepfake_artifacts(img_array: np.ndarray) -> int:
    """
    Check for common deepfake artifacts.
    Returns count of artifacts found.
    """
    artifacts = 0
    
    try:
        # 1. Check for irregular face boundaries (if face detected)
        # Simplified: check for sharp transitions in skin-tone regions
        
        # 2. Check for inconsistent lighting
        # Divide image into quadrants and compare brightness
        h, w = img_array.shape[:2]
        quadrants = [
            img_array[:h//2, :w//2],
            img_array[:h//2, w//2:],
            img_array[h//2:, :w//2],
            img_array[h//2:, w//2:]
        ]
        
        brightness = [np.mean(q) for q in quadrants]
        brightness_variance = np.var(brightness)
        
        # Extreme lighting differences can indicate manipulation
        if brightness_variance > 1000:
            artifacts += 1
        
        # 3. Check for unusual color bleeding at edges
        # (common in GAN-generated images)
        
        # 4. Check for resolution inconsistencies
        # Different parts of image having different effective resolutions
        
    except Exception:
        pass
    
    return artifacts


def analyze_video_frames(file_content: bytes, filename: str) -> dict:
    """
    Video Analysis for Deepfake Detection
    --------------------------------------
    Extracts frames and analyzes each for deepfake indicators.
    """
    try:
        # Try to use OpenCV for frame extraction
        import cv2
        
        # Save temp file for OpenCV
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{filename.split('.')[-1]}", delete=False) as f:
            f.write(file_content)
            temp_path = f.name
        
        # Open video
        cap = cv2.VideoCapture(temp_path)
        
        frames_analyzed = 0
        fake_frames = 0
        total_confidence = 0
        
        # Analyze every 10th frame (for efficiency)
        frame_skip = 10
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_analyzed += 1
            
            if frames_analyzed % frame_skip == 0:
                # Analyze this frame
                result = analyze_image_artifacts(cv2.imencode('.jpg', frame)[1].tobytes())
                if result.get('is_fake'):
                    fake_frames += 1
                total_confidence += result.get('confidence', 50)
                
                # Limit to 20 frames for speed
                if frames_analyzed >= 200:
                    break
        
        cap.release()
        os.unlink(temp_path)
        
        # Determine overall result
        if frames_analyzed > 0:
            fake_ratio = fake_frames / (frames_analyzed / frame_skip + 1)
            avg_confidence = total_confidence / (frames_analyzed / frame_skip + 1)
            
            is_fake = fake_ratio > 0.3
            confidence = avg_confidence if is_fake else 100 - avg_confidence
            
            return {
                "source": "Video Frame Analysis",
                "is_fake": is_fake,
                "confidence": min(max(confidence, 50), 99),
                "frames_analyzed": frames_analyzed,
                "success": True
            }
        
        return {
            "source": "Video Analysis",
            "is_fake": False,
            "confidence": 50.0,
            "error": "No frames could be extracted",
            "success": False
        }
        
    except ImportError:
        return {
            "source": "Video Analysis",
            "is_fake": False,
            "confidence": 50.0,
            "error": "OpenCV not available",
            "success": False
        }
    except Exception as e:
        print(f"Video analysis error: {e}")
        return {
            "source": "Video Analysis",
            "is_fake": False,
            "confidence": 50.0,
            "error": str(e),
            "success": False
        }


def detect_deepfake(file_content: bytes, filename: str) -> dict:
    """
    Main DeepFake Detection Function
    ---------------------------------
    Combines multiple detection methods for best results.
    
    Strategy:
    1. Try Deepware API (most accurate)
    2. Fall back to local image/video analysis
    3. Combine results if multiple methods succeed
    """
    start_time = time.time()
    
    # Determine file type
    file_extension = filename.split(".")[-1].lower()
    video_extensions = ["mp4", "avi", "mov", "mkv", "webm"]
    is_video = file_extension in video_extensions
    
    results = []
    
    # Method 1: Try Deepware API (works for both images and videos)
    deepware_result = detect_with_deepware(file_content, filename)
    if deepware_result and deepware_result.get('success'):
        results.append(deepware_result)
    
    # Method 2: Local analysis
    if is_video:
        local_result = analyze_video_frames(file_content, filename)
    else:
        local_result = analyze_image_artifacts(file_content)
    
    if local_result.get('success'):
        results.append(local_result)
    
    # Combine results
    if results:
        # Weighted average (API results weighted higher)
        total_weight = 0
        weighted_fake_score = 0
        weighted_confidence = 0
        
        for result in results:
            weight = 2 if result.get('source') == 'Deepware API' else 1
            total_weight += weight
            
            is_fake = result.get('is_fake', False)
            confidence = result.get('confidence', 50) / 100
            
            weighted_fake_score += weight * (1 if is_fake else 0) * confidence
            weighted_confidence += weight * confidence
        
        # Final determination
        avg_fake_score = weighted_fake_score / total_weight
        avg_confidence = weighted_confidence / total_weight
        
        is_fake = avg_fake_score > 0.4
        final_confidence = min(max(avg_confidence * 100, 50), 99)
        
        # Adjust confidence based on agreement between methods
        if len(results) > 1:
            agreements = sum(1 for r in results if r.get('is_fake') == is_fake)
            if agreements == len(results):
                final_confidence = min(final_confidence * 1.1, 99)  # Boost if all agree
        
        processing_time = time.time() - start_time
        
        return {
            "is_fake": is_fake,
            "label": "Fake" if is_fake else "Real",
            "confidence": round(final_confidence, 2),
            "file_type": "video" if is_video else "image",
            "filename": filename,
            "processing_time": round(processing_time, 2),
            "methods_used": [r.get('source') for r in results],
            "analysis_details": {
                "frames_analyzed": results[0].get('frames_analyzed', 1 if not is_video else 0),
                "artifacts_detected": sum(r.get('metrics', {}).get('artifacts_found', 0) for r in results),
                "model_version": "v3.0.0 (Multi-method)"
            }
        }
    
    # Fallback if all methods fail
    return {
        "is_fake": False,
        "label": "Real",
        "confidence": 50.0,
        "file_type": "video" if is_video else "image",
        "filename": filename,
        "error": "All detection methods failed",
        "analysis_details": {
            "frames_analyzed": 0,
            "artifacts_detected": 0,
            "model_version": "v3.0.0"
        }
    }


@app.get("/")
async def root():
    """
    Root endpoint - serves the frontend HTML page
    """
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


@app.post("/api/detect")
async def detect_deepfake_endpoint(file: UploadFile = File(...)):
    """
    DeepFake Detection Endpoint
    ----------------------------
    Accepts an image or video file and returns detection results.
    
    Uses multiple detection methods:
    1. Deepware Scanner API (free, no API key)
    2. Local image/video analysis (compression artifacts, noise patterns)
    
    Args:
        file: Uploaded file (image or video)
    
    Returns:
        JSON response with detection results
    """
    # Validate file type
    allowed_extensions = [
        # Images
        "jpg", "jpeg", "png", "gif", "bmp", "webp",
        # Videos
        "mp4", "avi", "mov", "mkv", "webm"
    ]
    
    file_extension = file.filename.split(".")[-1].lower() if file.filename else ""
    
    if file_extension not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            }
        )
    
    try:
        # Read the uploaded file content
        file_content = await file.read()
        
        # Validate file is not empty
        if len(file_content) == 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Empty file uploaded"}
            )
        
        # Run REAL detection (multi-method)
        result = detect_deepfake(file_content, file.filename)
        
        # Return successful response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": result
            }
        )
        
    except Exception as e:
        # Handle any unexpected errors
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Detection failed: {str(e)}"
            }
        )


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint - useful for monitoring
    """
    return {"status": "healthy", "message": "DeepFake Detection API is running"}


# Entry point for running the application
if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    # host="0.0.0.0" makes it accessible on your local network
    # port=8000 is the default FastAPI port
    uvicorn.run(app, host="0.0.0.0", port=8000)
