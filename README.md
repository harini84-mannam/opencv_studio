# 👁️ OpenCV Studio: Object Detection & Image Processing Toolkit

A Flask-based web application that lets you run a wide range of **OpenCV image and video operations from the browser**, with no code needed. It also includes a built-in **Document Scanner** with OCR, so photographed documents can be turned into clean scanned PDFs and searchable text.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract_OCR-3C873A?style=flat-square)

---

## ✨ Features

### 🎥 Face & Eye Detection (video)
- Upload a video and watch it processed frame by frame with faces and eyes highlighted
- Faces detected with a pretrained **Caffe SSD (res10_300x300_ssd)** deep-learning model, which is more robust than Haar cascades alone, especially for angled faces
- Eyes detected with a **Haar cascade**, searching only the upper part of each detected face for better accuracy and speed
- Results streamed to the browser using MJPEG-style multipart JPEG streaming

### 🖼️ Single-image tools
| Tool | Details |
|---|---|
| Image Editor | 2×2 grid: original, grayscale, Gaussian blur, Canny edges |
| Grayscale | Colour to intensity conversion |
| Blur & Smoothing | Gaussian, median and bilateral filters |
| Edge Detection | Canny (adjustable thresholds) and Sobel |
| Brightness & Contrast | Adjustable alpha (contrast) and beta (brightness) |
| Colour Filtering | HSV masking for red, blue, green and yellow |
| Thresholding | Fixed, Otsu and adaptive |
| Morphological Operations | Erosion, dilation, opening and closing (5×5 kernel) |
| Background Removal | OpenCV GrabCut |
| Corner Detection | Shi-Tomasi corners marked on the image |
| Shapes & Text | Rectangle, circle, line, arrow and custom text overlays |
| Splitting | Left/right halves and four quadrants |

### 🖼️🖼️ Two-image tools
- **Template Matching** using normalized cross-correlation, with a bounding box on the best match (the template is scaled down automatically if it is larger than the source)
- **Feature Matching** using ORB keypoints and descriptors with a brute-force matcher (Hamming distance), drawing the best 50 matches
- **Image Arithmetic**: weighted blend of two images
- **Combine**: join images horizontally or vertically, or recombine four quadrants

### 🎞️ Video tools
- **Frame Extraction**: choose how many frames you want, evenly spaced across the video, and download each one as an image

### 📄 Document Scanner (Flask Blueprint)
Supports images, PDFs, Word documents and text files, in two modes:

- **Scanner Mode:** detect the document's edges and four corners (with a fallback if detection fails), apply perspective correction, enhance the image, classify the document type, and generate a scanned-style **PDF**
- **Extraction Mode:** run OCR and text extraction, choose between a **summarized** or **full** text result, and download it as a `.txt` file or PDF

---

## 🔄 How It Works

1. Open a tool from the home page (each tool has its own card)
2. Upload an image, two images, a video or a document
3. The Flask app validates the file and runs the selected OpenCV or document pipeline
4. The result is displayed or streamed in the browser
5. Where relevant, download the processed image, frames, PDF or text file

---

## 🧰 Tech Stack

| Component | Purpose |
|---|---|
| Flask | Web framework (the Document Scanner is a Blueprint) |
| OpenCV | Image and video processing |
| Caffe SSD model | Face detection |
| Haar cascade | Eye detection |
| Tesseract OCR | Text extraction from documents |
| PyMuPDF | Reading PDF documents |
| ReportLab | Generating scanned PDFs |
| Pillow | Image handling |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and available on your PATH
- The Caffe face-detection model files (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`) placed where the app expects them

### Installation

```bash
# Clone the repository
git clone https://github.com/harini84-mannam/<repo-name>.git
cd <repo-name>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open http://127.0.0.1:5000/ in your browser.

---

## 📸 Screenshots

<!-- Add screenshots here, for example:
![Home Page](screenshots/home.png)
![Face and Eye Detection](screenshots/face-detection.png)
![Document Scanner](screenshots/document-scanner.png)
-->

---

## 👩‍💻 Author

**Harini Mannam**
[Portfolio](https://hariniport-hwshwlfa.manus.space/) · [LinkedIn](https://www.linkedin.com/in/harini-mannam-052aa730b/) · [GitHub](https://github.com/harini84-mannam)
