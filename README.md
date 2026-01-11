# ML AI-Powered Precision Farming Advisor

A FastAPI-based machine learning service for tomato disease detection and crop health analysis. This application leverages deep learning to identify tomato plant diseases from image uploads and provides confidence scores for disease classification.

## Features

- **Disease Detection**: Identifies tomato plant health status including:

  - Healthy leaves
  - Leaf blight
  - Rust
  - Pest damage

- **Multiple Image Format Support**: Accepts common image formats including HEIC/HEIF (iPhone photos)
- **Confidence Scoring**: Provides confidence levels for predictions
- **Top-3 Predictions**: Returns the top 3 most likely classifications
- **CORS Enabled**: Ready for frontend integration from any origin

## Project Structure

```
.
├── app.py                          # FastAPI application with prediction endpoint
├── requirements.txt                # Python dependencies
├── model/
│   └── tomato_disease_model.keras  # Pre-trained deep learning model
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- FastAPI
- TensorFlow/Keras
- Pillow (PIL)
- NumPy
- pillow-heif (for HEIC/HEIF support)
- uvicorn (ASGI server)

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ml-ai-powered-precision-farming-advisor
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the server**

   ```bash
   uvicorn app:app --reload
   ```

   The API will be available at `http://localhost:8000`

2. **Access the API documentation**

   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

3. **Make predictions**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -F "file=@path/to/image.jpg"
   ```

## API Endpoints

### POST `/predict`

Upload an image and get disease prediction.

**Request:**

- Content-Type: `multipart/form-data`
- Parameter: `file` (image file)

**Response:**

```json
{
  "label": "healthy",
  "confidence": 0.95,
  "top3": [
    { "label": "healthy", "confidence": 0.95 },
    { "label": "leaf_blight", "confidence": 0.04 },
    { "label": "rust", "confidence": 0.01 }
  ]
}
```

**Error Responses:**

- `400 Bad Request`: Empty or invalid image file
- `500 Internal Server Error`: Server processing error

## Model Details

- **Input Size**: 224x224 pixels (RGB)
- **Confidence Threshold**: 0.5 (returns "uncertain" if below threshold)
- **Output Classes**: 4 disease types
- **Format**: Keras (.keras)

## Configuration

You can modify the following parameters in `app.py`:

- `MODEL_PATH`: Path to the Keras model file
- `CLASS_NAMES`: List of disease labels
- `CONFIDENCE_THRESHOLD`: Minimum confidence for a valid prediction (default: 0.5)

## Development

The application uses:

- **FastAPI**: Modern Python web framework for building APIs
- **TensorFlow/Keras**: Deep learning framework for the ML model
- **Pillow**: Image processing library
- **Uvicorn**: ASGI server

## License

[Add your license here]

## Author

[Prabodh]

## Contributing

Contributions are welcome! Please ensure all dependencies are installed and the server runs without errors.
