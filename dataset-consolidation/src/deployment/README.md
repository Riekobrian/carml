# Car Price Prediction API

This directory contains the deployment code for the car price prediction model.

## Setup

1. Create a new virtual environment:
```bash
python -m venv deploy_env
```

2. Activate the environment:
```bash
# Windows
deploy_env\Scripts\activate

# Linux/Mac
source deploy_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

Start the API server:
```bash
uvicorn predict_api:app --reload --host 0.0.0.0 --port 8000
```
python -m uvicorn predict_api:app --reload
The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

## Endpoints

### 1. Predict Price (`POST /predict`)

Make a prediction for a car's price based on its features.

Example request:
```json
{
    "make_name": "toyota",
    "model_name": "corolla",
    "year_of_manufacture": 2018,
    "mileage": 50000,
    "engine_size_cc": 1800,
    "fuel_type": "petrol",
    "transmission": "automatic",
    "body_type": "sedan",
    "condition": "excellent",
    "usage_type": "foreign used"
}
```

Example response:
```json
{
    "status": "success",
    "predicted_price": 2500000,
    "predicted_price_formatted": "KES 2,500,000.00",
    "prediction_timestamp": "2025-05-21T00:31:37.607000",
    "model_version": "model_20250521_003134"
}
```

### 2. Health Check (`GET /health`)

Check if the API is running and which model version is loaded.

Example response:
```json
{
    "status": "healthy",
    "model_version": "model_20250521_003134",
    "timestamp": "2025-05-21T00:31:37.607000"
}
```

## Error Handling

The API includes proper error handling for:
- Invalid input data
- Missing required fields
- Out-of-range values
- Internal server errors

All errors return appropriate HTTP status codes and error messages.

## Monitoring

The API logs all predictions and errors to help with monitoring and debugging.
Logs include:
- Timestamp
- Request details
- Prediction results
- Error messages (if any)

## Security Notes

1. The API currently runs without authentication
2. For production, consider adding:
   - API key authentication
   - Rate limiting
   - HTTPS
   - Input sanitization
   - Request logging
   - Monitoring alerts 