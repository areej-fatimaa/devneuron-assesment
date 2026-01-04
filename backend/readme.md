# FGSM Backend

This FastAPI service exposes an endpoint to perform the
Fast Gradient Sign Method (FGSM) adversarial attack on
input images.

## Endpoint

POST /attack

Input:
- Image file (PNG/JPEG)
- Epsilon (float)

Output:
- Clean prediction
- Adversarial prediction
- Base64 encoded adversarial image
- Attack success status
