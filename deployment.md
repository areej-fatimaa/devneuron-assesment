# AWS Deployment Documentation

## Frontend Deployment
- Platform: AWS Lambda
- Framework: Next.js (static export)
- Public URL: https://main.d2bebgs5kqxbg.amplifyapp.com/

### Reason for Choice
AWS Lambda was chosen to host the frontend due to its serverless nature,
zero idle cost, and easy public URL exposure. It allows quick deployment
without managing servers and fits well within AWS Free Tier constraints.

---

## Backend Deployment
- Platform: AWS EC2 (t2.micro)
- Framework: FastAPI
- Public URL: https://fgsm.duckdns.org/docs

### Reason for Choice
EC2 t2.micro was selected for the backend to allow full control over the
FastAPI application, simplify model loading, and avoid Lambda cold start
issues for ML inference. This option is fully covered under the AWS
12-month Free Tier.

---

## End-to-End Validation
The deployed frontend successfully communicates with the backend API to:
- Upload an image
- Specify epsilon
- Run FGSM attack
- Display clean and adversarial predictions with images

Screenshots are attached to demonstrate successful execution.
