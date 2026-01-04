# Software Engineer Intern Assessment – DevNeuron

This project demonstrates adversarial attacks on neural
networks using the Fast Gradient Sign Method (FGSM).

---

## Tech Stack
- FastAPI
- PyTorch
- Next.js
- AWS EC2 & Lambda

---

## How to Run Locally

### Backend
cd backend
pip install -r requirements.txt
uvicorn app_fgsm:app --reload

### Frontend
cd frontend
npm install
npm run dev

---

## Deployed URLs

Frontend:
https://main.d2bebgs5kqxbg.amplifyapp.com/

Backend:
https://fgsm.duckdns.org/docs

---

## Fast Gradient Sign Method (FGSM)

FGSM is an adversarial attack that perturbs an input image
in the direction of the gradient of the loss with respect
to the input. This perturbation is scaled by epsilon,
which controls attack strength.

Even small perturbations can cause neural networks to
make incorrect predictions, exposing robustness issues.

---

## Observations

- Increasing epsilon reduced model accuracy
- Model misclassified adversarial images while
  correctly classifying clean images
- Higher epsilon values made perturbations more visible

---

## Screenshots

Screenshots for backend, frontend, and deployment
are included in the screenshots/ directory.
