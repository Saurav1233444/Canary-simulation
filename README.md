# Canary - Early Warning AI for Critical Systems

Canary is a proof-of-concept AI system that detects early regime shifts in time-series data using **Bayesian Change Point Detection (BCPD)**. It provides probabilistic alerts before system failure, enabling human-in-the-loop decision-support.

## 🚀 Features

* **Real-time Monitoring**: Simulates streaming time-series data (e.g., CPU, ML loss).
* **AI-Powered Detection**: Uses a solid implementation of BCPD to calculate the probability of a structural break (regime shift) at every time step.
* **Early Warning Alerts**: Triggers alerts when the prediction probability crosses a threshold.
* **Human-in-the-Loop Dashboard**: Clean React UI to monitor telemetry and accept/reject automatically generated alerts.

---

## 📁 Project Structure

* `/backend` - FastAPI Python server running the AI logic.
* `/frontend` - React Vite application for the Dashboard.
* `/models` - Core Machine Learning implementation (BCPD algorithm).

---

## 🛠️ Step-by-Step Instructions

### 1. Run the Backend

The backend is built with **FastAPI** and uses **NumPy/SciPy** for calculations.

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
*The API will start on `http://localhost:8000`*

### 2. Run the Frontend

The front end is built with **React** (Vite), **TailwindCSS**, and **Recharts**.

```bash
cd frontend
npm install
npm run dev
```

*The UI will start on `http://localhost:5173`*

---

## 📊 How it Works

1. **Simulation Engine**: Once you click *Start Monitoring* on the dashboard, the frontend continually polls `GET /api/step` to fetch the next simulated data point.
2. **Synthetic Data**: The backend contains a sequence of normal data -> drift -> sudden shift.
3. **BCPD Engine**: For every point, the backend calculates the predictive probability using a Student-T distribution, updates run length distributions, and returns the *Changepoint Probability*.
4. **Alerts**: If the probability crosses our set threshold, the UI triggers a pending alert where a human operator can confirm or dismiss it.

---

Built by an AI Engineer.
