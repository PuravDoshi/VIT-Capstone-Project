# AI Hiring Portal

An end-to-end **AI-powered hiring system** built using Streamlit that automates:

- Resume Screening  
- Face Verification  
- Technical Interview Evaluation  

---

## Features

### 1. Resume Analysis (Phase 1)
- Parses PDF resumes using `pypdf`
- Compares with Job Description using LLM (`gpt-4o-mini`)
- Generates:
  - Experience, Projects, Education scores
  - Missing skills
  - Improvement suggestions
- Computes a final weighted score

---

### 2. Face Verification (Phase 2)
- Uses:
  - `MTCNN` for face detection
  - `FaceNet` for embeddings
- Compares:
  - Uploaded ID photo
  - Live webcam capture
- Uses cosine distance for verification

---

### 3. Technical Interview (Phase 3)
- Dynamically generates 5 technical questions
- Evaluates answers using LLM
- Provides:
  - Individual scores (out of 10)
  - Detailed feedback
  - Final hiring decision

---

## Tech Stack

- **Frontend:** Streamlit  
- **LLM Integration:** LangChain + OpenAI  
- **Computer Vision:** OpenCV, MTCNN, FaceNet  
- **Data Processing:** NumPy, SciPy  
- **Validation:** Pydantic  

---

## Project Structure

```bash
AI-Hiring-Portal/
│
├── Sample Resumes/
│   ├── Average Quality Resume.pdf
│   ├── Low Quality Resume.pdf
│   └── Purav Doshi - CV - Final copy.pdf
│
├── final_app.py
├── job_description.txt
├── README.md
```

---

## Environment Setup

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Run the Application

```bash
streamlit run final_app.py
```

---

## Scoring Logic

### Resume Score:
```text
Final Score =
  (Experience * 0.5) +
  (Projects * 0.3) +
  (Education * 0.2)
```

### Interview Decision:
- **Hired:** Score ≥ 40/50  
- **Rejected:** Score < 40/50  

---

## Constraints & Validations

### Face Verification:
- Face must be:
  - Clearly visible
  - Well-lit
  - Detection confidence ≥ 0.90

### Interview:
- Minimum answer length enforced
- Prevents shallow responses

---

## Key Components

### 1. `AnalysisResult` (Pydantic)
Structured output for resume evaluation

### 2. `InterviewEvaluation`
Structured scoring + feedback for interview

### 3. `compute_embedding()`
- Detects face
- Extracts embedding
- Handles edge cases (no face, low confidence)

---

## Future Improvements

- Voice-based interview
- Dashboard analytics for recruiters
- Deployment (AWS / GCP)
- Fine-tuned LLM for hiring-specific evaluation

---

## Author

**Purav Doshi**

---

## If you like this project

Give it a ⭐ on GitHub and consider building on top of it!
