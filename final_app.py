import streamlit as st
import pypdf
import cv2
import numpy as np
import io
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from PIL import Image

load_dotenv()

# 1. Models and Schemas 

@st.cache_resource
def load_face_models():
    detector = MTCNN()
    embedder = FaceNet()
    return detector, embedder

detector, embedder = load_face_models()

class AnalysisResult(BaseModel):
    name: str = Field("Give the name of the candidate whose resume you are parsing.")
    
    experience_summary: str = Field(description="Detailed summary of experience match after comparing the years / months of experience candidate has on their resume and alignment with experience required by the job description. This can be obtained by checking the start and end month and year of each job.")
    
    experience_score: float = Field(description="Score out of 100 (upto 1 decimal place), based on how good the alignment is.")
    
    projects_summary: str = Field(description="Detailed summary of projects match after comparing the projects candidate has on their resume with the skills learnt during that project and alignment with skill set required by the job description.")
    
    projects_score: float = Field(description="Score out of 100 (upto 1 decimal place), based on how good the alignment is.")
    
    education_summary: str = Field(description="Detailed summary of education/skills match after comparing the education qualifications and skills candidate has on their resume and alignment with skill set required by the job description.")
    
    education_score: float = Field(description="Score out of 100 (upto 1 decimal place), based on how good the alignment is.")
    
    missing_skills: list[str] = Field(description="List of missing skills after thorough comparison of the resume content with the job description.")
    
    suggestions: list[str] = Field(description="List of improvement suggestions to become better aligned for the job whose job description is given.")

class InterviewEvaluation(BaseModel):
    individual_scores: list[float] = Field(description="List of 5 scores for each answer, each out of 10.")
    
    feedback: list[str] = Field(description="You are an interviewer. Please analyse the responses carefully and give a detailed feedback of one paragraph for each answer. Make sure the feedback is easy to understand by the candidate.")
    
    total_score: float = Field(description="Add all the scores that you have given for each answer. The maximum total of the scores should be lesser than or equal to 50")
    
    hiring_decision: str = Field(description="If the sum total of the scores is 40 or more, give the status 'Hired' otherwise given the status 'Rejected'.")

# 2. Helper Functions
def compute_embedding(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    
    # CONSTRAINT: Handle missing or partial faces
    if not faces: 
        raise ValueError("No clear face detected. Please ensure your face is fully visible and well-lit.")
    
    # If face is too small or partial (heuristic check on detection confidence or size)
    face = faces[0]
    if face['confidence'] < 0.90:
        raise ValueError("Face detection confidence too low. Please center your face in the frame.")

    x, y, w, h = face['box']
    x, y = max(0, x), max(0, y)
    face_crop = img[y:y+h, x:x+w]
    
    if face_crop.size == 0:
        raise ValueError("Invalid face capture. Please try again.")

    face_crop = cv2.resize(face_crop, (160, 160))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = np.expand_dims(face_crop, axis=0)
    return embedder.embeddings(face_crop)[0]

# 3. Streamlit UI Logic

st.set_page_config(page_title="AI Hiring Portal", layout="wide")

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = "analysis"
if 'final_score' not in st.session_state:
    st.session_state.final_score = 0
if 'raw_analysis' not in st.session_state:
    st.session_state.raw_analysis = None
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'interview_complete' not in st.session_state:
    st.session_state.interview_complete = False

# PAGE 1: RESUME ANALYSIS
if st.session_state.page == "analysis":
    st.title("Phase 1: AI Resume Analyzer")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    with col2:
        job_description = st.text_area("Paste Job Description (JD)", height=300)

    if st.button("Analyze Resume"):
        if uploaded_file and job_description:
            with st.spinner("Analyzing profile..."):
                pdf_reader = pypdf.PdfReader(uploaded_file)
                full_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                
                model = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(AnalysisResult)
                template = "Compare the Resume against the JD: \nRESUME: {resume_content}\nJD: {job_description}"
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | model
                
                st.session_state.raw_analysis = chain.invoke({"resume_content": full_text, "job_description": job_description})
                st.session_state.jd_text = job_description 
                st.session_state.resume_text = full_text
                
                res = st.session_state.raw_analysis
                st.session_state.final_score = (res.experience_score * 0.5) + (res.projects_score * 0.3) + (res.education_score * 0.2)
                st.rerun()
        else:
            st.error("Please upload a resume and provide a job description.")

    if st.session_state.raw_analysis:
        raw_analysis = st.session_state.raw_analysis
        final_score = st.session_state.final_score
        
        st.divider()
        st.header(f"FINAL REPORT for {raw_analysis.name}")
        
        metric_cols = st.columns(4)
        metric_cols[0].metric("Overall Score", f"{final_score:.1f}/100")
        metric_cols[1].metric("Experience", f"{raw_analysis.experience_score}")
        metric_cols[2].metric("Projects", f"{raw_analysis.projects_score}")
        metric_cols[3].metric("Education", f"{raw_analysis.education_score}")

        with st.expander("Experience Details", expanded=True):
            st.write(raw_analysis.experience_summary)

        with st.expander("Projects Details", expanded=True):
            st.write(raw_analysis.projects_summary)

        with st.expander("Education Details", expanded=True):
            st.write(raw_analysis.education_summary)

        st.subheader("Missing Skills")
        st.write(", ".join(raw_analysis.missing_skills))

        st.subheader("Suggestions")
        for suggestion in raw_analysis.suggestions:
            st.write(f"- {suggestion}")

        st.divider()
        if final_score >= 60:
            st.success("Target Score Met! Proceed to Face Verification.")
            if st.button("Start Verification"):
                st.session_state.page = "verification"
                st.rerun()
        else:
            st.warning("Score below 60. Not eligible for the next round.")

# PAGE 2: FACE VERIFICATION
elif st.session_state.page == "verification":
    st.title("Phase 2: Identity Verification")
    if st.button("← Back"):
        st.session_state.page = "analysis"
        st.rerun()

    id_file = st.file_uploader("Upload ID Photo", type=["jpg", "png"])
    live_capture = st.camera_input("Take a Live Photo")

    if id_file and live_capture:
        try:
            id_img = cv2.cvtColor(np.array(Image.open(id_file)), cv2.COLOR_RGB2BGR)
            live_img = cv2.cvtColor(np.array(Image.open(live_capture)), cv2.COLOR_RGB2BGR)

            with st.spinner("Comparing faces..."):
                # compute_embedding now handles "no face" or "partial face" errors
                id_emb = compute_embedding(id_img)
                live_emb = compute_embedding(live_img)
                dist = cosine(id_emb, live_emb)
            
            if dist < 0.5: 
                st.success("Verification Successful")
                if st.button("Proceed to Technical Interview"):
                    st.session_state.page = "interview"
                    st.rerun()
            else:
                st.error("Face mismatch. Verification failed. Please ensure the ID and Live capture are of the same person.")
        except ValueError as ve:
            st.warning(f"Detection Issue: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# PAGE 3: TECHNICAL INTERVIEW
elif st.session_state.page == "interview":
    st.title("Phase 3: Technical Interview")
    
    if not st.session_state.questions:
        with st.spinner("LLM is preparing your technical questions..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            q_prompt = f"Based on JD: {st.session_state.jd_text} and Resume: {st.session_state.resume_text}, generate 5 tough technical questions. Return only questions separated by newlines."
            response = llm.invoke(q_prompt)
            st.session_state.questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()][:5]

    with st.form("interview_form"):
        st.write("Please answer the following questions clearly:")
        user_answers = []
        for i, question in enumerate(st.session_state.questions):
            st.markdown(f"**{question}**")
            ans = st.text_area(f"Response {i+1}", key=f"ans_{i}")
            user_answers.append(ans)
        
        submitted = st.form_submit_button("Submit Final Answers")

    if submitted:
        if any(len(a) < 15 for a in user_answers):
            st.error("Please provide more substantive answers to all questions.")
        else:
            with st.spinner("Evaluating performance..."):
                eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(InterviewEvaluation)
                transcript = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(st.session_state.questions, user_answers)])
                eval_res = eval_llm.invoke(f"Evaluate these interview answers (Threshold 40/50):\n{transcript}")
                st.session_state.interview_result = eval_res
                st.session_state.interview_complete = True

    if st.session_state.interview_complete:
        res = st.session_state.interview_result
        st.divider()
        st.header(f"Final Result: {res.hiring_decision}")
        st.progress(min(float(res.total_score / 50), 1.0))
        st.write(f"**Total Interview Score:** {res.total_score}/50")
        
        if res.total_score >= 40:
            st.success("Congratulations! You have been selected for the position.")
        else:
            st.error("Unfortunately, you did not meet the score threshold for this role.")
        
        with st.expander("Review Detailed Feedback"):
            for i, feedback in enumerate(res.feedback):
                st.write(f"**Q{i+1} ({res.individual_scores[i]}/10):** {feedback}")