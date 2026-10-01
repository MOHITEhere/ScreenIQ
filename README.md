# ScreenIQ

An AI-powered, multi-agent resume screening assistant built with **Streamlit**, **LangChain**, **LangGraph**, and **Llama 3 (via Groq)**. ScreenIQ lets a recruiter upload a resume and a job description, choose exactly which criteria to evaluate the candidate on, and get a transparent, category-wise match score with a clear recommendation.

---

## Why ScreenIQ

Generic resume-screening tools score every candidate the same way, regardless of what the recruiter actually cares about for a given role. ScreenIQ is built around three ideas instead:

- **Recruiter-controlled scoring.** The recruiter picks which categories matter for this screening pass — Skills, Education, Projects, Experience, Achievements & Certifications, and/or Extra (Hobbies) — and the score is computed only from those.
- **Fair to freshers.** Experience is scored in relevance tiers (unrelated industry / same industry, different role / same industry, same role) instead of penalizing candidates who simply haven't had a job yet — freshers are judged on projects and academic work instead.
- **Aware of multi-stream job postings.** Some job descriptions cover multiple possible roles/streams (e.g. a company that assigns your team after hiring). ScreenIQ detects this and scores the candidate separately against each stream, then tells you which one(s) they're actually a fit for.

---

## Key Features

- Upload a **resume (PDF)** and a **job description** as PDF, TXT, or pasted text.
- **Choose evaluation criteria** via checkboxes — Skills, Education, Projects, Experience, Achievements & Certifications, Extra (Hobbies), or All. Scoring weights automatically rebalance to sum to 100 based on what's selected.
- **Multi-role / multi-stream detection** — if a JD lists several possible roles or streams, the candidate is scored against each one individually, with a final comparison of best/worst fit.
- **Category-aware Red Flag detection** — the Red Flag agent only checks concerns relevant to the categories the recruiter selected (e.g. it won't comment on employment gaps if "Experience" wasn't selected), plus always-on checks for spelling, grammar, and broken contact links.
- **Deterministic scoring** — the LLM is called with `temperature=0` and a fixed seed so the same resume and JD produce consistent, reproducible results.
- **Color-coded verdicts** — the final recommendation is shown in green (strong fit, ≥70%), yellow (borderline / internship-level, 50–69%), or red (not a fit, <50%), with the analysis-stage agent outputs shown in a neutral white card.
- Clean Markdown rendering for the LLM's output — headings, bold/italic text, bullet lists, and score-breakdown tables all render properly instead of showing raw Markdown symbols.

---

## How Scoring Works

Each category has a **base weight** (used when all categories are selected):

| Category | Base Weight |
|---|---|
| Skills | 25 |
| Education | 10 |
| Projects | 20 |
| Experience | 30 |
| Achievements & Certifications | 10 |
| Extra (Hobbies) | 5 |

If the recruiter selects only some categories, their weights are proportionally rescaled so the total is always 100. For example, selecting only Skills and Projects rescales them to roughly 56 and 44.

**Experience** is scored in relevance tiers rather than a flat number:
- Relevant work experience in a different industry → partial credit
- Relevant work experience in the same industry, different role → more credit
- Relevant work experience in the same industry and a similar role → full marks

**Recommendation thresholds** (based on percentage of the selected total):
- **70%+** → Recommended for the job
- **50–69%** → Not recommended for this specific role, but evaluated separately for internship/entry-level fit
- **Below 50%** → Not recommended, with the biggest gaps called out

---

## Agent Workflow

| Agent | Role |
|---|---|
| **Resume Agent** | Extracts the candidate's name and contact details from the resume. |
| **JD Agent** | Extracts a readable summary of the job description's requirements (for display purposes). |
| **Red Flag Agent** | Flags concerns in the resume — scoped to only the categories the recruiter selected, plus general resume-quality issues (spelling, formatting, broken links). |
| **Recruiter Agent** | Reads the full original resume and job description, detects single vs. multi-role JDs, applies the selected rubric, and produces the final score, breakdown table, and recommendation. |

---

## Tech Stack

- **Python**, **Streamlit**
- **LangChain + LangGraph** for agent orchestration
- **Llama 3 family models via Groq API** (`langchain-groq`)
- **PyPDFLoader** for resume/JD PDF parsing

---

## Project Structure

```
screenIQ_agents/
├── app.py            # Streamlit UI, criteria selection, result rendering
├── IQ_agents.py       # Agent definitions, scoring rubric, LangGraph workflow logic
├── .env               # GROQ_API_KEY goes here (not committed)
└── data/               # Sample resume/JD files for testing
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install streamlit langchain langgraph langchain-groq chromadb langchain-huggingface langchain-community
```

Add your key to a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Then run:

```bash
streamlit run app.py
```

---

## Example Use Case

1. Upload a candidate's resume (PDF).
2. Upload or paste the job description (PDF, TXT, or pasted text).
3. Select which criteria to evaluate the candidate on (or leave "All" checked).
4. Click **"Match Resume"**.
5. Get a category-wise score breakdown, a color-coded recommendation, and (for multi-stream JDs) a per-role comparison.

---

## 📬 Author

**Atharva Mohite** – *B.Tech AIML | AI & Data Enthusiast*
- 📧 **Email:** [matharva655@gmail.com](mailto:matharva655@gmail.com)
- 🔗 **LinkedIn:** [linkedin.com/in/reachmohiteatharva](https://linkedin.com/in/reachmohiteatharva)
- 💻 **GitHub:** [github.com/MOHITEhere](https://github.com/MOHITEhere)

> Feel free to reach out for suggestions, feedback, or collaboration!