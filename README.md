# ScreenIQ

An AI-powered, multi-agent resume screening assistant built with **Streamlit**, **LangChain**, **LangGraph**, and **Llama 3 (via Groq)**. A recruiter uploads a resume and a job description, chooses exactly which criteria to evaluate, and gets a transparent, category-wise match score with a clear reason and recommendation.

---

## Problems Solved

| # | Problem in typical resume screening | How ScreenIQ solves it |
|---|---|---|
| 1 | **One-size-fits-all scoring.** Every candidate is scored the same way, even if the company only cares about some parts of a resume. | **Recruiter-selectable criteria.** The recruiter picks any combination of Skills, Education, Projects, Experience, Achievements & Certifications, Extra (Hobbies), or All. Weights rebalance automatically to total 100, so the score reflects only what the company cares about. |
| 2 | **Black-box scores.** A number with no explanation is hard to trust or act on. | **Explainable results.** ScreenIQ gives a category-wise score breakdown and writes the reason behind each score, based on the criteria selected. |
| 3 | **Unclear role fit.** Many job descriptions cover several roles or streams, and one overall score hides which role suits the candidate. | **Multi-role / multi-stream detection.** The candidate is scored separately against each role in the JD, and the final output says which role(s) they fit best and which they fit least. |
| 4 | **Inconsistent LLM output.** The same resume and JD can give different scores on different runs, which makes screening unfair. | **Deterministic scoring.** The LLM runs with `temperature=0` and a fixed seed, so results stay faithful to the JD and resume and are reproducible. |
| 5 | **Freshers are penalized** for having no work experience. | **Tiered experience scoring** (different industry, same industry, same industry and role) and a focus on projects and academics, so freshers are judged fairly. |
| 6 | **Irrelevant red flags.** Generic tools flag issues the recruiter did not ask about. | **Category-aware Red Flag agent.** It checks only the selected categories, plus always-on checks for spelling, grammar, and broken contact links. |

---

## Key Features

- Upload a **resume (PDF)** and a **job description** as PDF, TXT, or pasted text.
- **Choose evaluation criteria** via checkboxes. Scoring weights rebalance to sum to 100.
- **Score with reasoning** for every selected category, shown in a breakdown table.
- **Multi-role / multi-stream detection** with a per-role comparison and a best/worst fit summary.
- **Category-aware Red Flag detection** scoped to the selected criteria.
- **Deterministic scoring** using `temperature=0` and a fixed seed.
- **Color-coded verdicts:** green (strong fit, 70% and above), yellow (borderline or internship-level, 50-69%), red (not a fit, below 50%).
- Clean Markdown rendering of the LLM output (headings, bullets, tables).

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

**Experience** is scored in relevance tiers:
- Relevant work experience in a different industry: partial credit
- Relevant work experience in the same industry, different role: more credit
- Relevant work experience in the same industry and a similar role: full marks

**Recommendation thresholds** (percentage of the selected total):
- **70%+**: Recommended for the job
- **50-69%**: Not recommended for this specific role, but evaluated separately for internship/entry-level fit
- **Below 50%**: Not recommended, with the biggest gaps called out

---

## Agent Workflow

| Agent | Role |
|---|---|
| **Resume Agent** | Extracts the candidate's name and contact details from the resume. |
| **JD Agent** | Extracts a readable summary of the job description's requirements. |
| **Red Flag Agent** | Flags concerns scoped to the selected categories, plus general resume-quality issues (spelling, formatting, broken links). |
| **Recruiter Agent** | Reads the full resume and JD, detects single vs. multi-role JDs, applies the selected rubric, and produces the final score, breakdown table, reasons, and recommendation. |

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
├── IQ_agents.py      # Agent definitions, scoring rubric, LangGraph workflow logic
├── .env              # GROQ_API_KEY goes here (not committed)
└── data/             # Sample resume/JD files for testing
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
5. Get a category-wise score breakdown with reasons, a color-coded recommendation, and (for multi-stream JDs) a per-role comparison.

---

## Author

**Atharva Mohite** - *B.Tech AIML | AI & Data Enthusiast*
- **Email:** [matharva655@gmail.com](mailto:matharva655@gmail.com)
- **LinkedIn:** [linkedin.com/in/reachmohiteatharva](https://linkedin.com/in/reachmohiteatharva)
- **GitHub:** [github.com/MOHITEhere](https://github.com/MOHITEhere)

> Feel free to reach out for suggestions, feedback, or collaboration!
