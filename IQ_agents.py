import warnings
warnings.filterwarnings("ignore")

import json
import operator
import os
from typing import Annotated, TypedDict, Sequence
from langgraph.graph import END, START, StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
# temperature=0 + a fixed seed make scoring as deterministic/reproducible
# as Groq's API allows - critical for a recruitment tool, since the same
# resume + JD should always get (close to) the same score. temperature=0
# alone isn't always enough on Groq's inference hardware, so a fixed seed
# is added on top for extra stability.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, model_kwargs={"seed": 42})


# TypedDict for AgentState
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Base weights used when the recruiter evaluates on ALL categories.
# These always sum to 100. If the recruiter selects a subset, app.py
# recomputes proportional weights for just that subset (still summing
# to 100) and writes them to criteria.json before the pipeline runs.
BASE_WEIGHTS = {
    "Skills": 25,
    "Education": 10,
    "Projects": 20,
    "Experience": 30,
    "Achievements & Certifications": 10,
    "Extra (Hobbies)": 5,
}

CATEGORY_DESCRIPTIONS = {
    "Skills": (
        "Skills Match: {w} points. Award based on how many of the JD's required technical "
        "skills, tools, and languages are demonstrably present in the resume (skills section, "
        "projects, or experience). Partial credit for closely related/transferable skills."
    ),
    "Education": (
        "Education Match: {w} points. Full marks if the candidate's degree/field matches what "
        "the JD requires. Award 0 if the field does not match at all, regardless of degree level."
    ),
    "Projects": (
        "Projects Match: {w} points. Award based on how relevant the candidate's academic, "
        "personal, or portfolio projects are to the job description's domain and required "
        "skills. Consider relevance, technical depth, and any quantifiable impact/results "
        "mentioned. This is evaluated independently of professional work experience."
    ),
    "Experience": (
        "Experience Match: {w} points, scored in tiers based on the candidate's actual "
        "professional work experience (jobs/internships only - not academic projects):\n"
        "            - Tier 1 ({tier1} out of {w}): relevant work experience, but in a "
        "different industry/sector than the JD.\n"
        "            - Tier 2 ({tier2} out of {w}): work experience in the SAME industry/sector "
        "as the JD, even if the specific role differs.\n"
        "            - Tier 3 ({w} out of {w}, full marks): work experience in the SAME "
        "industry/sector AND a similar/related job role.\n"
        "            - If there is no relevant work experience at all, award 0.\n"
        "            - Also separately classify the JD's required experience level as one of: "
        "Fresher (0 yrs expected), Junior (0-2 yrs), Intermediate (2-10 yrs), Advanced (10+ yrs), "
        "and note in your summary whether the candidate's years of experience fit that band."
    ),
    "Achievements & Certifications": (
        "Achievements & Certifications: {w} points. Award for explicitly listed certifications, "
        "competitive exam ranks (e.g. GATE), awards, or published work relevant to the field. "
        "Do not invent achievements that are not explicitly stated in the resume."
    ),
    "Extra (Hobbies)": (
        "Extra (Hobbies/Interests): {w} points. Award for hobbies or interests that show "
        "soft skills, leadership, or culture fit relevant to the role (e.g., teaching, "
        "community involvement, communication-heavy hobbies). Do not penalize for hobbies "
        "unrelated to the role; award partial credit for general well-roundedness."
    ),
}


def load_criteria():
    """Read which categories the recruiter chose to evaluate on, and their
    weights (already redistributed by app.py to sum to 100). Falls back to
    all categories at base weights if no selection file exists, so the
    pipeline still works if this is run outside the Streamlit app."""
    try:
        with open("criteria.json", "r", encoding="utf-8") as f:
            weights = json.load(f)
            if weights:
                return weights
    except Exception:
        pass
    return dict(BASE_WEIGHTS)


def build_rubric_text(weights: dict) -> str:
    """Turn the selected {category: weight} dict into the rubric block of
    the prompt, dynamically - only the categories the recruiter picked are
    included, each with its actual point allocation."""
    lines = []
    for category, w in weights.items():
        template = CATEGORY_DESCRIPTIONS.get(category)
        if not template:
            continue
        if category == "Experience":
            tier1 = round(w * 0.3)
            tier2 = round(w * 0.5)
            lines.append("        - " + template.format(w=w, tier1=tier1, tier2=tier2))
        else:
            lines.append("        - " + template.format(w=w))
    return "\n".join(lines)


# ----------------- Resume Name Agent -----------------
def agent(agentState: AgentState):
    try:
        pdf_file = "Resume.pdf"
        data = PyPDFLoader(pdf_file).load()
        resume_text = " ".join([page.page_content for page in data])
        response = llm.invoke(
            f"Your task is to extract the candidate name and contact details from the resume data. "
            f"Only respond with the candidate name, contact details and nothing else. Resume Data: {resume_text}"
        )
        answer = response.content
    except Exception as ex:
        answer = f"Error extracting name: {ex}"
    return {"messages": [answer]}


# ----------------- Job Description Agent -----------------
def JD_agent(agentState: AgentState):
    try:
        with open("JD.txt", "r", encoding="utf-8") as f:
            jd_data = f.read()
        response = llm.invoke(
            f"Your task is to extract the exact job requirements from the given data. "
            f"Only respond with the job requirements and nothing else. Data: {jd_data}"
        )
        result = response.content.replace("\n", "")
    except Exception as ex:
        result = f"Error extracting job description: {ex}"
    return {"messages": [result]}


# Which red-flag checks are relevant to each evaluation category. The
# Red Flag agent only runs the checks tied to categories the recruiter
# actually selected - e.g. if "Experience" isn't selected, it won't flag
# job-hopping or employment gaps, since the recruiter isn't scoring on
# experience at all this time.
CATEGORY_REDFLAG_CHECKS = {
    "Skills": [
        "Claims a skill/technology (e.g. in a Skills section) with no supporting "
        "evidence anywhere in projects or experience (an unsubstantiated claim).",
    ],
    "Education": [
        "Missing education details (no degree, no institution, or no completion/expected year).",
        "Education field does not match what would typically be required for this kind of role.",
    ],
    "Projects": [
        "Projects listed with no links, no metrics, or no clear description of what was built "
        "or the candidate's specific contribution (hard to verify or assess impact).",
    ],
    "Experience": [
        "Frequent job switching (e.g., jobs lasting <1 year repeatedly).",
        "Unexplained employment gaps.",
        "Irrelevant experience relative to the kind of role implied by the resume's skills/projects.",
        "Lack of relevant work experience for technical claims made elsewhere in the resume.",
    ],
    "Achievements & Certifications": [
        "Achievements, awards, or certifications mentioned with no way to verify them (no link, "
        "issuing body, date, or score/metric).",
    ],
    "Extra (Hobbies)": [
        "Hobbies/interests section that is inconsistent with the rest of the resume or appears "
        "to pad the resume without adding real signal.",
    ],
}

# These checks are about overall resume quality/professionalism, not tied
# to any specific scoring category, so they always run regardless of what
# the recruiter selected.
ALWAYS_ON_CHECKS = [
    "Spelling, grammar, or formatting errors.",
    "Malformed or broken contact links (LinkedIn, GitHub, email, phone).",
]


# ----------------- Red Flag Detection Agent -----------------
def redflag_agent(agentState: AgentState):
    try:
        pdf_file = "Resume.pdf"
        data = PyPDFLoader(pdf_file).load()
        resume_text = " ".join([page.page_content for page in data])

        weights = load_criteria()
        selected_categories = list(weights.keys())

        check_lines = list(ALWAYS_ON_CHECKS)
        for cat in selected_categories:
            check_lines.extend(CATEGORY_REDFLAG_CHECKS.get(cat, []))

        checks_text = "\n".join(f"        - {c}" for c in check_lines)

        prompt = f"""
        You are a Resume Screening Assistant.

        Your task is to analyze the candidate's resume and identify any potential red flags or
        concerns a recruiter might have, but ONLY for the following areas (the recruiter has
        chosen to evaluate this candidate on: {", ".join(selected_categories)}, plus general
        resume-quality issues which always apply):

{checks_text}

        Do NOT raise concerns about any area not listed above (for example, if experience-related
        checks are not listed, do not comment on job gaps or job-hopping at all).

        Return a list of clear points like:
        - "Employment gap between 2020-2022"
        - "Mentions Python skills but no project or job experience using it"
        - "No education information found"

        Resume Data: {resume_text}
        """

        response = llm.invoke(prompt)
        result = response.content
    except Exception as ex:
        result = f"Error in redflag agent: {ex}"
    return {"messages": [result]}


# ----------------- Recruit Agent (Evaluation) -----------------
def recruit_agent(agentState: AgentState):
    try:
        pdf_file = "Resume.pdf"
        data = PyPDFLoader(pdf_file).load()
        resume_text = " ".join([page.page_content for page in data])

        # Read the ORIGINAL, full job description straight from JD.txt
        # instead of a summarized version, so no structural context
        # (like "this JD covers multiple roles/streams") gets lost.
        with open("JD.txt", "r", encoding="utf-8") as f:
            jd_data = f.read()

        weights = load_criteria()
        total_possible = sum(weights.values())
        rubric_text = build_rubric_text(weights)
        category_list = ", ".join(weights.keys())

        prompt = f"""
        You are a Recruitment AI Assistant.

        Your task is to evaluate how well a candidate's resume matches a given job description,
        using ONLY the categories the recruiter has chosen to evaluate on. Follow these steps
        in order.

        STEP 0 - Multi-role check:
        First check whether the Job Description describes ONE specific job role, OR MULTIPLE
        distinct job roles / streams / tracks the candidate could end up in (this happens when a
        JD lists several possible teams/streams/roles and says the actual assignment will be
        decided later based on business needs or academic/technical background).

        - If it is ONE role: do STEPS 1-2 below exactly once, for that role.
        - If it lists MULTIPLE roles/streams: do STEPS 1-2 separately for EACH distinct
          role/stream, scoring the candidate only against the requirements relevant to that
          specific role/stream (general/company-wide requirements still apply to all). Give each
          role its own heading, its own score table, and its own recommendation line:

          #### <Role/Stream Name> - <total score>/{total_possible}
          <score breakdown table for this role, same table format as described in Step 2>
          <recommendation line for this role, same rules as described in Step 2>

          After all individual roles are scored, add ONE final line summarizing the comparison
          across all roles, in this exact style:
          "Candidate is best fit for <role(s) with the highest score(s)>, and not a fit for
          <role(s) with low scores>."

        STEP 1 - The recruiter has selected the following evaluation categories: {category_list}.
        The total possible score is {total_possible} (out of the categories chosen; this is NOT
        always out of 100 - it depends on which categories the recruiter picked). Score the
        candidate using EXACTLY this rubric, and no other categories:

{rubric_text}

        STEP 2 - After scoring (for a single role, or for EACH role if multi-role), return:
        1. Total score (out of {total_possible})
        2. Score breakdown by category - you MUST format this as a Markdown table, always, with
           exactly these columns: | Category | Points Earned | Points Possible | Comments |
           Use a proper Markdown table header separator row (|---|---|---|---|) right after the
           header row. Include ONE row for each of these categories, in this order: {category_list}.
           Do not add extra categories and do not use plain inline text - it must be a table.
        3. A short summary (3-4 lines) covering major strengths and missing areas, based only on
           the categories evaluated.
        4. A final recommendation, based on the percentage score (points earned / {total_possible}):
            - If the candidate scores 70% or above:
                - Say: I recommend this candidate for the job.
            - If the candidate scores between 50% and 69%:
                - Say: I do not recommend this candidate for this specific job.
                - Then separately evaluate and state whether the candidate would be a good fit
                  for an internship or entry-level position instead, with a one-line reason why
                  or why not.
            - If the candidate scores below 50%:
                - Say: I do not recommend this candidate for the job.
                - Follow with a reason based on the biggest gaps among the evaluated categories.

        Resume Data:
        {resume_text}

        Job Description:
        {jd_data}
        """

        response = llm.invoke(prompt)
        answer = response.content
    except Exception as ex:
        answer = f"Error in recruit agent: {ex}"
    return {"messages": [answer]}