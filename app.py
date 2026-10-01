import os
import re
import json
import streamlit as st
from IQ_agents import *
from langgraph.graph import StateGraph, END


def extract_score(text: str):
    """Pull a representative match percentage out of the Recruiter agent's
    free-text output. The total possible score is no longer always 100 -
    it depends on which categories the recruiter selected - so we look for
    "<earned>/<possible>" pairs and compute a percentage from them. When the
    JD covers multiple roles/streams, several such pairs will be present;
    we use the HIGHEST percentage, since that reflects whether the
    candidate is a strong fit for at least one of the roles on offer."""
    percentages = []
    for match in re.finditer(r"(\d{1,3})\s*(?:/|out of)\s*(\d{1,3})", text, re.IGNORECASE):
        earned, possible = int(match.group(1)), int(match.group(2))
        if possible > 0 and 0 <= earned <= possible:
            percentages.append((earned / possible) * 100)

    if percentages:
        return max(percentages)

    match = re.search(r"total score[^0-9]{0,15}(\d{1,3})", text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score
    return None


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    # e.g. "|---|:---:|---|" - only made of |, -, :, and spaces
    stripped = line.strip()
    return _is_table_row(stripped) and re.fullmatch(r"[\|\-\:\s]+", stripped) is not None


def _split_row(line: str):
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def _render_table(row_lines):
    header_cells = _split_row(row_lines[0])
    body_rows = row_lines[2:] if len(row_lines) > 1 and _is_table_separator(row_lines[1]) else row_lines[1:]

    thead = "".join(f"<th style='padding:6px 10px;border:1px solid rgba(0,0,0,0.25);text-align:left;'>{c}</th>" for c in header_cells)
    trs = []
    for row in body_rows:
        cells = _split_row(row)
        tds = "".join(f"<td style='padding:6px 10px;border:1px solid rgba(0,0,0,0.25);'>{c}</td>" for c in cells)
        trs.append(f"<tr>{tds}</tr>")

    return (
        "<table style='border-collapse:collapse;width:100%;margin:8px 0;font-size:0.95em;'>"
        f"<thead><tr>{thead}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    )


def markdown_to_html(text: str) -> str:
    """Convert the LLM's lightweight Markdown (#### headings, **bold**,
    *italics*, - bullets, tables, newlines) into real HTML so it renders
    properly inside a raw <div>, instead of showing literal #, *, and |
    characters."""
    # Bold: **text** -> <b>text</b>  (must run before the italics regex)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italics: *text* -> <i>text</i> (restricted to a single line so it
    # can't accidentally swallow bullet markers or table rows on other lines)
    text = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"<i>\1</i>", text)

    lines = text.split("\n")
    html_lines = []
    in_list = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- Markdown table block ---
        if _is_table_row(stripped):
            table_lines = []
            while i < len(lines) and _is_table_row(lines[i].strip()):
                table_lines.append(lines[i].strip())
                i += 1
            html_lines.append(_render_table(table_lines))
            continue

        # --- Heading lines (#, ##, ####, etc.) -> bold text, no literal #s ---
        heading_match = re.match(r"^#{1,6}\s*(.+)$", stripped)
        if heading_match:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<b>{heading_match.group(1).strip()}</b><br>")
            i += 1
            continue

        # --- Bullet list ---
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul style='margin-top:4px;margin-bottom:4px;'>")
                in_list = True
            html_lines.append(f"<li>{stripped[2:].strip()}</li>")
            i += 1
            continue

        if in_list:
            html_lines.append("</ul>")
            in_list = False

        if stripped:
            html_lines.append(stripped + "<br>")
        i += 1

    if in_list:
        html_lines.append("</ul>")
    return "".join(html_lines)


def render_neutral_box(label: str, content: str):
    """White background, black text - used for the analysis-stage agent
    outputs (Resume, JD, Red Flag) that are informational, not a verdict."""
    formatted = markdown_to_html(content)
    st.markdown(
        f"""
        <div style="background-color:#ffffff;color:#000000;
                    border:1px solid #ddd;border-radius:8px;
                    padding:16px;margin-bottom:12px;">
            <b>{label}</b><br>{formatted}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict_box(content: str, score):
    """Colored verdict box for the Recruiter agent's final call.
    Green = best fit (score >= 70), Yellow = internship-level (50-69),
    Red = not recommended (< 50 or score not found and text reads as a reject)."""
    if score is not None and score >= 70:
        bg, fg = "#d4edda", "#155724"   # light green / dark green text
    elif score is not None and 50 <= score < 70:
        bg, fg = "#fff3cd", "#856404"   # light yellow / dark yellow-brown text
    elif score is not None and score < 50:
        bg, fg = "#f8d7da", "#721c24"   # light red / dark red text
    else:
        # Fallback if a score number couldn't be parsed from the text -
        # guess from wording so the box is never left uncolored.
        lowered = content.lower()
        if "i recommend this candidate for the job" in lowered and "do not recommend" not in lowered:
            bg, fg = "#d4edda", "#155724"
        elif "internship" in lowered:
            bg, fg = "#fff3cd", "#856404"
        else:
            bg, fg = "#f8d7da", "#721c24"

    formatted = markdown_to_html(content)
    st.markdown(
        f"""
        <div style="background-color:{bg};color:{fg};
                    border-radius:8px;padding:16px;margin-bottom:12px;
                    font-weight:500;">
            <b>Recruiter_agent Output:</b><br>{formatted}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(layout="wide")
    st.title("ScreenIQ")

    st.markdown("""
    <div style="background-color:#003366;padding:10px">
        <h2 style="color:white;text-align:center;">
            Screening using LangGraph, RAG, and Llama3
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Upload resume PDF
    pdf_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    if pdf_file is not None:
        with open("Resume.pdf", "wb") as f:
            f.write(pdf_file.read())

    # Upload JD as PDF or TXT, or paste manually
    jd_file = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"])
    job_description = ""

    if jd_file is not None:
        if jd_file.name.lower().endswith(".pdf"):
            with open("JD_upload.pdf", "wb") as f:
                f.write(jd_file.read())
            try:
                jd_pages = PyPDFLoader("JD_upload.pdf").load()
                job_description = " ".join([page.page_content for page in jd_pages])
            except Exception as ex:
                st.error(f"Could not read the JD PDF: {ex}")
        else:
            job_description = jd_file.read().decode("utf-8", errors="ignore")
    else:
        job_description = st.text_area("Or paste the Job Description here:")

    if job_description.strip() != "":
        with open("JD.txt", "w", encoding="utf-8") as f:
            f.write(job_description)

    # ---- Recruiter-selectable evaluation criteria ----
    st.markdown("### Evaluate candidate on:")
    category_options = list(BASE_WEIGHTS.keys())  # Skills, Education, Projects, Experience, Achievements & Certifications, Extra (Hobbies)

    all_selected = st.checkbox("All", value=True)
    cols = st.columns(3)
    manual_selection = []
    for idx, cat in enumerate(category_options):
        with cols[idx % 3]:
            checked = st.checkbox(cat, value=all_selected, disabled=all_selected, key=f"crit_{cat}")
            if checked:
                manual_selection.append(cat)

    selected_categories = category_options if all_selected else manual_selection

    if not selected_categories:
        st.warning("Select at least one evaluation category (or check 'All').")

    # Recompute weights proportionally so they always sum to 100, based on
    # only the categories the recruiter picked.
    selected_base_sum = sum(BASE_WEIGHTS[c] for c in selected_categories)
    final_weights = {}
    if selected_base_sum > 0:
        for c in selected_categories:
            final_weights[c] = round(100 * BASE_WEIGHTS[c] / selected_base_sum)
        # Rounding can leave the total a point or two off 100 - fix it by
        # adjusting whichever category has the largest weight.
        diff = 100 - sum(final_weights.values())
        if diff != 0 and final_weights:
            biggest = max(final_weights, key=final_weights.get)
            final_weights[biggest] += diff

    with open("criteria.json", "w", encoding="utf-8") as f:
        json.dump(final_weights, f)

    # Start pipeline
    if st.button("Match Resume"):
        if pdf_file is None:
            st.error("Please upload a resume PDF first.")
            return
        if not os.path.exists("JD.txt"):
            st.error("Please upload or paste a job description first.")
            return
        if not selected_categories:
            st.error("Please select at least one evaluation category above.")
            return

        inputs = {
            "messages": ["You are a recruitment expert and your role is to match a candidate's profile with a given job description."]
        }

        with st.spinner("Running the agent pipeline... this can take 20-40 seconds"):
            workflow = StateGraph(AgentState)
            workflow.add_node("Resume_agent", agent)
            workflow.add_node("JD_agent", JD_agent)
            workflow.add_node("Redflag_agent", redflag_agent)
            workflow.add_node("Recruiter_agent", recruit_agent)

            workflow.set_entry_point("Resume_agent")

            workflow.add_edge("Resume_agent", "JD_agent")
            workflow.add_edge("Resume_agent", "Redflag_agent")
            workflow.add_edge("JD_agent", "Recruiter_agent")
            workflow.add_edge("Redflag_agent", "Recruiter_agent")
            workflow.add_edge("Recruiter_agent", END)
            compiled_app = workflow.compile()

            results = []
            try:
                outputs = compiled_app.stream(inputs)
                for output in outputs:
                    for key, value in output.items():
                        messages = value.get("messages", [])
                        for msg in messages:
                            results.append((key, msg))
            except Exception as ex:
                st.error(f"Pipeline failed: {ex}")
                return

        # Show results - analysis-stage agents get a neutral white/black box,
        # the Recruiter agent's final verdict gets colored by outcome.
        st.markdown("## Results")
        for agent_name, content in results:
            if agent_name == "Recruiter_agent":
                score = extract_score(content)
                render_verdict_box(content, score)
            else:
                render_neutral_box(f"{agent_name} Output:", content)


if __name__ == "__main__":
    main()