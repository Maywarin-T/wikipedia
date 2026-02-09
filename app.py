# Market Research Assistant
# A Streamlit app that generates market research reports using Wikipedia + Google Gemini.
# The user types an industry, we find the best Wikipedia pages, and Gemini writes a report.

# Standard library stuff
import time
import json
import io
import re

# Streamlit for the web app
import streamlit as st

# Wikipedia retrieval from LangChain
from langchain_community.retrievers import WikipediaRetriever

# Google Gemini LLM
from google import genai
from google.genai import types

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# Page setup (must be first Streamlit command)
st.set_page_config(page_title="Market Research Assistant", page_icon="📊")
st.title("📊 Market Research Assistant")
st.markdown(
    "Enter an industry name below and I'll generate a concise market research report "
    "based on the most relevant Wikipedia pages."
)


# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google Gemini API Key", type="password")
    model_options = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-preview-05-20", "Custom"]
    model_choice = st.selectbox("Model", model_options, index=0)
    if model_choice == "Custom":
        model_name = st.text_input("Enter model name", placeholder="e.g. gemini-1.5-pro")
    else:
        model_name = model_choice


# HELPER: Call Gemini with retry logic
# Gemini's free tier has low rate limits, so we retry with increasing waits.
# This one function handles all LLM calls so we don't repeat retry code everywhere.
def call_gemini(client, model, prompt, system_instruction=None, temperature=0.0):
    max_retries = 5
    config = types.GenerateContentConfig(temperature=temperature)
    if system_instruction:
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait = 60 * attempt
                time.sleep(wait)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "RESOURCE_EXHAUSTED" in error_msg and attempt < max_retries - 1:
                continue
            elif "API_KEY_INVALID" in error_msg or "PERMISSION_DENIED" in error_msg:
                return None
            else:
                return None
    return None


# HELPER: Parse JSON from Gemini's response
# Gemini sometimes wraps JSON in ```json ... ``` markdown blocks.
# This strips that away so we can parse the actual JSON.
def parse_json(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


# HELPER: Turn Wikipedia documents into plain text for Gemini
# Each page gets a title, URL, and the first N characters of content.
def docs_to_text(docs, max_chars=5000):
    parts = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        url = doc.metadata.get("source", "")
        content = doc.page_content[:max_chars]
        parts.append(f"### {title}\nSource: {url}\n\n{content}")
    return "\n\n".join(parts)


# HELPER: Generate PDF report
# Creates a nice PDF with the report text, sources, and SWOT table.
def generate_pdf(industry, report_text, sources, swot_data=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionHead", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.grey))
    styles.add(ParagraphStyle(name="Cell", parent=styles["Normal"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="CellBold", parent=styles["Normal"], fontSize=10, leading=13, fontName="Helvetica-Bold"))

    story = []

    # Title
    story.append(Paragraph(f"{industry} Industry Report", styles["ReportTitle"]))
    story.append(Spacer(1, 4))

    # Sources
    story.append(Paragraph("Sources", styles["SectionHead"]))
    for i, src in enumerate(sources, 1):
        story.append(Paragraph(
            f'{i}. <a href="{src["url"]}" color="blue">{src["title"]}</a>',
            styles["Body"],
        ))
    story.append(Spacer(1, 6))

    # Report body
    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["SectionHead"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["SectionHead"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], styles["SectionHead"]))
        elif line.startswith("**") and line.endswith("**"):
            story.append(Paragraph(line.strip("*"), styles["SectionHead"]))
        else:
            clean = line.replace("**", "")
            story.append(Paragraph(clean, styles["Body"]))

    story.append(Spacer(1, 10))

    # SWOT Analysis table
    if swot_data:
        story.append(Paragraph("SWOT Analysis", styles["SectionHead"]))
        story.append(Spacer(1, 4))

        swot_factors = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
        table_data = [[
            Paragraph("<b>Factor</b>", styles["CellBold"]),
            Paragraph("<b>Details</b>", styles["CellBold"]),
        ]]
        for factor in swot_factors:
            points = swot_data.get(factor, [])
            bullets = "<br/>".join(f"• {p}" for p in points)
            table_data.append([
                Paragraph(f"<b>{factor}</b>", styles["Cell"]),
                Paragraph(bullets, styles["Cell"]),
            ])
        t = Table(table_data, colWidths=[doc.width * 0.25, doc.width * 0.75])
        t.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1.5, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)

    story.append(Spacer(1, 16))
    story.append(Paragraph("Generated by Market Research Assistant | Sources: Wikipedia", styles["Small"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# STEP 1: Check the user's input (Q1, 25 marks)
# The user types an industry name. We need to make sure it's valid before
# doing anything else. This step has two parts:
# 1a. Basic checks (empty, symbols, too short)
# 1b. Ask Gemini if this is a real industry, and get key player names

st.subheader("Step 1: Provide an Industry")

# We use session_state to remember results so they don't disappear
# when the user clicks buttons or interacts with the UI.
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "report_data" not in st.session_state:
    st.session_state.report_data = None

# If the user just picked a suggestion, write it into the text input
# before the widget renders so it shows up in the box.
if st.session_state.get("_picked_suggestion"):
    st.session_state.industry_input = st.session_state._picked_suggestion
    st.session_state._picked_suggestion = None
    st.session_state.suggestions = []

# The key="industry_input" lets Streamlit remember what the user typed
# across reruns. We only override it when picking a suggestion (above).
industry = st.text_input(
    "Industry",
    key="industry_input",
    placeholder="e.g. Renewable Energy, Semiconductor, Pharmaceutical",
)

run = st.button("Generate Report")

# If Gemini previously suggested industries, show them as buttons.
# The user picks one and it fills the text box.
if st.session_state.suggestions:
    st.info("Did you mean one of these?")
    for suggestion in st.session_state.suggestions:
        if st.button(f"✅ {suggestion}", key=f"suggest_{suggestion}"):
            st.session_state._picked_suggestion = suggestion
            st.rerun()

if run:
    # If there are old suggestions showing, clear them and rerun
    # so they disappear before we start the pipeline.
    if st.session_state.suggestions:
        st.session_state.suggestions = []
        st.rerun()

    # Step 1a: Basic checks
    # Make sure the user actually typed something sensible.

    if not api_key:
        st.error("Please enter your Google Gemini API key in the sidebar.")
        st.stop()

    if not industry or industry.strip() == "":
        st.warning("Please enter an industry name.")
        st.info("Examples: Renewable Energy, Semiconductor, Pharmaceutical, Automotive")
        st.stop()

    industry = industry.strip()

    # Allow letters, numbers, spaces, hyphens, parentheses, ampersand, and slashes.
    # These cover common industry names like "Artificial Intelligence (AI)",
    # "Oil & Gas", "IT/Technology". Block things like @#$%!^* that are clearly wrong.
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -()&/,.")
    if not all(c in allowed for c in industry):
        st.warning("Please enter an industry name without special characters like @, #, $, !, etc.")
        st.info("Examples: Renewable Energy, Semiconductor, Artificial Intelligence (AI)")
        st.stop()

    # Need at least 2 letters so we don't get things like "1" or "a".
    alpha_count = sum(1 for c in industry if c.isalpha())
    if alpha_count < 2:
        st.warning("Please enter a valid industry name with at least 2 letters.")
        st.info("Examples: AI, IT, Renewable Energy, Semiconductor")
        st.stop()

    # Set up the Gemini client
    client = genai.Client(api_key=api_key)

    st.success(f"✅ Generating report for: **{industry}**")

    # Step 1b: Ask Gemini if this is a real industry and get key players
    # We combine two tasks into one LLM call to save API usage:
    # 1. Is this input actually an industry?
    # 2. If yes, who are the 2-3 biggest companies in it?
    with st.spinner("Validating industry..."):
        validate_prompt = (
            f'Is "{industry}" a recognized industry or industry sector?\n\n'
            f"If YES, respond with this JSON:\n"
            f'{{"is_industry": true, "key_players": ["Company1", "Company2", "Company3"]}}\n'
            f"List the 2-3 most dominant companies globally in this industry.\n\n"
            f"If NO (e.g. it's a company name, a person, a random word, or a misspelling), "
            f"suggest 3-5 closely related real industry names. Respond with:\n"
            f'{{"is_industry": false, "suggestions": ["Industry 1", "Industry 2", "Industry 3", "Industry 4"]}}\n\n'
            f"Return ONLY the JSON, nothing else."
        )
        validate_text = call_gemini(client, model_name, validate_prompt)
        validate_result = parse_json(validate_text)

    # If validate returns None, the API key might be bad or the model unavailable.
    if validate_text is None:
        st.error("Could not connect to Gemini. Please check your API key and model name.")
        st.stop()

    if validate_result is None:
        st.warning("Could not validate the industry. Proceeding anyway...")
        key_players = []
    elif not validate_result.get("is_industry", True):
        suggestions = validate_result.get("suggestions", [])
        st.warning(f'"{industry}" doesn\'t look like a recognized industry name.')
        if suggestions:
            # Save the suggestions and rerun so the buttons appear
            st.session_state.suggestions = suggestions
            st.rerun()
        else:
            st.info("Try something like: Renewable Energy, Semiconductor, Pharmaceutical")
        st.stop()
    else:
        key_players = validate_result.get("key_players", [])
        # Key players are used internally for Wikipedia search, no need to show them.


    # STEP 2: Get the 5 best Wikipedia pages (Q2, 25 marks)
    # Wikipedia search by itself isn't great at finding the right pages.
    # Sometimes you get unrelated articles that just happen to share keywords.
    # So we do two things:
    # 1. Search Wikipedia for the industry name AND for each key player
    # 2. Let Gemini read the actual content of each page and pick the best 5

    st.subheader("Step 2: Retrieving Wikipedia Pages")

    progress = st.progress(0, text="Searching Wikipedia...")

    # Step 2a: Search Wikipedia for industry pages and key player pages
    # We search for the industry name to get general overview pages,
    # and for each key player to get company-specific pages.
    retriever = WikipediaRetriever(top_k_results=8, load_max_docs=8)
    all_docs = []
    seen_titles = set()

    # Search for the industry with two queries to get a good spread of pages.
    # One for "industry" overview pages and one for "market" pages.
    for query in [f"{industry} industry", f"{industry} market"]:
        try:
            results = retriever.invoke(query)
            for doc in results:
                title = doc.metadata.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_docs.append(doc)
        except Exception:
            continue

    # Search for each key player from Step 1c
    player_retriever = WikipediaRetriever(top_k_results=1, load_max_docs=1)
    for player in key_players:
        try:
            results = player_retriever.invoke(player)
            for doc in results:
                title = doc.metadata.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_docs.append(doc)
        except Exception:
            continue

    progress.progress(40, text="Filtering candidates...")

    if not all_docs:
        st.error(f"No Wikipedia pages found for '{industry}'. Try a broader or different industry name.")
        st.stop()

    # Step 2b: Remove junk pages
    # Wikipedia has meta pages like "List of..." or "Category:..." that
    # are useless for a market report. We also skip country-specific pages
    # like "Electronics industry in Japan" because we want a global perspective.
    junk_keywords = ["disambiguation", "list of", "index of", "category:", "portal:", "template:"]
    candidates = []
    for doc in all_docs:
        title = doc.metadata.get("title", "").lower()
        if any(kw in title for kw in junk_keywords):
            continue
        # Skip country-specific articles like "Electronics industry in Japan".
        # We check if the title contains " in " followed by a known pattern.
        title_original = doc.metadata.get("title", "")
        if " in " in title_original:
            after_in = title_original.split(" in ", 1)[-1].strip()
            # Only skip if what comes after "in" looks like a country/region name
            # (starts with uppercase and is short, like "Japan", "Bangladesh", "the United States")
            if after_in and after_in[0].isupper() and len(after_in.split()) <= 4:
                continue
        if len(doc.page_content.split()) < 100:
            continue
        candidates.append(doc)

    if not candidates:
        st.error("No useful Wikipedia pages found. Try a different industry name.")
        st.stop()

    progress.progress(60, text="Checking relevance...")

    # Step 2c: Let Gemini pick the best 5 by reading actual content
    # This is the key part. We don't just match titles, we send Gemini
    # the first 300 words of each article so it can verify the page
    # is actually about our industry.
    snippets = []
    for i, doc in enumerate(candidates):
        title = doc.metadata.get("title", "Unknown")
        preview = " ".join(doc.page_content.split()[:300])
        snippets.append(f"[{i}] {title}\n{preview}")

    select_prompt = (
        f"You are selecting Wikipedia articles for a GLOBAL market research report about "
        f"the **{industry}** industry.\n\n"
        f"Read the CONTENT of each article below (not just the title). "
        f"An article is relevant ONLY if its MAIN TOPIC is directly about:\n"
        f"  - The {industry} industry itself (overview, history, market data)\n"
        f"  - A company whose PRIMARY business is in the {industry} industry\n"
        f"  - A product, service, or technology that is CORE to the {industry} industry\n\n"
        f"REJECT articles where:\n"
        f"  - The article only MENTIONS {industry} briefly but is mainly about something else\n"
        f"  - The article is about a general topic (e.g. a country's economy, a broad technology) "
        f"that is not specifically about the {industry} industry\n"
        f"  - The company or topic belongs to a different industry\n"
        f"  - The focus is on a single country or region only\n\n"
        f"Among the relevant articles, PREFER ones with market size, revenue, or growth data.\n\n"
        f"Articles:\n\n" + "\n\n---\n\n".join(snippets) + "\n\n"
        f"Return ONLY a JSON array of the indices of the relevant articles, best first.\n"
        f"Example: [2, 0, 5, 3, 7]\n"
        f"If fewer than 5 are truly relevant, return fewer. Do NOT pad with loosely related articles."
    )

    with st.spinner("Selecting most relevant pages..."):
        select_text = call_gemini(client, model_name, select_prompt)
        selected_indices = parse_json(select_text)

    # If Gemini's selection works, use it. Otherwise fall back to first 5 candidates.
    docs = []
    if selected_indices and isinstance(selected_indices, list):
        for idx in selected_indices[:5]:
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                docs.append(candidates[idx])

    # If Gemini returned nothing useful, just use the first 5 candidates.
    # This can happen if Gemini's response was malformed or indices were wrong.
    if not docs:
        docs = candidates[:5]

    progress.progress(100, text="Done!")

    if len(docs) < 5:
        st.info(
            f" It looks like **{industry}** might not have enough Wikipedia coverage "
            f"to provide 5 relevant pages as expected. Could you try a broader or "
            f"slightly different industry name so I can find better sources for you?"
        )
        st.stop()

    # Show the selected pages as clickable links
    sources = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        url = doc.metadata.get("source", "")
        if url:
            sources.append({"title": title, "url": url})

    st.markdown(f"**Top {len(sources)} relevant Wikipedia pages:**")
    for i, src in enumerate(sources, 1):
        st.markdown(f"{i}. [{src['title']}]({src['url']})")


    # STEP 3: Generate the report and SWOT analysis (Q3, 25 marks)
    # We send all the Wikipedia content to Gemini in two focused calls:
    # one for the market research report and one for the SWOT analysis.

    st.subheader("Step 3: Industry Report")

    # Turn the Wikipedia pages into text that Gemini can read
    context = docs_to_text(docs, max_chars=5000)

    # Check if the Wikipedia sources contain any recent data (2020+).
    # If not, warn the user that the report may lack up-to-date figures.
    recent_years = re.findall(r'\b(202[0-9])\b', context)
    if not recent_years:
        st.warning(
            "⚠️ The Wikipedia sources for this industry do not appear to contain "
            "recent data (2020 or later). The report may rely on qualitative insights "
            "rather than up-to-date statistics."
        )

    # This is the main prompt. It tells Gemini exactly what we want:
    # a 5-section report using ONLY the Wikipedia content.
    report_system = (
        "You are a senior market research analyst writing a GLOBAL industry report. "
        "Take a worldwide perspective, not focused on any single country.\n\n"
        "Base the report ONLY on the Wikipedia sources provided. Do NOT add any "
        "information from your own knowledge. If a fact is not in the sources, leave it out.\n\n"
        "Do NOT include source citations or references in the report text. "
        "Keep it clean and easy to read.\n\n"
        "QUANTITATIVE DATA IS ESSENTIAL. You MUST include specific numbers wherever "
        "available in the sources: market size in dollars, growth rates in percentages, "
        "company revenue figures, market share numbers, number of employees, units sold, "
        "year-over-year changes. A good analyst always backs claims with numbers.\n\n"
        "DATA RECENCY RULES (very important, current year is 2026):\n"
        "- Always use the MOST RECENT data available in the sources (2020 or later only). "
        "If both 2022 and 2024 figures exist, use the 2024 one. Skip anything before 2020.\n"
        "- If the sources only have old data (before 2020), do NOT cite those numbers. "
        "Instead focus on qualitative insights like market structure, competitive dynamics, "
        "and strategic positioning.\n"
        "- Always include the year when citing a number, e.g. 'revenue of $520B (2023)'.\n"
        "- Never present old data as if it is current.\n\n"
        "The report MUST have exactly these 5 sections, each with a markdown heading:\n"
        "## Executive Summary\n50-70 words. The big picture and key takeaway.\n\n"
        "## Market Size and Segments\n70-90 words. Global market size in dollars, growth rate, "
        "and the main sub-segments.\n\n"
        "## Key Players\n80-100 words. Use bullet points for this section only. "
        "List 3-4 companies, each as a bullet starting with the company name in bold. "
        "Include a number (revenue, market share) for each if available.\n\n"
        "## Trends\n70-90 words. 2-3 key forces shaping this industry with data points.\n\n"
        "## Outlook\n70-80 words. Growth trajectory, main risks, and opportunities.\n\n"
        "WORD COUNT IS ABSOLUTELY CRITICAL. The report text MUST be between 330 and 370 "
        "words. Count carefully. If your draft is under 330, go back and add more detail. "
        "The SWOT analysis adds ~100-120 words on top, bringing the total to 430-490.\n"
        "Write in clear, professional English. Each section should flow into the next.\n"
        "Do NOT include a SWOT analysis in the report. Only write the 5 sections above. "
        "SWOT is handled separately."
    )

    # Step 3a: Generate the report as plain markdown (not JSON).
    # Letting the LLM write freely as markdown gives better word count
    # control than asking it to stuff text inside a JSON string.
    report_prompt = (
        f"Write a market research report on the **{industry}** industry.\n\n"
        f"Use ONLY the facts from these Wikipedia sources:\n\n{context}\n\n"
        f"Write the report as clean markdown with ## headings.\n"
        f"REMEMBER: 400 to 440 words. No less, no more."
    )

    with st.spinner("Generating report..."):
        report = call_gemini(client, model_name, report_prompt, system_instruction=report_system)

    if not report:
        st.error("Failed to generate the report. Please try again.")
        st.stop()

    # Step 3b: Generate SWOT analysis in a separate quick call.
    # SWOT (Strengths, Weaknesses, Opportunities, Threats) works much better
    # with Wikipedia data than PESTEL because Wikipedia articles naturally
    # cover things like industry advantages, challenges, and future trends.
    swot_prompt = (
        f"You are a senior market research analyst. Based on the Wikipedia sources "
        f"about the **{industry}** industry, provide a SWOT analysis.\n\n"
        f"Analyze the INDUSTRY AS A WHOLE, not any single company.\n"
        f"Base your analysis ONLY on the Wikipedia sources provided.\n\n"
        f"For each factor, provide exactly 2 bullet points. Each bullet should be "
        f"10-18 words, specific and detailed enough to be useful to a business analyst. "
        f"Include numbers or company names where relevant.\n\n"
        f"Return ONLY JSON:\n"
        f'{{"Strengths": ["point 1", "point 2"], '
        f'"Weaknesses": ["point 1", "point 2"], '
        f'"Opportunities": ["point 1", "point 2"], '
        f'"Threats": ["point 1", "point 2"]}}\n\n'
        f"Wikipedia context:\n{context[:3000]}"
    )

    with st.spinner("Generating SWOT analysis..."):
        swot_text = call_gemini(client, model_name, swot_prompt)
        swot_data = parse_json(swot_text)

    # Save to session state so it doesn't disappear on UI interaction
    st.session_state.report_data = {
        "industry": industry,
        "report": report,
        "swot": swot_data,
        "sources": sources,
    }

    # Display the report
    st.markdown(report.replace("$", "\\$"))

    # SWOT Analysis
    # Display as a clean 2x2 grid showing Strengths, Weaknesses,
    # Opportunities, and Threats for the industry.
    if swot_data:
        st.markdown("### SWOT Analysis")

        swot_factors = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
        # Build the entire SWOT as one HTML table so the boxes
        # are always aligned and equal height. Clean look, no icons.
        def swot_cell(factor):
            points = swot_data.get(factor, [])
            bullets = "".join(f"<p style='margin:4px 0; font-size:14px;'>{p}</p>" for p in points)
            return (
                f"<td style='border:1px solid #ddd; border-radius:8px; "
                f"padding:12px; width:50%; vertical-align:top;'>"
                f"<h4 style='margin:0 0 8px 0;'>{factor}</h4>"
                f"{bullets}</td>"
            )

        swot_html = (
            "<table style='width:100%; border-collapse:separate; border-spacing:8px;'>"
            f"<tr>{swot_cell('Strengths')}{swot_cell('Weaknesses')}</tr>"
            f"<tr>{swot_cell('Opportunities')}{swot_cell('Threats')}</tr>"
            "</table>"
        )
        st.markdown(swot_html, unsafe_allow_html=True)

    # Word count includes both the report text AND SWOT points,
    # since the assignment says "everything included except appendices and code."
    total_text = report
    if swot_data:
        for factor, points in swot_data.items():
            for point in points:
                total_text += f" {point}"
    clean = re.sub(r'[#*\-\n]+', ' ', total_text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    word_count = len(clean.split())
    st.markdown("---")
    st.caption(f"Word count (report + SWOT): {word_count}")
    if word_count >= 500:
        st.warning("Total exceeds 500 words. Consider regenerating.")

    # PDF Download
    st.markdown("---")
    st.subheader("Download Report")
    pdf_buffer = generate_pdf(
        industry=industry,
        report_text=report,
        sources=sources,
        swot_data=swot_data,
    )
    st.download_button(
        label="📥 Download Report as PDF",
        data=pdf_buffer,
        file_name=f"{industry.replace(' ', '_')}_Market_Report.pdf",
        mime="application/pdf",
    )
