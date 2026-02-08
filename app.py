# =============================================================================
# Market Research Assistant — Streamlit Application
# =============================================================================
# This app generates a market research report for any industry using:
# 1. Wikipedia as the data source (via LangChain's WikipediaRetriever)
# 2. Google Gemini LLM for intelligent page selection and report generation
# 3. PESTEL analysis with radar chart visualisation
# 4. PDF export using ReportLab
#
# Architecture:
#   User Input → Wikipedia Search → LLM Key Player Identification →
#   LLM Page Selection → LLM Report Generation → LLM PESTEL Scoring →
#   Radar Chart + PDF Export
# =============================================================================

# --- Standard library imports ---
import time          # For retry delays when hitting API rate limits
import json          # For parsing JSON responses from Gemini
import io            # For in-memory file buffers (PDF generation)
import re            # For regex-based text cleaning (word count)

# --- Data and visualisation libraries ---
import numpy as np           # For calculating radar chart angles
import matplotlib.pyplot as plt  # For drawing the PESTEL radar chart

# --- Streamlit (web app framework) ---
import streamlit as st
import pandas as pd

# --- Wikipedia retrieval (LangChain) ---
# WikipediaRetriever searches Wikipedia and returns document objects
# containing page content and metadata (title, URL)
from langchain_community.retrievers import WikipediaRetriever

# --- Google Gemini API (LLM) ---
# genai is the Google Generative AI client library
# types provides configuration objects like GenerateContentConfig
from google import genai
from google.genai import types

# --- PDF generation (ReportLab) ---
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

# =============================================================================
# Page Configuration & Welcome Message
# =============================================================================
# set_page_config must be the first Streamlit command
st.set_page_config(page_title="Market Research Assistant", page_icon="📊")
st.title("📊 Market Research Assistant")
st.markdown(
    "Welcome! I'm your market research assistant. "
    "Simply enter an industry name below and I'll generate a concise report "
    "based on the most relevant Wikipedia pages — including key players, trends, "
    "outlook, and a PESTEL analysis."
)

# =============================================================================
# Sidebar — User Settings
# =============================================================================
# The sidebar allows users to configure their API key and model choice
# without cluttering the main interface
with st.sidebar:
    st.header("Settings")

    # API key input — masked with type="password" for security
    api_key = st.text_input("Google Gemini API Key", type="password")

    # Model selection — includes preset options and a custom input field
    # so the teacher can use any model they have access to
    model_options = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-preview-05-20", "Custom"]
    model_choice = st.selectbox(
        "Model",
        model_options,
        index=0,
        help="gemini-2.0-flash is recommended. Select 'Custom' to enter your own model name.",
    )
    # If "Custom" is selected, show a text input for the user to type any model name
    if model_choice == "Custom":
        model_name = st.text_input("Enter model name", placeholder="e.g. gemini-1.5-pro")
    else:
        model_name = model_choice

    # Temperature set to 0 for deterministic, factual outputs
    # (not exposed in UI — we want consistent, reproducible reports)
    temperature = 0.0

# =============================================================================
# LLM Prompts (not displayed on the UI)
# =============================================================================
# System prompt defines the LLM's role and output format.
# It instructs Gemini to act as a senior market research analyst
# and produce a structured report with 4 sections under 500 words.
system_prompt = (
    "You are a senior market research analyst at a top-tier consulting firm. "
    "Write a concise, professional industry report based on the Wikipedia "
    "sources provided.\n\n"
    "CRITICAL: The report MUST be less than 500 words. Aim for around 400-450 words. "
    "Be thorough but concise.\n\n"
    "You MUST include ALL four of the following sections:\n"
    "1. **Industry Overview** — Define the industry scope, estimated global market size "
    "(if mentioned), key segments, and value chain.\n"
    "2. **Key Players & Market Structure** — Identify 3-5 major companies and their "
    "competitive positioning. Keep descriptions brief (1 sentence per company max).\n"
    "3. **Recent Trends & Developments** — Focus on 2-3 key technological shifts, "
    "regulatory changes, or investment trends with clear business impact.\n"
    "4. **Outlook** — Provide a forward-looking perspective on growth trajectory "
    "and emerging risks in 2-3 sentences.\n\n"
    "IMPORTANT GUIDELINES:\n"
    "- Include quantitative data (market size, growth rates, revenue) when available.\n"
    "- Name specific companies and products — avoid vague statements.\n"
    "- Write like a McKinsey analyst, not a Wikipedia summary.\n"
    "- Only state facts directly supported by the sources. Do NOT fabricate data.\n"
    "- If a data point is not in the sources, omit it — do NOT guess.\n"
    "- Use clear, business-appropriate language with a global industry perspective."
)

# User prompt template — gets filled with the industry name and Wikipedia context
user_prompt = (
    "Write a market research report on the **{industry}** industry.\n\n"
    "Extract business-relevant data from these sources. "
    "Only include facts directly stated in the sources — do not invent data.\n\n"
    "Wikipedia Sources:\n\n{context}"
)

# =============================================================================
# Helper Function: Convert Wikipedia Documents to Plain Text
# =============================================================================
def convert_docs_to_text(docs, max_chars_per_doc=3000):
    """
    Convert WikipediaRetriever document objects into a single formatted
    text string that can be sent to the LLM as context.
    
    Each document is formatted with its title, source URL, and content
    (truncated to max_chars_per_doc to stay within token limits).
    
    Args:
        docs: List of Document objects from WikipediaRetriever
        max_chars_per_doc: Maximum characters per document (prevents token overflow)
    
    Returns:
        String containing all documents formatted as plain text
    """
    context_parts = []
    for doc in docs:
        # Extract metadata from the document object
        title = doc.metadata.get("title", "Unknown")
        source_url = doc.metadata.get("source", "")
        
        # Truncate content to prevent exceeding LLM token limits
        content = doc.page_content[:max_chars_per_doc]
        
        # Format with clear section markers so the LLM can distinguish sources
        text_section = f"### {title}\n"
        if source_url:
            text_section += f"Source: {source_url}\n\n"
        text_section += content
        
        context_parts.append(text_section)
    
    return "\n\n".join(context_parts)

# =============================================================================
# Helper Function: Generate PDF Report
# =============================================================================
def generate_pdf(industry, report_text, sources, pestel_data=None):
    """
    Generate a downloadable PDF report containing:
    - Report title and Wikipedia source links
    - Full report text with markdown headings converted to PDF styles
    - PESTEL radar chart (as embedded image) and explanations table
    
    Uses ReportLab to build the PDF in memory (BytesIO buffer).
    
    Args:
        industry: Name of the industry (string)
        report_text: The generated report text from Gemini (string)
        sources: List of dicts with 'title' and 'url' keys
        pestel_data: Dict with 'scores' and 'explanations' (optional)
    
    Returns:
        BytesIO buffer containing the PDF file ready for download
    """
    from reportlab.lib.utils import ImageReader
    
    # Create an in-memory buffer to write the PDF to (no temp file needed)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )
    
    # Get default styles and add custom ones for our report layout
    styles = getSampleStyleSheet()
    
    # Large title style for the report header
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"],
        fontSize=20, spaceAfter=12,
    ))
    # Section heading style (e.g., "Industry Overview", "Key Players")
    styles.add(ParagraphStyle(
        name="SectionHead", parent=styles["Heading2"],
        fontSize=13, spaceBefore=14, spaceAfter=6,
    ))
    # Body text style for report paragraphs
    styles.add(ParagraphStyle(
        name="BodyText2", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=8,
    ))
    # Small grey text for footer
    styles.add(ParagraphStyle(
        name="SmallText", parent=styles["Normal"],
        fontSize=8, leading=10, textColor=colors.grey,
    ))
    # Table cell styles for PESTEL explanations table
    styles.add(ParagraphStyle(
        name="CellText", parent=styles["Normal"],
        fontSize=9, leading=12,
    ))
    styles.add(ParagraphStyle(
        name="CellBold", parent=styles["Normal"],
        fontSize=10, leading=13, fontName="Helvetica-Bold",
    ))
    
    # story = ordered list of PDF elements (paragraphs, tables, images)
    story = []
    
    # --- Report Title ---
    story.append(Paragraph(f"{industry} Industry Report", styles["ReportTitle"]))
    story.append(Spacer(1, 4))
    
    # --- Wikipedia Sources (displayed as clickable hyperlinks) ---
    story.append(Paragraph("Sources", styles["SectionHead"]))
    for i, src in enumerate(sources, 1):
        story.append(Paragraph(
            f'{i}. <a href="{src["url"]}" color="blue">{src["title"]}</a>',
            styles["BodyText2"],
        ))
    story.append(Spacer(1, 6))
    
    # --- Report Body ---
    # Parse each line from the LLM output and apply appropriate PDF styling
    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Convert markdown headings (# ## ###) to PDF section headings
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["SectionHead"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["SectionHead"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], styles["SectionHead"]))
        elif line.startswith("**") and line.endswith("**"):
            # Bold-only lines treated as headings
            story.append(Paragraph(line.strip("*"), styles["SectionHead"]))
        else:
            # Regular paragraph — strip markdown bold markers for clean PDF
            clean = line.replace("**", "")
            story.append(Paragraph(clean, styles["BodyText2"]))
    
    story.append(Spacer(1, 10))
    
    # --- PESTEL Radar Chart (embedded as PNG image in PDF) ---
    if pestel_data and "scores" in pestel_data:
        story.append(Paragraph("PESTEL Analysis", styles["SectionHead"]))
        story.append(Spacer(1, 4))
        
        # Extract PESTEL factor scores and explanations from Gemini's response
        factors = ["Political", "Economic", "Social", "Technological", "Environmental", "Legal"]
        industry_scores = [pestel_data["scores"].get(f, 3) for f in factors]
        explanations = pestel_data.get("explanations", {})
        
        # Generate radar chart with matplotlib
        # Angles are evenly spaced around 360 degrees (one per factor)
        angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False).tolist()
        # Close the polygon by repeating the first data point
        industry_scores_plot = industry_scores + [industry_scores[0]]
        angles_plot = angles + [angles[0]]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles_plot, industry_scores_plot, 'o-', linewidth=2.5, 
                label=industry, color='#2563EB', markersize=7)
        ax.fill(angles_plot, industry_scores_plot, alpha=0.15, color='#2563EB')
        ax.set_xticks(angles)
        ax.set_xticklabels(factors, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=8, color='grey')
        ax.set_rlabel_position(30)
        ax.grid(color='grey', linestyle='-', linewidth=0.3, alpha=0.5)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.tight_layout()
        
        # Save chart to in-memory buffer as PNG, then embed in PDF
        chart_buffer = io.BytesIO()
        fig.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)  # Free memory
        chart_buffer.seek(0)
        
        # Add chart image to PDF document
        from reportlab.platypus import Image
        chart_img = Image(chart_buffer, width=400, height=400)
        story.append(chart_img)
        story.append(Spacer(1, 8))
        
        # Add PESTEL explanations as a formatted table below the chart
        pestel_table_data = [[
            Paragraph("<b>Factor</b>", styles["CellBold"]),
            Paragraph("<b>Score</b>", styles["CellBold"]),
            Paragraph("<b>Explanation</b>", styles["CellBold"]),
        ]]
        for i, factor in enumerate(factors):
            pestel_table_data.append([
                Paragraph(f"<b>{factor}</b>", styles["CellText"]),
                Paragraph(f"{industry_scores[i]}/5", styles["CellText"]),
                Paragraph(explanations.get(factor, ""), styles["CellText"]),
            ])
        
        pestel_table = Table(pestel_table_data, 
                           colWidths=[doc.width * 0.2, doc.width * 0.1, doc.width * 0.7])
        pestel_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1.5, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(pestel_table)
    
    # --- Footer ---
    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Generated by Market Research Assistant | Sources: Wikipedia",
        styles["SmallText"],
    ))
    
    # Build the complete PDF and return the buffer for download
    doc.build(story)
    buffer.seek(0)
    return buffer

# =============================================================================
# STEP 1 — Industry Input & Validation
# =============================================================================
# Task: Collect and validate the industry name from the user.
# Subtasks: Check for empty input, special characters, and minimum length.
# If validation fails, show helpful examples and stop execution.
st.subheader("Step 1: Provide an Industry")
industry = st.text_input(
    "Industry",
    placeholder="e.g. Renewable Energy, Semiconductor, Pharmaceutical",
)

# Button triggers the entire report generation pipeline
run = st.button("Generate Report")

if run:
    # --- Validate API key ---
    # The Gemini API key is required for all LLM calls; stop early if missing
    if not api_key:
        st.error("Please enter your Google Gemini API key in the sidebar.")
        st.stop()

    # --- Validate industry input ---
    # Check 1: Empty input — user clicked Generate without typing anything
    if not industry or industry.strip() == "":
        st.warning("⚠️ No industry provided. Please enter an industry name to continue.")
        st.success("💡 Examples: Renewable Energy, Semiconductor, Pharmaceutical, Automotive, Ecommerce, AI")
        st.stop()
    
    # Remove leading/trailing whitespace for clean processing
    industry = industry.strip()
    
    # Check 2: Special characters — only allow letters, numbers, and spaces
    # This prevents potential issues with Wikipedia search queries
    if not all(c.isalnum() or c.isspace() for c in industry):
        st.warning("⚠️ Please enter an industry name without special characters or symbols.")
        st.success("💡 Examples: Renewable Energy, Semiconductor, Pharmaceutical, Automotive, Ecommerce, AI")
        st.stop()
    
    # Check 3: Minimum length — need at least 2 alphabetic characters
    # Filters out inputs like "1" or single letters that aren't valid industries
    alpha_chars = [c for c in industry if c.isalpha()]
    if len(alpha_chars) < 2:
        st.warning("⚠️ Please enter a valid industry name with at least 2 letters.")
        st.success("💡 Examples: AI, IT, Renewable Energy, Semiconductor, Pharmaceutical")
        st.stop()
    
    # All validation passed — show confirmation message
    st.success(f"✅ Great! Generating report for: **{industry}**")

    # --- Initialise Gemini API client ---
    # Creates a client instance using the user's API key
    client = genai.Client(api_key=api_key)

    # Maximum retry attempts for API calls
    # Free-tier Gemini API has low rate limits (~15 requests/minute)
    # Each retry waits progressively longer: 0s, 60s, 120s, 180s, 240s
    max_retries = 5

    # =================================================================
    # STEP 2 — Retrieve the 5 Most Relevant Wikipedia Pages
    # =================================================================
    # Task: Retrieve the 5 most relevant Wikipedia pages for the industry.
    # 
    # Approach: Content-based relevance scoring
    # Instead of selecting pages by title alone (which can be misleading —
    # e.g., "EA" appears for "Electronics" but is actually a gaming company),
    # we send article CONTENT snippets to Gemini for relevance scoring.
    # Articles scoring below the threshold are rejected, and the system
    # re-searches until 5 relevant articles are found or options are exhausted.
    #
    # Flow:
    #   2A. Search Wikipedia with multiple queries → collect candidates
    #   2B. Filter out Wikipedia meta pages and stubs
    #   2C. Send article titles + content snippets to Gemini → score relevance
    #   2D. Keep articles scoring above threshold, reject the rest
    #   2E. If < 5 found, identify key players and search for more candidates
    #   2F. Score new candidates and repeat until 5 found or exhausted
    st.subheader("Step 2: Retrieving Wikipedia Pages")

    # Relevance score threshold (1-10 scale)
    # Articles scoring below this are rejected as irrelevant
    RELEVANCE_THRESHOLD = 7

    # -----------------------------------------------------------------
    # Helper: Score articles using Gemini (reads content, not just titles)
    # -----------------------------------------------------------------
    def score_articles(candidate_docs, industry, client, model_name, max_retries):
        """
        Send article titles + content snippets to Gemini for relevance scoring.
        Gemini reads the first 300 words of each article to determine if the
        article is genuinely relevant to the industry (not just keyword match).
        
        Returns a list of dicts: [{"index": 0, "score": 9, "reason": "..."}]
        """
        # Build snippets: title + first 300 words of content for each candidate
        snippets = []
        for i, doc in enumerate(candidate_docs):
            title = doc.metadata.get("title", "Unknown")
            # Use first 300 words — enough for Gemini to understand what the article is about
            content_preview = " ".join(doc.page_content.split()[:300])
            snippets.append(f"[{i}] {title}\n{content_preview}\n")
        
        snippets_text = "\n---\n".join(snippets)
        
        score_prompt = (
            f"You are a market research analyst. Score each Wikipedia article below "
            f"on how RELEVANT it is for writing a market research report about "
            f"the **{industry}** industry.\n\n"
            f"IMPORTANT: Read the article content carefully, not just the title. "
            f"A company might have a name that sounds related but actually operates "
            f"in a completely different industry (e.g., 'Electronic Arts' sounds like "
            f"electronics but is actually a gaming company).\n\n"
            f"Score each article 1-10:\n"
            f"- 9-10: Directly about this industry or a major player in it\n"
            f"- 7-8: Relevant technology, market, or significant company\n"
            f"- 4-6: Loosely related but not core to this industry\n"
            f"- 1-3: Unrelated or wrong industry entirely\n\n"
            f"Articles:\n{snippets_text}\n\n"
            f"Return ONLY a JSON array, one object per article:\n"
            f'[{{"index": 0, "score": 9, "reason": "brief reason"}}, ...]\n'
            f"Include ALL articles. No markdown, no extra text."
        )
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(60 * attempt)
                score_response = client.models.generate_content(
                    model=model_name,
                    contents=score_prompt,
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                score_text = score_response.text.strip()
                if score_text.startswith("```"):
                    score_text = score_text.split("```")[1]
                    if score_text.startswith("json"):
                        score_text = score_text[4:]
                return json.loads(score_text)
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                    continue
                else:
                    return None
        return None

    # -----------------------------------------------------------------
    # STEP 2A — Search Wikipedia with multiple query variations
    # -----------------------------------------------------------------
    # Search with multiple query formats to cast a wide net
    with st.spinner("Searching Wikipedia…"):
        search_queries = [
            f"{industry} industry",   # Targets industry overview pages
            f"{industry} market",     # Targets market analysis pages
            f"{industry}",            # General search as fallback
        ]
        
        all_docs = []          # Stores all retrieved Wikipedia documents
        seen_titles = set()    # Tracks page titles to prevent duplicates
        
        retriever = WikipediaRetriever(top_k_results=5, load_max_docs=5)
        
        for query in search_queries:
            try:
                results = retriever.invoke(query)
                for doc in results:
                    title = doc.metadata.get("title", "")
                    if title not in seen_titles:
                        seen_titles.add(title)
                        all_docs.append(doc)
            except Exception:
                continue

    if not all_docs:
        st.error(
            "No Wikipedia pages found for this industry. "
            "Try a different or broader industry name."
        )
        st.stop()

    # -----------------------------------------------------------------
    # STEP 2B — Filter out Wikipedia meta pages and stubs
    # -----------------------------------------------------------------
    excluded_keywords = [
        "disambiguation", "list of", "index of", "category:", 
        "portal:", "template:", "wikipedia:", "file:"
    ]
    
    candidate_docs = []
    for doc in all_docs:
        title = doc.metadata.get("title", "").lower()
        if any(kw in title for kw in excluded_keywords):
            continue
        if len(doc.page_content.split()) < 100:
            continue
        candidate_docs.append(doc)

    if not candidate_docs:
        st.error("No relevant pages found. Please try a different industry.")
        st.stop()

    # -----------------------------------------------------------------
    # STEP 2C — Score candidates by reading article CONTENT (not just titles)
    # -----------------------------------------------------------------
    # Gemini reads first 300 words of each article and scores relevance 1-10.
    # This catches cases where a title sounds relevant but content is not
    # (e.g., "Electronic Arts" for "Electronics" industry).
    with st.spinner("Evaluating article relevance…"):
        scores = score_articles(candidate_docs, industry, client, model_name, max_retries)

    # -----------------------------------------------------------------
    # STEP 2D — Keep articles above threshold, reject the rest
    # -----------------------------------------------------------------
    approved_docs = []
    rejected_titles = []
    
    if scores:
        for item in scores:
            idx = item.get("index", -1)
            score = item.get("score", 0)
            reason = item.get("reason", "")
            if 0 <= idx < len(candidate_docs):
                title = candidate_docs[idx].metadata.get("title", "Unknown")
                if score >= RELEVANCE_THRESHOLD:
                    approved_docs.append(candidate_docs[idx])
                else:
                    rejected_titles.append(f"{title} (score: {score}/10 — {reason})")
    else:
        # If scoring fails, fall back to using all candidates
        approved_docs = candidate_docs

    # Rejected articles are filtered out but not displayed in the UI
    # if rejected_titles:
    #     with st.expander(f"🔍 {len(rejected_titles)} article(s) rejected as irrelevant"):
    #         for r in rejected_titles:
    #             st.caption(f"❌ {r}")

    # -----------------------------------------------------------------
    # STEP 2E — If fewer than 5, search for key players and score them too
    # -----------------------------------------------------------------
    # When initial search doesn't yield enough relevant articles,
    # we ask Gemini to identify key companies from the best overview page,
    # search Wikipedia for those companies, and score the new results.
    if len(approved_docs) < 5:
        # Find the best overview from approved articles (or all candidates as fallback)
        source_docs = approved_docs if approved_docs else candidate_docs
        best_overview = max(source_docs, key=lambda d: len(d.page_content))
        overview_text = best_overview.page_content[:6000]
        
        with st.spinner("Searching for more relevant articles…"):
            # Ask Gemini to identify key players from overview
            identify_prompt = (
                f"Based on the following Wikipedia content about the {industry} industry, "
                f"list the 6-8 most important COMPANIES in this industry globally. "
                f"Focus on the largest, most dominant companies by revenue, market cap, "
                f"or market share. Prioritize industry leaders and major corporations over "
                f"smaller startups or niche players. "
                f"Do NOT include concepts, technologies, datasets, or non-company entities. "
                f"Return ONLY a JSON array of company names, nothing else. "
                f"Example: [\"Google\", \"Microsoft\", \"OpenAI\"]\n\n"
                f"Content:\n{overview_text}"
            )
            
            key_players = []
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        time.sleep(60 * attempt)
                    identify_response = client.models.generate_content(
                        model=model_name,
                        contents=identify_prompt,
                        config=types.GenerateContentConfig(temperature=0.0),
                    )
                    players_text = identify_response.text.strip()
                    if players_text.startswith("```"):
                        players_text = players_text.split("```")[1]
                        if players_text.startswith("json"):
                            players_text = players_text[4:]
                    key_players = json.loads(players_text)
                    break
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                        continue
                    else:
                        key_players = []
                        break
            
            # Search Wikipedia for each key player
            new_candidates = []
            if key_players:
                player_retriever = WikipediaRetriever(top_k_results=1, load_max_docs=1)
                approved_titles = set(d.metadata.get("title", "") for d in approved_docs)
                
                for player in key_players:
                    if player in seen_titles:
                        continue
                    try:
                        player_docs = player_retriever.invoke(player)
                        if player_docs:
                            doc = player_docs[0]
                            title = doc.metadata.get("title", "")
                            if title not in seen_titles and title not in approved_titles:
                                seen_titles.add(title)
                                # Skip stubs
                                if len(doc.page_content.split()) >= 100:
                                    new_candidates.append(doc)
                    except Exception:
                        continue
            
            # Score the new candidates using content-based relevance
            if new_candidates:
                new_scores = score_articles(new_candidates, industry, client, model_name, max_retries)
                if new_scores:
                    for item in new_scores:
                        idx = item.get("index", -1)
                        score = item.get("score", 0)
                        reason = item.get("reason", "")
                        if 0 <= idx < len(new_candidates):
                            title = new_candidates[idx].metadata.get("title", "Unknown")
                            if score >= RELEVANCE_THRESHOLD:
                                approved_docs.append(new_candidates[idx])
                            else:
                                rejected_titles.append(f"{title} (score: {score}/10 — {reason})")

    # -----------------------------------------------------------------
    # STEP 2F — Final selection: take top 5 by relevance, handle shortfall
    # -----------------------------------------------------------------
    # Limit to 5 articles maximum
    docs = approved_docs[:5]

    if len(docs) == 0:
        st.error("No relevant pages found. Please try a different industry.")
        st.stop()
    
    # If fewer than 5 relevant articles found, warn user with suggestion
    if len(docs) < 5:
        st.warning(
            f"⚠️ Only {len(docs)} relevant Wikipedia pages found (target: 5). "
            "Consider adjusting your industry name for better results "
            "(e.g., 'Consumer Electronics' instead of 'Electronics')."
        )

    # Extract URLs and titles, then display as clickable links
    urls = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "Unknown")
        if source:
            urls.append({"title": title, "url": source})

    st.markdown(f"**Top {len(urls)} relevant Wikipedia pages:**")
    for i, item in enumerate(urls, 1):
        st.markdown(f"{i}. [{item['title']}]({item['url']})")

    # =================================================================
    # STEP 3 — Generate Industry Report + PESTEL Analysis
    # =================================================================
    # Task: Generate a concise market research report (< 500 words) with PESTEL visualisation.
    # Subtasks:
    #   3a. Convert selected Wikipedia pages to plain text context
    #   3b. Send context + prompts to Gemini to generate the 4-section report
    #   3c. Generate PESTEL scores via separate Gemini call
    #   3d. Visualise PESTEL scores as radar chart using matplotlib
    st.subheader("Step 3: Industry Report")

    # Convert selected Wikipedia documents to a single text string
    # max_chars_per_doc=5000 balances detail vs LLM token limits
    context = convert_docs_to_text(docs, max_chars_per_doc=5000)

    # Fill in the user prompt template with industry name and Wikipedia text
    formatted_user_prompt = user_prompt.format(industry=industry, context=context)

    # Call Gemini API to generate the market research report
    # Includes exponential backoff retry logic for rate limit handling
    for attempt in range(max_retries):
        try:
            with st.spinner(
                "Generating report…"
                if attempt == 0
                else f"Rate limited. Waiting {60 * attempt}s then retrying ({attempt + 1}/{max_retries})…"
            ):
                if attempt > 0:
                    time.sleep(60 * attempt)
                response = client.models.generate_content(
                    model=model_name,
                    contents=formatted_user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                    ),
                )
            break
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                continue
            else:
                st.error(f"Error: {e}")
                st.stop()

    # Display the generated report
    # Escape "$" signs to prevent Streamlit from rendering them as LaTeX
    # (e.g., "$42.2 billion" would otherwise show as green math text)
    st.markdown(response.text.replace("$", "\\$"))

    # -----------------------------------------------------------------
    # PESTEL Radar Chart
    # -----------------------------------------------------------------
    # PESTEL = Political, Economic, Social, Technological, Environmental, Legal
    # This framework analyses the macro-environment affecting an industry.
    # Each factor is scored 1-5 by Gemini based on the Wikipedia sources.
    # The radar chart provides a visual summary of the industry's environment.
    st.markdown("---")
    # Visualisation of PESTEL factors scored 1-5 for the industry
    st.markdown("### PESTEL Radar Chart")
    
    pestel_data = None

    # Prompt asks Gemini to score each PESTEL factor and explain each score
    # Returns structured JSON that we parse into scores and explanations
    pestel_prompt = f"""You are a senior market research analyst. Based on the Wikipedia sources about the {industry} industry, score each PESTEL factor on a scale of 1-5 based on how FAVORABLE it is for the industry.

Scoring guide:
- 1 = Very unfavorable (major risk/threat)
- 2 = Somewhat unfavorable
- 3 = Neutral / mixed
- 4 = Somewhat favorable
- 5 = Very favorable (strong opportunity/tailwind)

Also provide a brief explanation (max 15 words) for each factor score.

Return your response in this EXACT JSON format (no markdown, no extra text):
{{
  "scores": {{
    "Political": 3,
    "Economic": 4,
    "Social": 3,
    "Technological": 5,
    "Environmental": 2,
    "Legal": 2
  }},
  "explanations": {{
    "Political": "Brief explanation",
    "Economic": "Brief explanation",
    "Social": "Brief explanation",
    "Technological": "Brief explanation",
    "Environmental": "Brief explanation",
    "Legal": "Brief explanation"
  }}
}}

Be specific and analytical. Base scores on evidence from the Wikipedia content."""

    # Call Gemini for PESTEL scoring (same retry logic as report generation)
    pestel_response = None
    for attempt in range(max_retries):
        try:
            with st.spinner(
                "Generating PESTEL radar chart..."
                if attempt == 0
                else f"Rate limited. Waiting {60 * attempt}s then retrying ({attempt + 1}/{max_retries})…"
            ):
                if attempt > 0:
                    time.sleep(60 * attempt)
                pestel_response = client.models.generate_content(
                    model=model_name,
                    contents=pestel_prompt + f"\n\nWikipedia Context:\n{context}",
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                    ),
                )
            break
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                continue
            else:
                st.error(f"Error generating PESTEL analysis: {e}")
                pestel_response = None
                break

    if pestel_response:
        try:
            # Parse JSON response — handle markdown code block wrapping
            pestel_text = pestel_response.text.strip()
            if pestel_text.startswith("```"):
                pestel_text = pestel_text.split("```")[1]
                if pestel_text.startswith("json"):
                    pestel_text = pestel_text[4:]
            pestel_data = json.loads(pestel_text)
            
            # Extract scores for the 6 PESTEL factors
            factors = ["Political", "Economic", "Social", "Technological", "Environmental", "Legal"]
            industry_scores = [pestel_data["scores"].get(f, 3) for f in factors]
            explanations = pestel_data.get("explanations", {})
            
            # --- Draw Radar Chart using matplotlib ---
            # Calculate 6 evenly-spaced angles around a circle (one per factor)
            angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False).tolist()
            
            # Close the polygon by appending the first value at the end
            industry_scores_plot = industry_scores + [industry_scores[0]]
            angles_plot = angles + [angles[0]]
            
            # Create a polar (radar) chart
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
            
            # Plot scores as a filled polygon with markers at each point
            ax.plot(angles_plot, industry_scores_plot, 'o-', linewidth=2.5, 
                    label=industry, color='#2563EB', markersize=7)
            ax.fill(angles_plot, industry_scores_plot, alpha=0.15, color='#2563EB')
            
            # Configure chart labels and grid
            ax.set_xticks(angles)
            ax.set_xticklabels(factors, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 5)          # Score range: 0 to 5
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=9, color='grey')
            ax.set_rlabel_position(30)  # Angle for radial labels
            ax.grid(color='grey', linestyle='-', linewidth=0.3, alpha=0.5)
            ax.spines['polar'].set_color('grey')
            ax.spines['polar'].set_linewidth(0.5)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
            ax.set_title(f"PESTEL Analysis: {industry}", fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            # Render the matplotlib chart in the Streamlit app
            st.pyplot(fig)
            
            # --- PESTEL Explanations Table ---
            # HTML table showing each factor's score and explanation
            st.caption("Scale: 1 = Very unfavorable → 5 = Very favorable")
            
            detail_html = "<table style='width:100%; border-collapse:collapse; margin:10px 0;'>"
            detail_html += (
                "<tr style='background-color:#f0f0f0;'>"
                "<th style='border:1px solid #ccc; padding:8px; text-align:left;'>Factor</th>"
                "<th style='border:1px solid #ccc; padding:8px; text-align:center;'>Score</th>"
                "<th style='border:1px solid #ccc; padding:8px; text-align:left;'>Explanation</th>"
                "</tr>"
            )
            for i, factor in enumerate(factors):
                score = industry_scores[i]
                expl = explanations.get(factor, "")
                detail_html += (
                    f"<tr>"
                    f"<td style='border:1px solid #ccc; padding:8px; font-weight:bold;'>{factor}</td>"
                    f"<td style='border:1px solid #ccc; padding:8px; text-align:center; "
                    f"font-weight:bold;'>{score}/5</td>"
                    f"<td style='border:1px solid #ccc; padding:8px;'>{expl}</td>"
                    f"</tr>"
                )
            detail_html += "</table>"
            st.markdown(detail_html, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating PESTEL analysis: {e}")

    # =================================================================
    # Word Count & PDF Download
    # =================================================================
    st.markdown("---")
    
    # Calculate total word count: report text + PESTEL factor names & explanations
    # This gives a count closer to what Microsoft Word would show
    total_text = response.text
    if pestel_data and "explanations" in pestel_data:
        # Include PESTEL content in the word count since it's part of the report
        for factor, expl in pestel_data["explanations"].items():
            total_text += f" {factor} {expl}"
    
    # Clean markdown formatting before counting words
    # Remove symbols like **, ##, #, --- that aren't actual words
    clean_text = total_text.replace("**", "").replace("##", "").replace("#", "")
    clean_text = clean_text.replace("---", "").replace("- ", "").replace("* ", "")
    clean_text = re.sub(r'\n+', ' ', clean_text)          # Newlines → spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()   # Collapse whitespace
    
    word_count = len(clean_text.split())
    st.caption(f"Word count: {word_count}")
    if word_count >= 500:
        st.warning("⚠️ Report exceeds 500 words. Consider regenerating.")
    
    # --- PDF Download ---
    st.subheader("Download Report")

    # Generate the PDF report in memory using ReportLab
    pdf_buffer = generate_pdf(
        industry=industry,
        report_text=response.text,
        sources=urls,
        pestel_data=pestel_data,
    )

    # Streamlit download button — allows user to save the PDF locally
    st.download_button(
        label="📥 Download Report as PDF",
        data=pdf_buffer,
        file_name=f"{industry.replace(' ', '_')}_Market_Report.pdf",
        mime="application/pdf",
    )
