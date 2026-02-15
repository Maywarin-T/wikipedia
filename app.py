
# A Streamlit app - Industry Research Assistant 

# Task 0: LLM selection dropdown + API key input (sidebar)
# Task 1: Input validation (check if the input is provided and is a valid industry) + LLM token verification
# Task  2: Assistant returns 5 most related Wikipedia  
# Task  3: Industry report generation — LLM summarises 5 URLs of Wikipedia from section 2 into structured report 

# Imports Libraries
# Standard library (built into Python, no install needed)
from langchain_community.retrievers import WikipediaRetriever # Call Wikipedia API 
from langchain_google_genai import ChatGoogleGenerativeAI # Call Google Gemini API 
import streamlit as st # Streamlit for building the interactive UI
import json    # json.loads(text) converts a JSON string into a Python dict/list
import re      # re.findall(pattern, text) finds all regex matches in a string
import time    # time.sleep(seconds) pauses execution (used for API rate-limit retries)


# Set page title name
st.set_page_config(page_title="Industry Research Assistant - Wikipedia")

# Display a heading name at the top of the app
st.title("Industry Research Assistant - Wikipedia")

# Display a description of the app
st.markdown(
    "Hi there! I'm your industry research assistant. "
    "Enter an industry name below and I'll generate a relevant industry report for you."
)


## Task0 : API Key & Model Selection 
# During development, multiple models were tested (Flash, Flash Lite, 2.5 Flash).

# Display a sidebar in the left side of the app
with st.sidebar:
    st.header("Settings")  # Display a heading name in the sidebar
    # Display a text box in the sidebar
    api_key = st.text_input("Google Gemini API Key", type="password")
    # Display a dropdown menu in the sidebar
    model_name = st.selectbox("Model", ["gemini-2.0-flash"])


# Helper Functions
# Helper1 - Send a prompt to Gemini and retries automatically if rate-limited.
def call_gemini(llm, prompt, system_instruction=None):
    max_retries = 5  # try up to 5 times before giving up
    if system_instruction:
        messages = [("system", system_instruction), ("user", prompt)]
    else:
        messages = [("user", prompt)]

    # The loop calls the Gemini API up to 5 times. Each time it either gets a response and returns the cleaned text, or an error occurs.
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait = 60 * attempt  
                time.sleep(wait)     
            response = llm.invoke(messages)

            # response.content is the LLM's text output
            # .strip() removes whitespace from the start and end of the string
            return response.content.strip()

        except Exception as e:  
            error_msg = str(e)  
            # If rate limit exceeded, it continues the loop
            if "RESOURCE_EXHAUSTED" in error_msg and attempt < max_retries - 1:
                continue  
            # If API key invalid/permission denied, it stops and returns None
            elif "API_KEY_INVALID" in error_msg or "PERMISSION_DENIED" in error_msg:
                return None  
            # If any other error occurs, it stops and returns None.
            else:
                return None

    return None  #If all retries are exhausted, it returns None


# Helper2 - Take AI text, clean it, and turn it into a dict or list, and returns None if parsing fails
def parse_json(text):
    if not text:  
        return None
    text = text.strip()  
    if text.startswith("```"): 
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]  
    # If the text isn't valid JSON, return None instead of crashing
    try:  
        return json.loads(text.strip())   
    except json.JSONDecodeError:  
        return None

# Helper3 - Convert Wikipedia documents into a plain text string for the LLM
def docs_to_text(docs, max_chars=5000):
    parts = []  # empty list to collect formatted text for each page
    for doc in docs:  # iterate over each Wikipedia document
        title = doc.metadata.get("title", "Unknown")   # safely gets the "title" key from the dict
        url = doc.metadata.get("source", "") #safely gets the "source" key from the dict
        content = doc.page_content[:max_chars] #uses slicing to get the first max_chars characters
        parts.append(f"### {title}\nSource: {url}\n\n{content}") #f-string: variables inside {} are replaced with their values
    return "\n\n".join(parts) #combines all strings in the list, separated by two newlines

# Task 1: Input validation (check if the input is provided and is a valid industry) and LLM token verification
# Sub-task 1.1: Input validation
# If the input isn't a valid industry, we suggest alternatives (3-5 options)
if st.session_state.get("_picked_suggestion"):
    st.session_state.industry_input = st.session_state._picked_suggestion
    st.session_state._picked_suggestion = None   # clear the flag
    st.session_state.suggestions = []            # hide the suggestion buttons

industry = st.text_input(
    "Industry",
    key="industry_input", # links this widget to st.session_state.industry_input so Streamlit remembers what the user typed across reruns.
    placeholder="e.g. FMCG, Automotive, Technology", # shows grey hint text when the box is empty
)

run = st.button("Generate Report") #creates a clickable button that user can click to trigger the report generation pipeline

if st.session_state.get("suggestions"):
    st.info("Did you mean one of these?")  # shows a blue info box that displays the suggestions
    for suggestion in st.session_state.suggestions:
        if st.button(f"✅ {suggestion}", key=f"suggest_{suggestion}"): #creates a clickable button for each suggestion.
            st.session_state._picked_suggestion = suggestion
            st.rerun()  # forces Streamlit to re-execute the script immediately

if run: # Everything below only runs when the user clicks "Generate Report"
    if st.session_state.get("suggestions"): # Clear old suggestions before starting the pipeline
        st.session_state.suggestions = []
        st.rerun()
    if not api_key: # if the API key is not provided, show an error message and stop the execution
        st.error("Please enter your Google Gemini API key in the sidebar.")
        st.stop() 
    if not industry or industry.strip() == "": # if the industry is empty or just whitespace, show a warning message and stop the execution
        st.warning("Please enter an industry name.")
        st.info("Examples: FMCG, Automotive, Technology, Retail, Banking, Insurance, Healthcare, Education, etc.")
        st.stop()
    industry = industry.strip() # remove leading/trailing whitespace from the string
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -()&/,.") # set() creates an unordered collection of unique characters. This defines which characters are allowed in industry names.
    if not all(c in allowed for c in industry): # if the industry contains any special characters, show a warning message and stop the execution
        st.warning("Please enter an industry name without special characters like @, #, $, !, etc.")
        st.info("Examples: FMCG, Automotive, Technology")
        st.stop()
    alpha_count = sum(1 for c in industry if c.isalpha()) # count how many letters are in the string
    if alpha_count < 2: # if the industry contains less than 2 letters, show a warning message and stop the execution
        st.warning("Please enter a valid industry name with at least 2 letters.")
        st.info("Examples: AI, IT, Renewable Energy, Semiconductor")
        st.stop()



# Sub-task 1.2: LLM token verification
  # ChatGoogleGenerativeAI creates a connection to the Gemini LLM.
    max_output_tokens = 1024 #max_output_tokens limits how long the LLM's response can be (1024 ≈ 750 words) in tokens
    llm = ChatGoogleGenerativeAI(
        model=model_name,             # selected model - gemini-2.0-flash
        temperature=0,                # temperature=0 makes responses deterministic (same input → same output)
        max_output_tokens=max_output_tokens,
        google_api_key=api_key,       # the API key from the sidebar
    )

    
    st.success(f"✅ Generating report for: **{industry}**") # shows a green success box with the message
# Sub-task 1.3: LLM industry validation
    validate_prompt = (
        "You are an expert at classifying business terms. Decide if the user input is an INDUSTRY/SECTOR "
        '(a broad business category, e.g. "Automotive", "Pharmaceutical", "Renewable Energy") or NOT.\n\n'
        "NOT an industry: company names (e.g. Tesla, Apple), people, product names, misspellings, or vague terms.\n\n"
        f'User input: "{industry}"\n\n'
        "Rules:\n"
        "- If it IS a recognized industry/sector: reply with JSON only:\n"
        '  {"is_industry": true, "key_players": ["CompanyA", "CompanyB"]}\n'
        "  Use exactly 2 real, globally dominant companies in that exact industry (well-known names only).\n"
        "- If it is NOT an industry: reply with JSON only:\n"
        '  {"is_industry": false, "suggestions": ["Industry1", "Industry2", "Industry3"]}\n'
        "  Suggest 3–5 real industry/sector names that are related to the input (e.g. if they typed a company, suggest that company's industry).\n\n"
        "Output ONLY valid JSON, no markdown or extra text."
    )
    validate_text = call_gemini(llm, validate_prompt)      # get raw text from LLM
    validate_result = parse_json(validate_text)             # parse it into a Python dict

    # Handle the three possible outcomes:
    if validate_text is None:
        # LLM call completely failed — likely bad API key
        st.error("Could not connect to Gemini. Please check your API key")
        st.stop()

    if validate_result is None:
        # LLM returned text but it wasn't valid JSON — proceed without key_players
        key_players = []

    # .get("is_industry", True) safely gets the value, defaulting to True if key missing
    elif not validate_result.get("is_industry", True):
        # Input is NOT a valid industry — show suggestion buttons
        suggestions = validate_result.get("suggestions", [])
        # \" inside a string is an escaped quote (lets us include quotes in the message)
        st.warning(f'"{industry}" doesn\'t look like a recognized industry name.')
        if suggestions:
            # Save suggestions to session state so they persist after rerun
            st.session_state.suggestions = suggestions
            st.rerun()  # rerun to show the suggestion buttons at the top
        else:
            st.info("Try something like: Renewable Energy, Semiconductor, Pharmaceutical")
        st.stop()
    else:
        # Valid industry — extract the list of key player names
        key_players = validate_result.get("key_players", [])
        # Key players are used internally for Wikipedia search, no need to show them.



    ## Task 2: Wikipedia retrieval
    # Sub-task 2.1: Two-pronged Wikipedia search
    st.subheader("Step 2: Retrieving Wikipedia Pages")
    progress = st.progress(0, text="Searching Wikipedia...") # creates a progress bar. 0 = empty, 100 = full
    retriever = WikipediaRetriever(top_k_results=8, load_max_docs=8) # fetches pages from Wikipedia's API
    all_docs = []           # list to collect all Wikipedia documents
    seen_titles = set()     # set to track which pages we've already seen (avoids duplicates)
    # Search with multiple query variations for broader coverage
    for query in [f"{industry} industry", f"{industry} market", f"{industry} sector", industry]:
        try:
            results = retriever.invoke(query) # calls the Wikipedia API and returns Document objects
            for doc in results:
                title = doc.metadata.get("title", "")
                # Only add if we haven't seen this title before
                if title not in seen_titles:
                    seen_titles.add(title)  # add to the set of seen titles
                    all_docs.append(doc)    # add the document to our collection
        except Exception:
            # If a search query fails (e.g. network error), skip it and try the next
            continue

    # Search for each key player (1 result each to avoid flooding with company pages)
    player_retriever = WikipediaRetriever(top_k_results=1, load_max_docs=1)
    for player in key_players:
        try:
            results = player_retriever.invoke(player)
            for doc in results:
                title = doc.metadata.get("title", "")
                if title not in seen_titles: # Only add if we haven't seen this title before
                    seen_titles.add(title) # add to the set of seen titles
                    all_docs.append(doc) # add the document to our collection
        except Exception:
            continue

    progress.progress(40, text="Filtering candidates...") # update the progress bar to 40%

    if not all_docs:
        st.error(f"No Wikipedia pages found for '{industry}'. Try a broader or different industry name.")
        st.stop()

   # Sub-task 2.2: Rule-based junk filtering
    # Remove pages that are obviously useless for an industry research report in order to be faster and cheaper than asking the LLM to evaluate every page
    junk_keywords = ["disambiguation", "list of", "index of", "category:", "portal:", "template:"]
    candidates = []  # pages that pass the filter

    for doc in all_docs:
        title = doc.metadata.get("title", "").lower() # converts string to lowercase for case-insensitive comparison
        if any(kw in title for kw in junk_keywords): # skip if any junk keyword appears in the title
            continue
        title_original = doc.metadata.get("title", "") # get the original title
        # Skip country/region-specific articles so we keep a global perspective
        country_phrases = [" in the united states", " in united states", " in the uk", " in the united kingdom",
                           " in japan", " in china", " in germany", " in france", " in india", " in brazil",
                           " in australia", " in canada", " in italy", " in spain", " in south korea", " in mexico"]
        if any(phrase in title for phrase in country_phrases):
            continue
        if " in " in title_original:
            after_in = title_original.split(" in ", 1)[-1].strip()
            if after_in and after_in[0].isupper() and len(after_in.split()) <= 4:
                continue
        if len(doc.page_content.split()) < 100: # skip if the page is too short
            continue

        candidates.append(doc)  # add the document to the list of candidates

    if not candidates:
        st.error(f"No useful Wikipedia pages found for '{industry}'. Try a different industry name.")
        st.stop()

    progress.progress(60, text="Checking relevance...") # update the progress bar to 60%

    # Sub-task 2.3: LLM content-based relevance selection
    # Instead of just matching titles, we send Gemini the first 300 words of each candidate article so it can verify the page is genuinely about our industry (not just a keyword match).
    # For example, "Digital twin" might appear in a Healthcare search but its content is mainly about manufacturing
   
    snippets = []  # list of text previews for each candidate page
    # enumerate() gives both the index (i) and the item (doc) in each iteration
    for i, doc in enumerate(candidates):
        title = doc.metadata.get("title", "Unknown")
        preview = " ".join(doc.page_content.split()[:300]) # split into words, then take the first 300 and combine them back into a single string with spaces
        snippets.append(f"[{i}] {title}\n{preview}") # add the title and preview to the list of snippets

    # Build the selection prompt with strict relevance criteria
    select_prompt = (
        f'I need you to choose which of these Wikipedia articles are actually useful for a global report on the {industry} industry. '
        f'Read each one and think about whether it really fits.\n\n'
        f'The best fits are: the main overview of the {industry} industry (global picture, market size, how the industry works), '
        f'then big sectors or sub-industries that are clearly part of {industry}, '
        f'then core products or practices that define the industry. '
        f'Articles with real numbers (market size, revenue, growth) are especially helpful.\n\n'
        f'Skip anything that is only about one country or region, about a single local market, one city, or a specific geographic marketplace. '
        f'Also skip articles that barely mention {industry}, or are about a different industry, or are too generic to be useful.\n\n'
        f'Below are the articles. Give me a JSON array of the indices of the ones you would keep, in order from most to least relevant. '
        f'Example: [1, 0, 2, 4]. You MUST return exactly 5 indices. Include closely related sub-industries or key sectors if the direct matches are fewer than 5.\n\n'
        f'Articles:\n\n{"\n\n---\n\n".join(snippets)}\n\n'
        f'Return ONLY the JSON array of indices, nothing else.'
    )

    select_text = call_gemini(llm, select_prompt)
    selected_indices = parse_json(select_text)  # parse the text into a Python list

    # Use Gemini's selection if valid, otherwise fall back to first 5 candidates
    docs = []  # final list of selected documents
    if selected_indices and isinstance(selected_indices, list): # check if the selected indices is a list
        for idx in selected_indices[:5]: # take at most the first 5 indices
            if isinstance(idx, int) and 0 <= idx < len(candidates): # check if the index is an integer and within the valid range
                docs.append(candidates[idx]) # add the document to the list of selected documents

    # Fallback: if Gemini returned malformed response or bad indices
    if not docs:
        docs = candidates[:5]  # just use the first 5 candidates

    # If LLM returned fewer than 5, fill remaining slots from unused candidates
    if len(docs) < 5:
        selected_titles = {doc.metadata.get("title", "") for doc in docs}
        for doc in candidates:
            if len(docs) >= 5:
                break
            if doc.metadata.get("title", "") not in selected_titles:
                docs.append(doc)
                selected_titles.add(doc.metadata.get("title", ""))

    progress.progress(100, text="Done!")

    # If fewer than 5 relevant pages were found, ask the user to try a different industry name
    if len(docs) < 5:
        st.info(
            f"It looks like **{industry}** might not have enough Wikipedia coverage "
            f"to provide 5 relevant pages as expected. Could you try a broader or "
            f"slightly different industry name so I can find better sources for you?"
        )
        st.stop()

    # Build a list of source dicts for the "Top N relevant Wikipedia pages" links
    sources = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        url = doc.metadata.get("source", "")
        if url:  # only add if the URL exists
            sources.append({"title": title, "url": url})  # dict with two keys

    # Display the selected pages as clickable markdown links
    st.markdown(f"**Top {len(sources)} relevant Wikipedia pages:**") # display the number of relevant Wikipedia pages
    for i, src in enumerate(sources, 1): # enumerate() gives both the index (i) and the item (src) in each iteration
        st.markdown(f"{i}. [{src['title']}]({src['url']})") # display the title and URL of the relevant Wikipedia page


    ## Task 3: Report generation
    # Sub-task 3.1: Generate the industry research report
    st.subheader("Step 3: Industry Report")
    context = docs_to_text(docs, max_chars=5000) # convert the Wikipedia pages to plain text for the LLM

    recent_years = re.findall(r'\b(202[0-9])\b', context) # find all the years in the context
    if not recent_years:  # check if there are no recent years found
        st.warning(
            f"The Wikipedia sources for this industry do not appear to contain recent data (2020 or later). The report may rely on qualitative insights rather than up-to-date statistics."
        )

    # System prompt for the industry research report
    report_system = (
        "You are a senior industry research analyst writing a GLOBAL industry report. "
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
    
    # User prompt for the industry research report
    report_prompt = (
        f"Write an industry research report on the **{industry}** industry.\n\n"
        f"Use ONLY the facts from these Wikipedia sources:\n\n{context}\n\n"
        f"Write the report as clean markdown with ## headings.\n"
        f"REMEMBER: 400 to 440 words. No less, no more."
    )

    report = call_gemini(llm, report_prompt, system_instruction=report_system)  # generate the industry research report
    
    # System prompt for the SWOT analysis
    swot_prompt = (
        f"You are a senior industry research analyst. Based on the Wikipedia sources "
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

    swot_text = call_gemini(llm, swot_prompt)  # generate the SWOT analysis
    swot_data = parse_json(swot_text)  # parse the text into a Python dict

    # Save the results to the session state so they persist across UI interactions
    st.session_state.report_data = {
        "industry": industry,
        "report": report,
        "swot": swot_data,
    }

    # Sub-task 3.2: Display the report
    st.markdown(report.replace("$", "\\$")) # replace the dollar signs with the LaTeX math expression

    # Sub-task 3.3: Display the SWOT analysis as table  
    if swot_data:
        st.markdown("### SWOT Analysis")
        swot_factors = ["Strengths", "Weaknesses", "Opportunities", "Threats"] # list of the SWOT factors
        rows = ""  # HTML string to accumulate table rows
        for factor in swot_factors:
            points = swot_data.get(factor, []) # get the points for the factor
            bullets = "".join(f"<div style='margin:2px 0;'>• {p}</div>" for p in points) # join the bullet points with a <div> element
            rows += (
                f"<tr>"
                f"<td style='padding:10px; font-weight:bold; vertical-align:top; "
                f"border-bottom:1px solid #ddd; width:25%;'>{factor}</td>"
                f"<td style='padding:10px; vertical-align:top; "
                f"border-bottom:1px solid #ddd;'>{bullets}</td>"
                f"</tr>"
            )

        # Build the complete HTML table with header row and data rows
        swot_html = (
            "<table style='width:100%; border-collapse:collapse; "
            "border:1px solid #ddd; margin-top:8px;'>"
            "<tr style='background-color:#f0f0f0;'>"
            "<th style='padding:10px; text-align:left; border-bottom:2px solid #999;'>Factor</th>"
            "<th style='padding:10px; text-align:left; border-bottom:2px solid #999;'>Details</th>"
            "</tr>"
            f"{rows}</table>"
        )
        st.markdown(swot_html, unsafe_allow_html=True)

    # Sub-task 3.4: Word count
    # Must stay under 500 words
    swot_text = ""
    if swot_data and isinstance(swot_data, dict):
        for points in swot_data.values():
            if isinstance(points, list):
                swot_text += " " + " ".join(str(p) for p in points)
            else:
                swot_text += " " + str(points)
    total_text = report + swot_text
    clean = re.sub(r'[#*\-\n]+', ' ', total_text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    word_count = len(clean.split())

    st.markdown("---")  # display a horizontal rule divider line
    st.caption(f"Word count (report + SWOT): {word_count}")
    if word_count >= 500:
        st.warning("Total exceeds 500 words. Consider regenerating.")

