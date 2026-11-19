# TruthCheck AI v2 — Local AI edition

This version runs an open-weight language model on the user's own Windows PC through Ollama. It does **not** use a paid AI API and does not ask the user to copy prompts into a separate AI website.

## What it does

1. Accepts a news URL or pasted article text.
2. Extracts the article text (some paywalled or protected pages may fail).
3. Sends the article to a local Ollama model to extract factual claims and search phrases.
4. Searches recent related news through the public GDELT DOC 2.0 endpoint (no API key).
5. Fetches accessible source articles and asks the local model to compare the claims with the retrieved material.
6. Displays claim-level assessments, source links, reliability and validity/balance rubrics, tone/bias signals, and corporate/audience impact notes.
7. Exports a Markdown report.

## Requirements

- Windows 10 version 22H2 or later (Ollama's current Windows requirement).
- A PC with enough RAM and free storage for a local model. The recommended `qwen3:4b` model download is about 2.6 GB; it needs additional memory while running. If your PC is slow or runs out of memory, use a smaller model such as `qwen3:1.7b` (analysis quality may be lower).
- Internet connection for article extraction and GDELT news search.
- Python 3.10 or later.

## Setup (Windows PC)

1. Install Ollama: https://ollama.com/download/windows
2. Open PowerShell or Command Prompt and download the model:

   ```powershell
   ollama pull qwen3:4b
   ```

3. Download/unzip this project to a folder.
4. Open PowerShell in that folder and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

5. Start the app:

   ```powershell
   streamlit run app.py
   ```

6. Open the local address shown by Streamlit, usually `http://localhost:8501`.

Ollama runs locally at `http://127.0.0.1:11434`. Keep Ollama running while TruthCheck is in use.

## Open the app on an iPad (same Wi-Fi only)

The AI model remains on the Windows PC; the iPad only displays the Streamlit page.

1. Keep the PC and iPad on the same trusted Wi-Fi network.
2. In PowerShell, start Streamlit with:

   ```powershell
   streamlit run app.py --server.address 0.0.0.0
   ```

3. Find the PC's local IPv4 address using `ipconfig` (for example `192.168.0.25`).
4. On the iPad, open `http://192.168.0.25:8501` using the actual PC address.
5. Keep the PC and its Ollama service running. Do not expose the port to the public internet or set up router port forwarding.

If Windows Firewall asks whether Python may communicate on a private network, only allow it on a trusted private network and follow the device owner's guidance.

## Important limits

- This version runs locally; it cannot run the local model inside Streamlit Community Cloud. The previous `ManualAI` version is not the intended deployment for this design.
- GDELT DOC 2.0 searches recent news coverage (the query here uses a rolling 3-month window). Older historical claims may not yield sources.
- GDELT provides news discovery, not a complete database of official filings, court decisions, or scientific papers. A retrieved news article is not automatically a primary source.
- A different domain does not prove independent reporting. Syndicated or copied stories may share the same underlying source.
- If evidence cannot be retrieved, the system must show insufficient evidence or evaluation hold rather than invent support.
- Local language models can hallucinate or misread sources. Verify important claims against original documents and official disclosures.
- Scores are rubric-based review indices, not probabilities that an article is true and not investment advice.

## GDELT attribution

News search uses the GDELT Project DOC 2.0. Please cite and link to https://www.gdeltproject.org/ when describing or redistributing GDELT-derived data.
