# IMF Report LLM Extraction

Extracts structured data from IMF staff-report PDFs (Article IV and program reviews) and
turns it into Excel datasets.

| File | Purpose |
|---|---|
| `imf_pdf_extraction_engine.py` | Extraction engine. Reads Box/Annex titles from the PDF table of contents, finds the Structural Benchmark (SB) table pages, and uses the Claude API to read those tables and to pick Selected Issues chapter titles. |
| `01_extract_pdfs_to_json.ipynb` | Runs the engine over every PDF in a country folder and saves the results as JSON. |
| `02_transform_json_to_excel.ipynb` | Converts the JSON into per-country and aggregate Excel workbooks. |

## Claude API key (required)

The engine sends rendered PDF pages and table-of-contents text to Claude, so it needs an
Anthropic API key. Without one, Structural Benchmark and Selected Issues extraction
returns nothing.

1. Create an account and generate a key in the [Anthropic Console](https://console.anthropic.com/) (Settings → API keys). Keys start with `sk-ant-`.
2. Copy `.env.example` to `.env` in the same folder as the notebooks.
3. Paste your key into `.env`:

```bash
# .env  (never commit this file)
ANTHROPIC_API_KEY=sk-ant-your-key-here
TOC_SCRIPT_DIR=/path/to/folder/containing/imf_pdf_extraction_engine.py
BASE_FOLDER=/path/to/folder/with/one/subfolder/per/country
EXCLUDE_LIST_PATH=/path/to/box_annex_exclude_list.xlsx
```

| Variable | Used by | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | notebook 01 | Your Claude API key |
| `TOC_SCRIPT_DIR` | notebook 01 | Folder containing `imf_pdf_extraction_engine.py` |
| `BASE_FOLDER` | notebooks 01 and 02 | Folder with one sub-folder of PDFs per country (e.g. `CPV/`, `BEN/`) |
| `EXCLUDE_LIST_PATH` | notebook 02 | Excel list of Box/Annex titles to exclude |

### Keep your key private

- `.env` must **not** be committed. Add it to `.gitignore`:
  ```bash
  echo ".env" >> .gitignore
  ```
- Clear notebook outputs before committing (the first notebook prints the start of the key when it loads).
- If a key is ever exposed, delete it in the Anthropic Console and create a new one.
- API calls are billed to your account. Each Structural Benchmark page is one image request, so try one country first.

### Google Colab

Colab doesn't read a local `.env`. Store the key under **Secrets** (key icon in the left sidebar) as `ANTHROPIC_API_KEY`, turn on **Notebook access**, and read it with:

```python
from google.colab import userdata
ANTHROPIC_API_KEY = userdata.get("ANTHROPIC_API_KEY")
```

## Setup

```bash
pip install pymupdf requests openpyxl pandas tqdm python-dotenv
```

## Usage

1. Put each country's PDFs in `BASE_FOLDER/<COUNTRY>/` (e.g. `BASE_FOLDER/CPV/`).
2. Open `01_extract_pdfs_to_json.ipynb`, set `COUNTRY`, `WORKERS` and `SI_ONLY` in the run cell, and run all cells. A `toc_results_<timestamp>.json` file is saved in each country folder.
3. Open `02_transform_json_to_excel.ipynb` and run all cells. It writes `IMF_<COUNTRY>.xlsx` in each country folder and `IMF_All_Countries.xlsx` in `BASE_FOLDER/_Aggregate/`.

## What gets extracted

- **Document date and report number** from the first pages.
- **Box and Annex titles** from the PDF outline (or a text scan of the table of contents).
- **Structural Benchmarks:** reform area, benchmark text, due date and status. Prior Actions are excluded.
- **Selected Issues chapter titles** for SI papers.
