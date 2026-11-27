# JobAds Pipeline: Job Advertisements in Historical Newspapers (1850–1950)

Processing pipeline of the **JobAds project** (FWF P35783, University of Graz), which studies the emergence and transformation of the Austrian labour market through job advertisements in digitized German-language newspapers from 1850 to 1950.

The pipeline takes newspaper page images from the [ANNO corpus](https://anno.onb.ac.at/) of the Austrian National Library (ÖNB) and turns them into a structured dataset of job advertisements: one row per text region, with corrected text, ad type, extracted entities (position titles, salaries, requirements, …), a gender label for each position and a IIIF link to the image snippet.

```
IIIF manifests ─► page classification ─► layout analysis ─► OCR ─► region classification
   (ANNO/ÖNB)      (pages with job ads)     (Eynollah)     (Tesseract - frak2021)  (job ad / not job ad)
                                                                              │
   ┌──────────────────────────────────────────────────────────────────────────┘
   ▼
OCR post-correction ─► ad type classification ─► hyphen & listing normalisation
     (ByT5)               (ELECTRA)                       (rule-based)
                                                               │
   ┌───────────────────────────────────────────────────────────┘
   ▼
information extraction (spaCy NER) ─► span mapping ─► gender labels + IIIF links ─► <tag>_extractions.csv
```

## Project context

- **Corpus:** 29 newspaper titles from ANNO, selected as the titles with the most editions containing job ads, 1850–1950.
- **Ground truth:** ~15,000 manually annotated job ads (14,985), used to train and evaluate the models below.
- **Language:** predominantly German, printed largely in Fraktur (long s `ſ`, `⸗` hyphen, etc.).
- **Team (alphabetic order):** Raven Adam, Saranya Balasubramanian, Manfred Füllsack, Jörn Kleinert (PI), Wiltrud Mölzer, Johanna Ofenauer, Klara Venglarova, Georg Vogeler.
- **Project website:** <https://historical-job-ads.uni-graz.at/en/>

Newspapers are identified by their three-letter ANNO code (e.g. `nwb`, `nfp`, `wrz`, `dvb`, `ptb`, `nwj`). In the notebook this code is called `tag` or `paper_name`, and every step reads and writes files named after it.

## Pipeline steps

### 1. Candidate pages from IIIF manifests

Input: `data_dh-infra.csv` with columns `anno_id` and `iiif_manifest` (one IIIF Presentation v3 manifest per issue).
The notebook reads each manifest, takes the issue date from the `Jahr/Datierung` metadata field, and collects all page image URLs (`painting` annotations) with IDs of the form `<code><yyyymmdd>_<page>`.

### 2. Page classification (does the page contain job ads?)

A fine-tuned BEiT/DiT image classifier ([`Var3n/DiT_classification_jobads_finetuned`](https://huggingface.co/Var3n/DiT_classification_jobads_finetuned)) predicts for each page whether it contains job ads. Images are downloaded in batches of 25, with a retry on rate limits.

Output: `all_ads.txt` (`image_id url prediction probability` per line), copied to `Backup/` every 100 batches.

### 3. Layout analysis (Eynollah)

[Eynollah](https://github.com/qurator-spk/eynollah) segments the selected pages into text regions and reading order and writes PAGE-XML. It runs in its own conda environment (`eyn_env`) via `layout_analysis.sh`, one job per `<code>_<year>.txt` file list. The notebook starts the jobs in parallel threads and pins each to the least busy GPU via `CUDA_VISIBLE_DEVICES` (`N_GPUS`, `JOBS_PER_GPU`); the output of each job goes to `logs/<code>_<year>.log`.

`layout_analysis.sh <file_list>` takes a list with one image path per line (e.g. `ptb/1889/ptb_18891212/ptb_18891212_027.jpg`) and:

- runs `eynollah -m $MODEL_DIR layout -i <batch> -o <output_dir> -cl -fl` in mini-batches of `BATCH_SIZE` images;
- writes results to `$OUTPUT_ROOT/<code>/<year>/`, with code and year taken from the path of the first image;
- skips images whose `.xml` already exists (`SKIP_DONE=1`), so interrupted runs can be resumed;
- keeps the failing batch file and stops if Eynollah crashes.

| Variable | Meaning | Default |
|---|---|---|
| `CONDA_PREFIX` | Prefix of the Eynollah conda environment | required |
| `OUTPUT_ROOT` | Root directory for the PAGE-XML output | required |
| `IMAGE_ROOT` | Directory that relative paths in the list are resolved against | current directory |
| `MODEL_DIR` | Eynollah model directory | `model_dir` |
| `BATCH_SIZE` | Images per Eynollah call | `200` |
| `SKIP_DONE` | `1` = skip images that already have an XML | `1` |

The script puts `$CONDA_PREFIX/lib` on `LD_LIBRARY_PATH` (TensorFlow 2.12 needs CUDA 11.8 / cuDNN 8 from the conda environment) and sets `TF_FORCE_GPU_ALLOW_GROWTH=true`, so that several processes can share one GPU.

Eynollah also writes enhanced images (`*_enhanced.png`). Two follow-up cells:
- rename the enhanced PNGs (remove `_enhanced`) on the Nextcloud storage via WebDAV `MOVE`;
- update `imageFilename` in each PAGE-XML to point to the PNG.

XML files that are corrupted or missing are collected, the matching original JPGs are written to `<code>_to_regenerate.txt` / `<code>_files_redo.txt`, and the broken XML/PNG files are deleted so that the pages can be processed again.

### 4. OCR (Tesseract)

`ocr_pipeline.run_pipeline` OCRs every text line of the PAGE-XML files with Tesseract 5 and the Fraktur model `frak2021-0.905`, writing the text back into PAGE-XML under `<code>_ocr/`. Already processed files are skipped (`gather_processed_stems`), so the step can be resumed. Parallelism: `ocr_workers`, `io_workers`, `max_in_flight`.

### 5. Job ad region identification

Each text region of each page is extracted in reading order (`extract_regions_in_reading_order`) with its text and bounding box (`min_x`, `max_x`, `min_y`, `max_y`). The [JobAds Identification Model](https://huggingface.co/Var3n/jobads-identification) (ELECTRA, fine-tuned from `stefan-it/zeitungs-lm-v1`) labels each region as job ad (`1`) or other (`0`) on 2 GPUs. The model is applied to the **raw** OCR text, as it was trained on uncorrected `frak2021` output.

Output: `finished_csv/<code>_classified.csv` with the job ad regions only.

### 6. OCR post-correction (ByT5)

1. Character clean-up of the raw OCR (ligatures `ﬁ ﬂ ﬀ ﬄ`, `¬`/`⸗` → `-`, combining characters, superscript-e umlauts, …).
2. Regions are split into chunks of at most 512 UTF-8 bytes (`split_for_byt5`), preferring paragraph, sentence, clause and word boundaries.
3. A ByT5-large model fine-tuned for OCR post-correction ([`Var3n/byt5-large-ocr-postcorrection`](https://huggingface.co/Var3n/byt5-large-ocr-postcorrection), beam search with 4 beams, `max_length=576`) corrects each chunk.
4. Chunks are joined back per region.

Output: `finished_csv/<code>_postcorrected.csv` with the new column `post_corrected`.

> Note: the post-corrected text keeps historical spelling, line breaks and long s (`ſ`); normalisation follows in step 8.

### 7. Ad type classification (ELECTRA)

The [JobAds Classification Model](https://huggingface.co/Var3n/jobads-classification) (ELECTRA, fine-tuned from `stefan-it/zeitungs-lm-v1`; local directory `electra_finetuned/` with `label_encoder.pkl`) assigns each job ad region to one of five categories:

| Label | Meaning |
|---|---|
| `heading` | Heading above or within the job ad sections, e.g. *Offene Stellen* |
| `job_offer` | An open position is advertised |
| `job_search` | Someone is looking for a position |
| `service_offer` | A service is offered (e.g. language lessons) without a full employment relationship |
| `vermittlung` | Mediation: positions are offered or sought through an intermediary, e.g. an agency |

The model was trained on manually corrected text with line breaks removed.

Output: `<code>_categories.csv` with the column `prediction_label`.

### 8. Hyphen and listing normalisation (`listing_hyphen.py`)

Produces `post_corrected_final` from `post_corrected`:

- line breaks → spaces, long s `ſ` → `s`;
- spaces around connectors glued to words (`SchlosserundSchmied` → `Schlosser und Schmied`);
- removal of line-break hyphenation (`wöchent⸗ lich` → `wöchentlich`), keeping hyphens before capitals and connectors;
- **expansion of elliptical listings** using a curated list of occupation suffixes, e.g.
  `Stuben⸗ und Kindermädchen` → `Stubenmädchen und Kindermädchen`,
  `Bureau⸗ oder Geschäftsdienerposten` → `Bureaudienerposten oder Geschäftsdienerposten`;
- removal of spaces inside hyphenated compounds.

```python
import listing_hyphen as lh
df = lh.process_dataframe(df, n_jobs=-1)   # parallel, with a plain-text progress bar
```

### 9. Information extraction (spaCy NER)

Three spaCy NER models (transformer based on `zeitungs-lm-v1` + NER). `ner_worker.MODELS` maps each to a local directory:

| Key | Local directory | Hugging Face | Entity labels (= output columns) |
|---|---|---|---|
| `salary` | `salary_model` | [`Var3n/jobads-salary`](https://huggingface.co/Var3n/jobads-salary) | `salary`, `salary_importance`, `salary_period`, `unspecific_salary`, `verpflegung` (board and lodging) |
| `positions` | `positions_model` | [`Var3n/jobads-position`](https://huggingface.co/Var3n/jobads-position) | `position` |
| `requirements` | `requirements_model` | [`Var3n/jobads-requirements-activity`](https://huggingface.co/Var3n/jobads-requirements-activity) | `activity`, `attitude towards work`, `background`, `interpersonal`, `job specific`, `language` |

Download them into the expected directories, e.g.:

```python
from huggingface_hub import snapshot_download
snapshot_download("Var3n/jobads-salary", local_dir="salary_model")
snapshot_download("Var3n/jobads-position", local_dir="positions_model")
snapshot_download("Var3n/jobads-requirements-activity", local_dir="requirements_model")
```

The models were trained on text with long s replaced by `s`, which is why they run on `post_corrected_final`.

All 3 models run in parallel on each GPU (6 processes for 2 GPUs); texts are sorted by length and batched by total characters (`MAX_CHARS`). Each entity is stored as `(start_char, end_char, text)` with offsets into `post_corrected_final`.

### 10. Mapping spans back to the post-corrected text (`map_positions.py`)

`add_pc_positions` aligns `post_corrected_final` with `post_corrected` character by character and projects each `position` span onto the original text, so that the span can be shown in its original spelling and line layout (e.g. `Geschäftsdiener` from an expanded listing becomes `Geſchäfts⸗`).

Output column: `position_post_corrected`.

### 11. Gender labels and IIIF links

- **Gender:** every extracted position gets `female`, `male` or `neutral` from rules: exception lists, indicator words (`mädchen`, `frau`, `herr`, `bursche`, …) and suffixes (`-in`, `-innen`, `-euse` vs. `-er`, `-eur`, `-ist`, …). Output column `gender`, as `(start, end, label)`.
- **IIIF link:** a URL to the cropped region on the ÖNB IIIF image server:
  `https://iiif.onb.ac.at/images/ANNO/<code><yyyymmdd>/00000<page>/<x>,<y>,<w>,<h>/full/0/default.png`

Final output: `<code>_extractions.csv`.

## Output schema (`<code>_extractions.csv`)

| Column | Description |
|---|---|
| `metadata` | Page ID `<code>_<yyyymmdd>_<page>` |
| `region_id`, `reading_order` | Region ID in the PAGE-XML and its position in the reading order |
| `min_x`, `max_x`, `min_y`, `max_y` | Region bounding box (pixels) |
| `text` | Raw OCR text |
| `post_corrected` | ByT5-corrected text (historical spelling, line breaks kept) |
| `post_corrected_final` | Normalised text used for NER |
| `prediction_label` | Ad type (`heading`, `job_offer`, `job_search`, `service_offer`, `vermittlung`) |
| `salary`, `salary_importance`, `salary_period`, `unspecific_salary`, `verpflegung` | Salary entities |
| `position` | Position titles |
| `position_post_corrected` | `position` spans mapped onto `post_corrected` |
| `activity`, `attitude towards work`, `background`, `interpersonal`, `job specific`, `language` | Requirement entities |
| `gender` | Gender label for each position span |
| `iiif_link` | Link to the region image |

Entity columns hold Python-literal lists of tuples, `[(start, end, 'text'), ...]`. Read them with `ast.literal_eval`.

## Requirements

- Python 3.11+ (possessive quantifiers in `listing_hyphen.py` need 3.11; a fallback exists for older versions)
- PyTorch with CUDA, `transformers`, `spacy` + `spacy-transformers`, `huggingface_hub`, `pandas`, `scikit-learn`, `Pillow`, `requests`, `tqdm`
- spaCy versions: the salary and position models were trained with spaCy 3.7.5, the requirements model with 3.8.5. They load under spaCy 3.8 with a compatibility warning (W095).
- `ocrd_models` (PAGE-XML parsing)
- Tesseract 5 with the `frak2021` model
- Eynollah in a separate conda environment (TensorFlow 2.12, CUDA 11.8, cuDNN 8)

## Running

Open `Job_ads_pipeline_final.ipynb`, set `tag` (newspaper code) at the top of the section and run the sections in order. Each section reads the output file of the previous one, so steps can be rerun individually. Paths in the notebook need to be adapted.

## Models

The project's own models are published on Hugging Face under CC BY 4.0.

| Step | Model | Location |
|---|---|---|
| Page classification | BEiT/DiT image classifier | [`Var3n/DiT_classification_jobads_finetuned`](https://huggingface.co/Var3n/DiT_classification_jobads_finetuned) |
| Layout analysis | Eynollah | <https://github.com/qurator-spk/eynollah> |
| OCR | Tesseract `frak2021-0.905` | <https://github.com/tesseract-ocr/tesstrain> (UB Mannheim) |
| Region classification | JobAds Identification Model | [`Var3n/jobads-identification`](https://huggingface.co/Var3n/jobads-identification) |
| Post-correction | ByT5-large OCR post-correction | [`Var3n/byt5-large-ocr-postcorrection`](https://huggingface.co/Var3n/byt5-large-ocr-postcorrection) |
| Ad type classification | JobAds Classification Model | [`Var3n/jobads-classification`](https://huggingface.co/Var3n/jobads-classification) |
| NER – salary | JobAds Salary Model | [`Var3n/jobads-salary`](https://huggingface.co/Var3n/jobads-salary) |
| NER – positions | JobAds Position Model | [`Var3n/jobads-position`](https://huggingface.co/Var3n/jobads-position) |
| NER – requirements | JobAds Requirements and Activity Model | [`Var3n/jobads-requirements-activity`](https://huggingface.co/Var3n/jobads-requirements-activity) |

All text models build on [`stefan-it/zeitungs-lm-v1`](https://huggingface.co/stefan-it/zeitungs-lm-v1) (Schweter, 2024).

Training data: the ground truth published with the CHR 2024 paper ([`JobAds-FWFProject/Ground-Truth-CHR2024`](https://github.com/JobAds-FWFProject/Ground-Truth-CHR2024)) and the position extraction data (`JobAds-FWFProject/PositionsExtraction`).

## Publications

- Venglarova, K., Adam, R., Mölzer, W., Balasubramanian, S., Kleinert, J., Füllsack, M., Vogeler, G. (2024). *Who Advertises in Newspapers? Data Criticism in Mining Historical Job Ads.* CHR 2024, CEUR-WS Vol. 3834. <https://ceur-ws.org/Vol-3834/paper12.pdf>
- Venglarova, K., Adam, R., Vogeler, G. (2024). *Extracting position titles from unstructured historical job advertisements.* NLP4DH 2024. <https://aclanthology.org/2024.nlp4dh-1.8/>
- Venglarova, K., Adam, R., Mölzer, W., Balasubramanian, S., Füllsack, M., Kleinert, J., Vogeler, G. (2025). *JobAds: From Digitized Newspapers to Economic Analysis.* DHd 2025. <https://zenodo.org/records/14943168>
- Adam, R., Venglarova, K., Vogeler, G. (2025). *Exploring Historical Labor Markets: Computational Approaches to Job Title Extraction.* Preprint, hal-04869347.
- Kleinert, J., Mölzer, W. (2026). *Emergence of the Austrian Labor Market: A Historical Perspective on Recruitment Channels and Newspaper Job Advertisements.* Graz Economics Papers 2026-12.

## Acknowledgements

Funded by the Austrian Science Fund (FWF), project P35783, in cooperation with the Austrian National Library (ÖNB). Newspaper images: ANNO – AustriaN Newspapers Online, ÖNB.

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
