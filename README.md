# MULTIS – Ready-Made Scanner Results

## Architecture
COPIEDDATA -> existing Python scanner engine -> SCANNER_RESULTS -> GitHub Pages HTML.

## Local run
Double-click:
RUN_ALL_SCANNERS.bat

It reads every available timeframe from COPIEDDATA and writes JSON result files under SCANNER_RESULTS.

## Website
docs/app.js loads:
../SCANNER_RESULTS/<scanner>/<timeframe>.json

No API, Render, FastAPI, tunnel, localhost, or separate backend URL is used.

## Automatic run
The included GitHub Actions workflow runs when COPIEDDATA or backend files change and commits updated SCANNER_RESULTS.
