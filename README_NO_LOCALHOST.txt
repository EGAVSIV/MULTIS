# EGAVSIV/MULTIS – No Localhost Update

## Data path
The Python loader reads directly from:

COPIEDDATA/
├── stock_data_15/
├── stock_data_1H/
├── stock_data_D/
├── stock_data_W/
└── stock_data_M/

No Data-Collector API is used during a scan.

## Frontend
`docs/app.js` contains no `127.0.0.1` or `localhost` fallback.
Set the public HTTPS backend address in:

`docs/config.js`

Example:
`window.SCANNER_API_BASE = "https://your-api-domain";`

## Important
GitHub Pages serves HTML/CSS/JS only. The existing Python scanner still requires a public Python runtime to execute `backend/api.py`.
