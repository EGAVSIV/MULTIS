# MULTIS B2 – PC Scanner
1. Keep `COPIEDDATA` in this project root.
2. Run `run_scanner_server.bat` on the Windows PC.
3. Expose port 8000 with a secure HTTPS tunnel.
4. Put that HTTPS tunnel URL in `docs/config.js`.
5. Push only `docs/` changes to GitHub Pages.

The Python server reads the same local COPIEDDATA and calls the extracted scanner functions. No Render, api.raosab.in, localhost URL, or cloud scanner is hardcoded.
