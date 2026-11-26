# PolySimulator Quant & News-Lag Trading Bot

Ein quantitativer, nachrichtengetriebener Paper-Trading-Bot für [PolySimulator](https://polysimulator.com) (Polymarket-Orderbuch-Simulator) mit kostenlosem 24/7-Betrieb via GitHub Actions und statischem HTML-Portfolio-Dashboard via GitHub Pages.

## Architektur

- **`bot/orchestrator.py`** — Deterministischer Ablaufplan (9 Schritte: Portfolio-Sync, Circuit-Breaker, Stale-Order-Cleanup, Exit-Check, Markt-Scanner, News-Radar, Superforecaster, Overseer, Execution + Dashboard-Export).
- **`bot/scanner.py`** — Filtert liquide, mehrtägige Märkte (2–21 Tage) und erkennt:
  - **Alpha A (Structural Math):** Multi-Outcome-Summen < 0.985 oder zeitliche Monotonie-Verletzungen (0 LLM-Tokens).
  - **Alpha B (News-Lag):** Liquide Mid-Tier-Märkte (`0.12–0.88`) mit frischen Nachrichten.
  - **Alpha C (High-Probability Carry):** Hochwahrscheinliche Ausgänge (`0.88–0.96`, 2–10 Tage) nach Trap-Check.
- **`bot/news.py` & `bot/state.py`** — Holt aktuelle Google-News-RSS-Meldungen und dedupliziert gesehene Artikel in SQLite (`data/state.db`).
- **`bot/forecaster.py`** — Blind-Superforecasting über Gemini Flash **ohne** Preis-Anchoring, kombiniert per Bayesian Log-Odds-Blending in Python.
- **`bot/overseer.py`** — Unabhängige Mathe-Risikokontrolle (0 LLM-Tokens): Quarter-Kelly (`0.25x`), max. 10 % pro Markt, max. 25 % pro Kategorie, max. 65 % Gesamt-Exposure, -10 % 24h-Drawdown Circuit Breaker.
- **`bot/execution.py`** — Maker-First (`post_only=True`, 0 % Gebühr vs. bis zu 5 % V2 Taker-Gebühr).
- **`docs/index.html`** — Statisches Dark-Mode-Dashboard (Chart.js Equity Curve, Kategorie-Exposure, offene Positionen, Superforecaster-Log mit Brier Score, Trade-Journal).

## Lokal ausführen

```powershell
# Tests ausführen
.venv\Scripts\python.exe -m pytest tests/test_math_and_overseer.py -v

# Dry-Run (sendet keine Orders)
.venv\Scripts\python.exe -m bot.orchestrator --dry-run

# Dashboard lokal ansehen
.venv\Scripts\python.exe -m http.server -d docs 8080
```

## GitHub 24/7 Deployment

1. Öffentliches Repository auf GitHub erstellen (z. B. `polysim-bot`).
2. Unter **Settings -> Secrets and variables -> Actions -> Secrets** hinterlegen:
   - `POLYSIM_API_KEY`
   - `GEMINI_API_KEY`
3. Unter **Settings -> Secrets and variables -> Actions -> Variables** optional anlegen:
   - `DRY_RUN` = `true` (für Live-Simulation auf `false` ändern)
   - `GEMINI_MODEL` = `gemini-3.5-flash-lite`
4. Unter **Settings -> Pages** Source auf `Deploy from a branch`, Branch `main`, Folder `/docs` stellen.
