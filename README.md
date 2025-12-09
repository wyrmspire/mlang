# MES Pattern Generator

A quantitative development project for analyzing and generating synthetic MES futures price paths.

## Architecture

- **Data Pipeline**: Python scripts (`src/`) to ingest JSON, process to Parquet, and extract hour-level features.
- **Pattern Learning**: K-Means clustering of intraday hours by Session, Day of Week, and Hour.
- **Generator**: A non-parametric bootstrapper that stitches historical hour patterns to create realistic synthetic days.
- **API**: FastAPI (`src/api.py`) serving data and generation capabilities.
- **Frontend**: React + TypeScript + Vite (`frontend/`) for visualization.
- **State Features**: Daily state (trend, volatility, gaps) is extracted (`src/state_features.py`) and used to bias multi-day generation.

## Setup

### Backend

1.  **Prerequisites**: Python 3.11+.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data Setup**:
    - Place raw MES JSON files in `data/raw/`.
    - Run the pipeline:
        ```bash
        # 1. Ingest and create 1-min parquet
        python -m src.preprocess
        
        # 2. Extract hour features
        python -m src.feature_engineering

        # 3. Extract daily state features
        python -m src.state_features
        
        # 4. Build pattern library (incorporates state)
        python -m src.pattern_library
        ```
4.  **Run API**:
    ```bash
    uvicorn src.api:app --reload --port 8000
    ```
    API will be available at `http://localhost:8000`.

### Frontend

1.  **Prerequisites**: Node.js 18+.
2.  **Install**:
    ```bash
    cd frontend
    npm install
    ```
3.  **Run**:
    ```bash
    npm run dev
    ```
    App will be available at `http://localhost:5173` (or similar, check console).

## Usage

### Single Day Mode
1.  Use the **Single** toggle in the sidebar.
2.  Select a specific date to view historical data.
3.  Configure "Generator" (Day of Week, Session) and click "Generate Session".
4.  Synthetic path overlays on top of the historical comparison.

### Multi-Day Mode (New)
1.  Use the **Multi-Day** toggle in the sidebar.
2.  Select a **Start Date** and number of days (e.g., 5, 20).
3.  Click "Generate Sequence".
4.  Two charts will appear:
    - **Top**: Real historical sequence starting from the selected date.
    - **Bottom**: Synthetic sequence generated continuously for N days, with state-aware transitions.

## Key Files

- `src/config.py`: Configuration constants.
- `src/generator.py`: Core generation logic (Single and Multi-day).
- `src/state_features.py`: Daily state extraction.
- `src/pattern_library.py`: Clustering logic.
- `frontend/src/App.tsx`: Main UI.
# mlang
