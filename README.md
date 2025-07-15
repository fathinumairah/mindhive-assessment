# MindHive Assessment Zus Coffee Assistant

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A Groq API key (for LLM access)

### Setting Up Virtual Environment

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.\.venv\Scripts\activate
```

3. Install dependencies on frontend and backend:

```bash
cd backend-fastapi
pip install -r requirements.txt
```

```bash
cd frontend-streamlit
pip install -r requirements.txt
```

4. Set up environment variables:

Create a `.env` file in the root directory with the following content:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace `your_groq_api_key_here` with your actual Groq API key.

### Running the Application

1. Make sure your virtual environment is activated (you should see `(venv)` in your terminal prompt)

2. Start the FastAPI backend server:

```bash
cd backend-fastapi
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

3. In a new terminal, start the Streamlit app:

```bash
cd frontend-streamlit
streamlit run app.py
```

The chat interface will open in your browser at `http://localhost:8501`

### Development Notes

- Always activate the virtual environment before running the application or installing new packages
- If you install new packages, update requirements.txt:
