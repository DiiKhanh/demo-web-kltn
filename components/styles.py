def load_css():
    return """
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .good-result {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .poor-result {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .debug-section {
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
    """