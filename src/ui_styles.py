import streamlit as st


def apply_styles() -> None:
    """Apply a neutral technical documentation theme."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f6f8;
            --surface: #ffffff;
            --text: #1f2933;
            --muted: #52606d;
            --accent: #0b6e99;
            --border: #d9e2ec;
        }

        .stApp {
            background:
                radial-gradient(circle at 5% 0%, #e6f1f5 0%, transparent 30%),
                radial-gradient(circle at 95% 100%, #eef2f7 0%, transparent 30%),
                var(--bg);
            color: var(--text);
        }

        .block-container {
            max-width: 1100px;
            padding-top: 1.25rem;
        }

        .demo-card {
            border: 1px solid var(--border);
            background: var(--surface);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.75rem;
        }

        .demo-meta {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .mode-badge {
            display: inline-block;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            background: #f8fafc;
            color: var(--muted);
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        }

        .grounded-yes {
            color: #0f766e;
            font-weight: 600;
        }

        .grounded-no {
            color: #b45309;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
