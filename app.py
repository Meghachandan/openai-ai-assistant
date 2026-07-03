import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="OpenAI GenAI Q&A",
    page_icon="🤖",
)

st.title("🤖 OpenAI GenAI End-to-End Q&A")
st.write("Ask any question using OpenAI GPT models.")

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.error(
        "OPENAI_API_KEY not found. Add it in Streamlit Cloud → App Settings → Secrets."
    )
    st.stop()

# Optional LangSmith
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "OpenAI GenAI Q&A"


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Answer the user's questions clearly and accurately.",
        ),
        ("user", "{question}"),
    ]
)

st.sidebar.header("Model Settings")

model = st.sidebar.selectbox(
    "Choose Model",
    [
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4.1",
    ],
)

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.7,
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    50,
    500,
    200,
)


question = st.text_input("Enter your question")

if st.button("Generate Response"):

    if question.strip():

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        chain = prompt | llm | StrOutputParser()

        with st.spinner("Generating..."):
            response = chain.invoke({"question": question})

        st.success(response)

    else:
        st.warning("Please enter a question.")
