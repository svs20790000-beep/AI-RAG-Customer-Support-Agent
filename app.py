import streamlit as st
import logging
from rag_pipeline.engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

#engine = RAGEngine()

# 2. Cache the Engine Instance
@st.cache_resource
def get_rag_engine():
    """Ensures the heavy models are only loaded once."""
    return RAGEngine()

engine = get_rag_engine()

# 4. Streamlit UI
#st.title('🚀 AI Customer Support Agent')
st.title('AI Customer Support Agent')
#st.markdown("This agent uses **DPR** for retrieval and **Flan-T5** for generation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                logger.info(f"User query received: {prompt}")
                response = engine.generate_answer(prompt)
                st.markdown(response)
                # --- KEY FIX: Append to history INSIDE the if block ---
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg="I'm sorry, I encountered an error processing that request."
                st.error(f"Error: {e}")
                logger.error(f"Generation error: {e}")    
    
    #st.session_state.messages.append({"role": "assistant", "content": response})

if st.sidebar.button("Run System Evaluation"):
    from evaluation.results_analysis import EvaluationRunner
    runner = EvaluationRunner(engine)
    results_df = runner.run_full_test()
    
    st.write("### Evaluation Results")
    st.dataframe(results_df)
    st.success(f"System Accuracy: {results_df['Correct_Answer'].mean() * 100}%")
