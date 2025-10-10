# support_chat_ui_app.py
import streamlit as st
from realtime_support_graph import app  # realtime_support_graph.py
from typing import Optional, List

st.set_page_config(page_title="Real-Time AI Support Graph", page_icon="🤖", layout="centered")

st.title("💬 Real-Time AI Support System")
st.caption("Powered by LangGraph workflow & your custom AI decision logic.")

# --- Input Form ---
with st.form("support_form"):
    st.subheader("📝 Enter Support Request")

    user_id = st.text_input("User ID", value="U001")
    message = st.text_area("Message", placeholder="Describe your issue here...", height=150)
    context_text = st.text_area("Conversation history (optional)", placeholder="One message per line...")
    
    submitted = st.form_submit_button("Submit")

# --- When user submits the form ---
if submitted:
    if not message.strip():
        st.error("Please enter a message.")
    else:
        # Build the state dictionary that matches SupportState
        context_list: Optional[List[str]] = [line.strip() for line in context_text.splitlines() if line.strip()]
        state = {
            "user_id": user_id,
            "message": message,
            "context": context_list
        }

        st.info("⏳ Running support workflow...")

        # ✅ Invoke your LangGraph workflow
        result = app.invoke(state)

        # --- Display results ---
        st.success("✅ Workflow executed successfully!")

        st.subheader("📊 Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Sentiment:** {result.get('sentiment', '-')}")
            st.write(f"**Category:** {result.get('category', '-')}")
            st.write(f"**Priority:** {result.get('priority', '-')}")
        with col2:
            st.write(f"**Requires Human:** {result.get('requires_human', False)}")
            st.write(f"**Escalated:** {result.get('escalate', False)}")
            if result.get("ticket_id"):
                st.write(f"**Ticket ID:** {result['ticket_id']}")

        st.divider()
        st.subheader("💬 Response")
        st.write(result.get("response", "No response generated."))

        st.divider()
        with st.expander("🧾 Raw State Data"):
            st.json(result)

st.markdown("---")
st.caption("Built with ❤️ using LangGraph + Streamlit")
