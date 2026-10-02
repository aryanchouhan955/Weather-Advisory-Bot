import streamlit as st
import streamlit.components.v1 as components

st.title("Test Chat Input")

if st.button("Click to fill"):
    text = "Hello world"
    js = f"""
    <script>
    const textarea = window.parent.document.querySelector('[data-testid="stChatInput"] textarea');
    if (textarea) {{
        textarea.value = '{text}';
        const tracker = textarea._valueTracker;
        if (tracker) tracker.setValue('');
        textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
    }}
    </script>
    """
    components.html(js, height=0, width=0)

st.chat_input("Type here")
