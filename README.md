# Day 1 - Project Notes

## Goal
- Build a simple AI chatbot using Python, LangChain, and Gradio.
- Use Gemma3 model via LangChain Ollama.
- Minimal UI, just textbox input/output.

## What I did today:
- Set up Python environment (Python 3.13).
- Installed dependencies: langchain_ollama, langchainP_core, gradio.
- Wrote `zen_bot.py`:
  - Used `ChatOllama` for LLM.
  - Created a basic system prompt for helpful assistant.
  - Defined a `chatbot()` function to handle user input.
  - Built a Gradio UI with input box, output box, and submit button.
  - Connected everything and launched the app.

## Learnings & Observations:
- LangChain makes chaining prompts and models easy.
- Gradio is super quick for prototyping UIs.
- The model responds well to simple prompts.
- Need to explore more advanced prompt engineering later.

## Next Steps:
- Add conversation history.
- Improve prompt for more context.
- Explore saving chat logs.

---

**Files created/modified:**
- `zen_bot.py`
- `requirements.txt`