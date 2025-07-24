# imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import gradio as gr
import uuid


# Set up gemma3 with Ollama
llm = ChatOllama(model="gemma3",temperature=0.8)

# Create a simple chat prompt here MessagePlaceholder is used to store chat history
# and it will be used to maintain the conversation context.
# and human means the user input

prompt = ChatPromptTemplate(
    messages=[
        ("system", "You are a friendly and helpful assistant named ZenBot.Start with the greeting 'Hello, I am ZenBot. How can I assist you today?'. Answer the user's questions to the best of your ability."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# store is used to maintain the session history and is a empty dictionary
store = {}

# Function to get session history based on session_id
# If session_id is not present in store, it initializes a new InMemoryChatMessageHistory
# This allows the bot to maintain conversation context across different sessions.
#InMemoryChatMessageHistory means that the chat history is stored in memory.
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# RunnableWithMessageHistory is used to combine the prompt and llm with the session history
# It allows the bot to maintain conversation context by keeping track of the messages exchanged.
# It takes the prompt, llm, and get_session_history function as input.
# The input_messages_key is used to specify the key for user input messages,
# and history_messages_key is used to specify the key for the chat history messages.
# The chain is a runnable that combines the prompt and llm with the session history.
chain = RunnableWithMessageHistory(
    runnable= prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Function to handle user input and generate a response
# It takes user_input, history_state, and session_id as input.
# The user_input is the question asked by the user,
# history_state is the current state of the chat history,
# and session_id is a unique identifier for the session.
# The function invokes the chain with the user input and session_id,
# and appends the user input and response to the history_state.
# It returns the updated history_state, the response, and the session_id.
# The session_id is generated using uuid.uuid4() to ensure uniqueness

def chatbot(user_input, history_state, session_id=str(uuid.uuid4())):
    if user_input is None or user_input.strip() == "":
        # Show error message for empty input
        return history_state, "Please enter a valid question!", session_id

# Invoke the chain with user input and session_id
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    ).content

    if not isinstance(history_state, list):
        history_state = []
    history_state.append((user_input, response))

    return history_state, response, session_id


# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Zensar Chatbot")
    history_state = gr.State(value=None)
    session_id= gr.State(value=str(uuid.uuid4()))
    input_box = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    output_box= gr.Textbox(label="Answer",interactive=False)
    submit_button=gr.Button("Submit")

# Connect the submit button to the chatbot function
    submit_button.click(
        fn=chatbot,
        inputs=[input_box, history_state, session_id],
        outputs=[output_box, history_state, session_id]
    )

    # Add a clear history button to reset the chat history
    clear_button = gr.Button("Clear History")

    def clear_history(history_state, session_id):
        return [], "", session_id

    clear_button.click(
        fn=clear_history,
        inputs=[history_state, session_id],
        outputs=[output_box, history_state, session_id]
    )
    # Add a slider for temperature control
    temperature_slider = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.8,
        step=0.01,
        label="Temperature",
        info="Adjust the creativity of ZenBot's responses"
    )

    # Update the chatbot function to accept temperature
    def chatbot(user_input, history_state, session_id, temperature):
        if user_input is None or user_input.strip() == "":
            return history_state, "Please enter a valid question!", session_id, temperature

        # Create a new ChatOllama instance with the selected temperature
        llm_dynamic = ChatOllama(model="gemma3", temperature=temperature)
        chain_dynamic = RunnableWithMessageHistory(
            runnable=prompt | llm_dynamic,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        response = chain_dynamic.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        ).content

        if not isinstance(history_state, list):
            history_state = []
        history_state.append((user_input, response))

        # Show response and temperature value
        display_response = f"{response}\n\n**Temperature:** {temperature:.2f}"

        return history_state, display_response, session_id, temperature

    # Update the submit button to include temperature
    submit_button.click(
        fn=chatbot,
        inputs=[input_box, history_state, session_id, temperature_slider],
        outputs=[output_box, history_state, session_id]
    )

    # Update clear_history to reset temperature to default (0.8)
    def clear_history(history_state, session_id, temperature):
        return [], "", session_id, temperature

    clear_button.click(
        fn=clear_history,
        inputs=[history_state, session_id, temperature_slider],
        outputs=[output_box, history_state, session_id, temperature_slider]
    )
# Launch the Gradio app
demo.launch()