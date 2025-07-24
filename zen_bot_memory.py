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

def chatbot(user_input,history_state,session_id=str(uuid.uuid4())):
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    ).content
    
    if  history_state is None:
        history_state = [] 
    history_state.append((user_input, response))

    return history_state,history_state,session_id 


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

# Launch the Gradio app
demo.launch()