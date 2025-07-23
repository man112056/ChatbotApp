#import necessary libraries
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# Set up chatollama with the specified model
llm = ChatOllama(model="gemma3")

#create a simple chat prmpt
prompt = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful assistant. Answer the user's questions to the best of your ability."),
        ("human", "{question}"),
    ]
)
 # Create a chain that combines the prompt and the language model
chain = prompt | llm 

#function to handle user input and generate a response  
def chatbot(question):
    response = chain.invoke({"question": question})
    return response.content

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Gyaan Doot- Manish's Personal Assistant")
    input_box = gr.Textbox(label="Ask your question:",
               placeholder="Type your question here:",)
    output_box = gr.Textbox(label="Response is:", interactive=False)
    submit_button = gr.Button("Submit")

    submit_button.click(
        fn=chatbot,
        inputs=input_box,
        outputs=output_box
    )

    # Launch the Gradio app
    demo.launch()