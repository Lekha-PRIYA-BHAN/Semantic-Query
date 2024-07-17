from dotenv import load_dotenv
import os


import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain, LLMChain

index_to_process="../tests/index1"
#index_to_process="../index2"
index_to_process="../index1"
index_to_process="../index3"
index_to_process="../index4"


terraform = """As an expert terraform coder use only the context and module descriptions provided below to answer the following query:
context:
-----
{context}
-----

modules:
----
virtualmachinelinux: terraform module to create an azure linux virtual machine for linux; has variables: 'virtualmachinedata', 'required_tags', 'additional_tags'
resourcegroup: terraform module to create an azure resource group; has variables: 'resource_groups', 'application_id', 'data_classification', 'regulations', 'servicenow_support_group', 'application_purpose', 'end_date_of_the_project'
region_codes: terraform module that provides region to code map
----

Use the above context to provide an answer for the following query.

query:
-----
{query}
-----
"""

normal = """As an expert in AWS cloud architecture answer the following query as per the given context:
context:
-----
{context}
-----

Use the above context to provide an answer for the following query.

query:
-----
{query}
-----
"""

yamlsearch = """As an expert in yaml syntax, answer the following query as per the given context:

context:
-----
{context}
-----

Use the above context to provide an answer for the following query.

query:
-----
{query}
-----

provide the yaml properly formatted and indented, highlighting in bold font the values that have changed but retaining unchanged values in the produced yaml
"""

template = {}
template['normal'] = normal
template['terraform'] = terraform
template['yamlsearch'] = yamlsearch
#document_variable_name = "context"

x=load_dotenv()

embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.load_local(index_to_process, embeddings)

st.set_page_config(page_title='Kyndryl Architecture Search', layout='wide')

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "metadata" not in st.session_state:
    st.session_state["metadata"] = []


def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_area("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')

    return input_text


def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
        save.append("Sources:" + st.session_state["metadata"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state["metadata"] = []
    #st.session_state.entity_memory.store = {}
    #st.session_state.entity_memory.buffer.clear()

def create_context(docs):
    return_value = []
    for doc in docs:
        return_value.append(doc.page_content)
    return return_value


with st.sidebar.expander(" 🛠️ Settings ", expanded=False):
    # Option to preview memory store
    #if st.checkbox("Preview memory store"):
    #    st.write(st.session_state.entity_memory.store)
    # Option to preview memory buffer
    #if st.checkbox("Preview memory buffer"):
    #    st.write(st.session_state.entity_memory.buffer)
    TEMPLATE = st.selectbox(label='Template', options=['yamlsearch', 'normal', 'terraform'])
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo', 'gpt-4', 'text-davinci-003','text-davinci-002'])
    #K = st.number_input(' (#)Summary of prompts to consider',min_value=0,max_value=1000)
    temperature = st.number_input(' temperature',min_value=0.00,max_value=1.00, value=0.45)


prompt = PromptTemplate(template=template[TEMPLATE], input_variables=["context", "query"])

# Set up the Streamlit app layout
st.title("🧠 Kyndryl Architecture Chatbot 🤖")
st.markdown(
        ''' 
        > :black[**A Chatbot that answers architecture questions**]
        ''')
# st.markdown(" > Powered by -  🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
API_O = st.sidebar.text_input(":blue[Enter Your OPENAI API-KEY :]", 
                placeholder="Paste your OpenAI API key here (sk-...)",
                value=os.getenv("OPENAI_API_KEY"),
                type="password") # Session state storage would be ideal

if API_O:
    # Create an OpenAI instance
    llm = OpenAI(temperature=temperature,
                openai_api_key=API_O, 
                model_name=MODEL, 
                verbose=False) 
   
else:
    st.markdown(''' 
        ```
        - 1. Enter API Key + Hit enter 🔐 

        - 2. Ask anything via the text input widget

        Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.
        ```
        
        ''')
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.sidebar.info("Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.")

st.sidebar.button("New Chat", on_click = new_chat, type='primary')

user_input = get_text()


if user_input:
    context_docs = knowledge_base.similarity_search(user_input, fetch_k=5)
    print("#######################START#########################################")
    print(prompt)
    print("#########################END#########################################")
    source_set = set({})
    for doc in context_docs:
        print("-----------------------START--------------------------------")
        print(type(doc))
        print(doc)
        print("++++++++++++++++++++++++END+++++++++++++++++++++++++++++++++")
        try:
            if doc.metadata != {}:
                for key in doc.metadata.keys():
                    source_set.add(str(doc.metadata[key]))
        except:
            if doc[0].metadata != {}:
                for key in doc[0].metadata.keys():
                    source_set.add(str(doc[0].metadata[key]))
        

    chain = LLMChain(llm=llm, prompt=prompt)
    #chain = StuffDocumentsChain(llm_chain=chain, document_prompt=prompt, document_variable_name=document_variable_name,)
    output=""
    with get_openai_callback() as cb:
        output = chain.run(context=context_docs, query=user_input)
        print(cb)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

    source_string = ""
    if source_set != {}:
        source_string = " , ".join(source_set)
    st.session_state.metadata.append(source_string)

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        st.info(st.session_state["metadata"][i], icon="🏠")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
        download_str.append(st.session_state["metadata"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)