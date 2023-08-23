import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
import streamlit as st
from dataclasses import dataclass
from typing import Literal
from langchain.callbacks import get_openai_callback
import subprocess
import re
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load environment variables from .env file
load_dotenv()


@dataclass
class Message:
    """Class for keeping track of a chat message."""

    origin: Literal["human", "ai"]
    message: str


def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


# def on_click_callback():
#     with get_openai_callback() as cb:
#         human_prompt = st.session_state.human_prompt
#         llm_response = st.session_state.conversation.run(human_prompt)
#         st.session_state.history.append(Message("human", human_prompt))
#         st.session_state.history.append(Message("ai", llm_response))
#         st.session_state.token_count += cb.total_tokens


def on_click_callback2():
    human_prompt = st.session_state.human_prompt
    llm_response = st.session_state.qa(
        {"question": human_prompt, "chat_history": st.session_state.history}
    )
    st.session_state.history.append((human_prompt, llm_response["answer"]))


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        # llm = OpenAI(
        #     temperature=0,
        #     openai_api_key=st.secrets["OPENAI_API_KEY"],
        #     model_name="text-davinci-003",
        # )

        # st.session_state.conversation = ConversationChain(
        #     llm=llm, memory=ConversationSummaryMemory(llm=llm)
        # )
        pass
    if "qa" not in st.session_state:
        embeddings = OpenAIEmbeddings()

        db = DeepLake(
            dataset_path="hub://matias/twitter-algorithm",
            read_only=True,
            embedding=embeddings,
        )

        # Retriever
        retriever = db.as_retriever()
        retriever.search_kwargs[
            "distance_metric"
        ] = "cos"  # similarity comparison function - cosine simularity
        retriever.search_kwargs["fetch_k"] = 100  # How many code chunks to fetch
        retriever.search_kwargs["maximal_marginal_relevance"] = True
        retriever.search_kwargs["k"] = 20  # Pick 20 top comparisons

        # turn on for custom filtering
        ### retriever.search_kwargs["filter"] = filter

        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
        st.session_state.qa = qa


initialize_session_state()


def download_repository(repo_url, base_folder):
    """
    Download a Git repository and store it in a subfolder with part of the repository name.

    Args:
        repo_url (str): The URL of the Git repository to download.
        base_folder (str): The path to the base folder where the repository subfolder will be created.

    Returns:
        bool: True if the repository was successfully downloaded or already exists and is up to date, False otherwise.
    """
    try:
        # Extract the repository name from the URL
        match = re.search(r"/([^/]+)\.git$", repo_url)
        if not match:
            print("Invalid repository URL.")
            return False

        repo_name = match.group(1)
        destination_folder = os.path.join(base_folder, repo_name)

        # Check if the destination folder already exists
        if os.path.exists(destination_folder):
            print(f"Checking for updates in {destination_folder}...")

            # Change the current working directory to the destination folder
            os.chdir(destination_folder)

            # Run the git fetch command to update the repository
            subprocess.run(["git", "fetch"])

            # Check if there are any changes to pull
            fetch_result = subprocess.run(
                ["git", "diff", f"HEAD..origin/main"], stdout=subprocess.PIPE
            )
            if fetch_result.stdout:
                # There are changes to pull
                print(f"Pulling updates for {destination_folder}...")
                pull_result = subprocess.run(["git", "pull"])
                if pull_result.returncode == 0:
                    print(f"Repository in {destination_folder} is up to date.")
                    return True
                else:
                    print(f"Failed to pull updates for {destination_folder}.")
                    return False
            else:
                # No changes to pull
                print(f"Repository in {destination_folder} is already up to date.")
                return True

        # If the destination folder doesn't exist, clone the repository
        else:
            print(f"Cloning {repo_url} to {destination_folder}...")
            os.makedirs(destination_folder)
            subprocess.run(["git", "clone", repo_url, destination_folder])
            print(f"Repository cloned to {destination_folder}.")
            return True

    except Exception as e:
        # Handle any exceptions that may occur during the process
        print(f"Error: {e}")
        return False


def start_streamlit_app():
    load_css()
    st.title("Talk To Your Repo!")
    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")
    token_count_placeholder = st.empty()

    with chat_placeholder:
        for chat in st.session_state.history:
            for index, message in enumerate(chat):
                origin = "human" if index % 2 == 0 else "ai"
                message = Message(origin, message)
                div = f"""
                <div class="chat-row 
                    {'' if message.origin == 'ai' else 'row-reverse'}">
                    <img class="chat-icon" src="app/static/{
                        'ai_icon.png' if message.origin == 'ai' 
                                    else 'user_icon.png'}"
                        width=32 height=32>
                    <div class="chat-bubble
                    {'ai-bubble' if message.origin == 'ai' else 'human-bubble'}">
                        &#8203;{message.message}
                    </div>
                </div>
                """
                st.markdown(div, unsafe_allow_html=True)

        for _ in range(3):
            st.markdown("")

    with prompt_placeholder:
        st.markdown("**Chat** - _press Enter to Submit_")
        cols = st.columns((6, 1))
        cols[0].text_input(
            "Chat",
            value="Hello Code Repository!",
            label_visibility="collapsed",
            key="human_prompt",
        )
        cols[1].form_submit_button(
            "Submit", type="primary", on_click=on_click_callback2
        )

    # token_count_placeholder.caption(
    #     f"""
    # Used {st.session_state.token_count} tokens \n
    # Debug Langchain conversation:
    # {st.session_state.conversation.memory.buffer}
    # """
    # )


def download_or_update_repository():
    repo_url = "https://github.com/twitter/the-algorithm.git"
    destination_folder = "repository2"  # Replace with the name of the folder where you want to store the repository

    if download_repository(repo_url, destination_folder):
        print("Repository downloaded successfully.")
    else:
        print("Failed to download the repository.")


# Used to return only relevant files
def filter(x):
    # filter based on source code
    if "com.google" in x["text"].data()["value"]:
        return False

    # filter based on path e.g. extension
    metadata = x["metadata"].data()["value"]
    return "scala" in metadata["source"] or "py" in metadata["source"]


def generate_and_store_semantic_embeddings_in_deeplake():
    embeddings = OpenAIEmbeddings()

    download_or_update_repository()
    os.chdir("C:\\Users\\matia\\Desktop\\Projects\\AI\\Repo Chat")
    print(os.getcwd())
    repo_root_dir = "./repository2/the-algorithm"
    docs = []
    for dirpath, dirnames, filenames in os.walk(repo_root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass

    print(len(docs))

    # Chunk Data
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    print(len(texts))

    db = DeepLake.from_documents(
        texts, embeddings, dataset_path="hub://matias/twitter-algorithm"
    )


def test():
    embeddings = OpenAIEmbeddings()

    db = DeepLake(
        dataset_path="hub://matias/twitter-algorithm",
        read_only=True,
        embedding=embeddings,
    )

    # Retriever
    retriever = db.as_retriever()
    retriever.search_kwargs[
        "distance_metric"
    ] = "cos"  # similarity comparison function - cosine simularity
    retriever.search_kwargs["fetch_k"] = 100  # How many code chunks to fetch
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20  # Pick 20 top comparisons

    # turn on for custom filtering
    # retriever.search_kwargs["filter"] = filter

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    questions = [
        "Can you tell me what the repository is about?",
        "Can you explain to me what the 'For You Timeline' is?",
        "Which services are used to construct the 'For You Timeline'?",
        "I am curious about the Feature Update Service. Which files in the repository should I look at?",
        "Can you give me a rundown of the main code functions present in the FeatureUpdateService.java file?",
        "Can you provide me the code present in the createResponse() function?",
    ]
    chat_history = []

    for question in questions:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")


if __name__ == "__main__":
    # generate_and_store_semantic_embeddings_in_deeplake()

    start_streamlit_app()

    # test()
