# LLM initialization and chain workflow creation

def switch_state_reader():
    """
    This function reads the switch states from the database.\n
    Returns:
        SwitchState: An instance of the SwitchState model containing the current switch states.
    """
    from .models import SwitchState

    return SwitchState.objects.first()


def retriever_creator(switch_state):
    """
    Creates and returns a retriever and an ensemble retriever for document retrieval.

    This function performs the following steps:
    1. Determines the directory for retrieving PDF documents based on the `switch_state`.
    2. Loads PDF documents from the specified directory.
    3. Splits the loaded documents into smaller chunks using a text splitter.
    4. Creates embeddings for the document chunks using the GPT4AllEmbeddings model.
    5. Initializes or loads a persistent vector database (Chroma) to store the embeddings.
    6. Creates a retriever from the vector database.
    7. Creates a keyword-based retriever using the BM25 algorithm.
    8. Combines the vector-based and keyword-based retrievers into an ensemble retriever.

    Args:
        switch_state (object): An object that contains the `retrieve_dir` attribute specifying the directory for retrieving PDF documents.

    Returns:
        tuple: A tuple containing the vector-based retriever and the ensemble retriever.
    
    Required modules: langchain, langchain_community, pypdf, tiktoken, gpt4all, langchain-nomic, chromadb, rank-bm25
    
    """

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    import os

    # Get the current dir and join the rest
    if not switch_state.retrieve_dir:
        try:
            retrieve_dir = os.path.join(os.path.abspath(os.curdir), 'docs')
        except:
            pass
    else:
        retrieve_dir = switch_state.retrieve_dir
    persist_dir = os.path.join(os.path.abspath(os.pardir), 'app\\chroma_persist')

    files = os.listdir(retrieve_dir)

    pdf_paths = [os.path.join(retrieve_dir, file) for file in files]

    docs = [PyPDFLoader(pdf_path).load() for pdf_path in pdf_paths]
    # Flatten the list
    docs_list = [item for sublist in docs for item in sublist] # list of tuples [('page_content', '...'), ('metadata', '...'), ('type', 'Document'), ...] 

    # Creating the text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=25
    )
    doc_splits = text_splitter.split_documents(docs_list)


    ### Create the embedding for the vector db ###

    # As per the source of GPT4AllEmbeddings model_name is required, however embedding=GPT4AllEmbeddings() can be used also.
    # If however it returns the error: you need to have model name and pip install gpt4all choose a model from the available ones:
    # https://docs.gpt4all.io/gpt4all_python_embedding.html

    from langchain_community.embeddings import GPT4AllEmbeddings

    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    # model_name = "nomic-embed-text-v1.5.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}

    embedding=GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
    )


    ### Establish vector db ###

    if not switch_state.files_retrieve_from_changed:
        ### Using the saved vectordb (Chroma) ###

        from langchain_community.vectorstores import Chroma

        # Using saved db important parameters: the name of the collection, persist dir, embeddings
        vectorstore = Chroma(collection_name="rag-chroma", persist_directory=persist_dir, embedding_function=embedding)

    else:
        ### Creating a persistant vectoredb using Chroma ###

        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embedding,
            persist_directory=persist_dir
        )


    ### Creating the retriever ###

    retriever = vectorstore.as_retriever()


    ### Keyword search ###

    from langchain.retrievers import BM25Retriever, EnsembleRetriever

    keyword_retriever = BM25Retriever.from_documents(doc_splits)
    keyword_retriever.k =  3

    ensemble_retriever = EnsembleRetriever(retrievers=[retriever,keyword_retriever],
                                        weights=[0.5, 0.5])
    
    return retriever, ensemble_retriever


def llm_chooser(switch_state):
    """
    Chooses the appropriate LLM (Large Language Model) based on the provided switch state.

    Args:
        switch_state: An object that contains the LLM switch state.

    Returns:
        tuple:
            - local_llm (str or None): The name of the local LLM if chosen, otherwise None.
            - model_name (str or None): The name of the remote model if chosen, otherwise None.

    Raises:
        Exception: If the switch state does not match any known LLM configurations.
    """

    if switch_state.llm_switch == "Groq/llama3-8b-8192":
        return None, "llama3-8b-8192"
    elif switch_state.llm_switch == "Groq/llama3-70b-8192":
        return None, "llama3-70b-8192"
    elif switch_state.llm_switch == "Groq/mixtral-8x7b-32768":
        return None, "mixtral-8x7b-32768"
    elif switch_state.llm_switch == "Groq/gemma-7b-it":
        return None, "gemma-7b-it"
    elif switch_state.llm_switch == "Ollama/llama3-8b-8192":
        return "llama3", None
    elif switch_state.llm_switch == "Ollama/phi3-mini-128K":
        return "phi3:mini", None
    else:
        raise Exception("Invalid LLM switch state")
    

def retrieval_grader(switch_state, local_llm='phi3:mini', model_name=None, groq_api_key=None):
    """
    Creates a retrieval grader to assess the relevance of retrieved documents to a user question using either a local LLM or Groq.

    This function performs the following steps:
    1. Determines which LLM to use based on the `switch_state`.
    2. Sets up a prompt template for grading document relevance.
    3. Initializes the chosen LLM (either ChatOllama or ChatGroq).
    4. Returns a pipeline that includes the prompt, the LLM, and a JSON output parser.

    Args:
        switch_state (object): An object that contains the `llm_switch` attribute to choose between local LLM or Groq.
        local_llm (str, optional): The model name for the local LLM. Defaults to 'phi3:mini'.
        model_name (str, optional): The model name for Groq if used.
        groq_api_key (str, optional): The API key for accessing Groq.

    Returns:
        Pipeline: A pipeline consisting of the prompt template, the LLM, and a JSON output parser.

    Required modules: langchain, langchain_community, langchain_core, langchain_groq
    """

    from langchain.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser


    if 'Ollama' in switch_state.llm_switch:

        # Local LLM

        llm = ChatOllama(model=local_llm, format="json", temperature=0)

        if 'phi3' in switch_state.llm_switch:

            prompt = PromptTemplate(
                template="""<|system|>You are a grader assessing relevance 
                of a retrieved document to a user question. If the document contains keywords related to the user question, 
                grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
                <|end|><|user|>
                Here is the retrieved document: \n\n {document} \n\n
                Here is the user question: {question} \n <|end|><|assistant|><|end|>
                """,
                input_variables=["question", "document"],
            )

        else:
            prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
                of a retrieved document to a user question. If the document contains keywords related to the user question, 
                grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Here is the retrieved document: \n\n {document} \n\n
                Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """,
                input_variables=["question", "document"],
            )

    else:

        # LLM with groq
        from langchain_groq import ChatGroq

        prompt = PromptTemplate(
        template = """
        You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
        Here is the retrieved document:\n\n {document}\n\n
        Here is the user question: {question}
        """,
            input_variables=["question", "document"]
        )

        llm = ChatGroq(groq_api_key=groq_api_key, temperature=0, model_name=model_name)

    return prompt | llm | JsonOutputParser()


def generator(switch_state, local_llm='phi3:mini', model_name=None, groq_api_key=None):
    """
    Creates a generator for generating answers to user questions using either a local LLM or Groq.

    This function performs the following steps:
    1. Determines which LLM to use based on the `switch_state`.
    2. Sets up a prompt template for generating answers to user questions.
    3. Initializes the chosen LLM (either ChatOllama or ChatGroq).
    4. Returns a pipeline that includes the prompt, the LLM, and a string output parser.

    Args:
        switch_state (object): An object that contains the `llm_switch` attribute to choose between local LLM or Groq.
        local_llm (str, optional): The model name for the local LLM. Defaults to 'phi3:mini'.
        model_name (str, optional): The model name for Groq if used.
        groq_api_key (str, optional): The API key for accessing Groq.

    Returns:
        Pipeline: A pipeline consisting of the prompt template, the LLM, and a string output parser.

    Required modules: langchain, langchain_community, langchain_core, langchain_groq
    """

    from langchain_core.output_parsers import StrOutputParser


    if 'Ollama' in switch_state.llm_switch:

        # Local LLM

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(model=local_llm, temperature=0)

        if 'phi3' in switch_state.llm_switch:
            # Individual Prompt from idea from a youtuber (seems better result than Lance's prompt)
            prompt = ChatPromptTemplate.from_template(
            template="""
            <|system|>
            You are an AI Assistant that follows instructions extremely well.
            Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

            Keep in mind, you will lose the job, if you answer out of CONTEXT questions

            CONTEXT: {context}
            <|end|><|user|>
            {question}
            <|end|><|assistant|><|end|>
            """
            )
        else:
            # Individual Prompt from idea from a youtuber (seems better result than Lance's prompt)
            prompt = ChatPromptTemplate.from_template(
            template="""
            <|start_header_id|>system<|end_header_id|>
            You are an AI Assistant that follows instructions extremely well.
            Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

            Keep in mind, you will lose the job, if you answer out of CONTEXT questions

            CONTEXT: {context}
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            {question}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
            )

    else:

        # LLM with groq

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq

        llm = ChatGroq(groq_api_key=groq_api_key, temperature=0, model_name=model_name)

        system = """
        You are an AI Assistant that follows instructions extremely well. lease be truthful and give direct answers.
        Please tell 'I don't know' if user query is not in CONTEXT

        # Keep in mind, you will lose the job, if you answer out of CONTEXT questions
        CONTEXT: {context}
        """

        human = "{question}"

        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    # Chain
    return prompt | llm | StrOutputParser()


def router(switch_state, local_llm='phi3:mini', model_name=None, groq_api_key=None):
    """
    Creates a router to direct user questions to either a vectorstore or a web search using either a local LLM or Groq.

    This function performs the following steps:
    1. Determines which LLM to use based on the `switch_state`.
    2. Sets up a prompt template for routing user questions.
    3. Initializes the chosen LLM (either ChatOllama or ChatGroq).
    4. Returns a pipeline that includes the prompt, the LLM, and a JSON output parser.

    Args:
        switch_state (object): An object that contains the `llm_switch` attribute to choose between local LLM or Groq.
        local_llm (str, optional): The model name for the local LLM. Defaults to 'phi3:mini'.
        model_name (str, optional): The model name for Groq if used.
        groq_api_key (str, optional): The API key for accessing Groq.

    Returns:
        Pipeline: A pipeline consisting of the prompt template, the LLM, and a JSON output parser.

    Required modules: langchain, langchain_community, langchain_core, langchain_groq
    """

    from langchain_core.output_parsers import JsonOutputParser
    from langchain.prompts import PromptTemplate

    if 'Ollama' in switch_state.llm_switch:

        # Local LLM

        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(model=local_llm, format="json", temperature=0)

        if 'phi3' in switch_state.llm_switch:
            prompt = PromptTemplate(
                template="""<|system|> You are an expert at routing a 
                user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
                prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
                in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
                or 'vectorstore' based on the question. Return a JSON with a single key 'datasource' and 
                no premable or explaination. Question to route: {question} <|end|><|assistant|><|end|>""",
                input_variables=["question"],
            )
        else:
            prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
                user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
                prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
                in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
                or 'vectorstore' based on the question. Return a JSON with a single key 'datasource' and 
                no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=["question"],
            )

    else:

        # LLM with groq

        from langchain_groq import ChatGroq

        prompt = PromptTemplate(
        template = """
        You are an expert at routing a user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents,
        prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return a JSON with a single key 'datasource' and 
        no premable or explaination. Question to route: {question}
        """,
        input_variables=["question"]
        )

        llm = ChatGroq(groq_api_key=groq_api_key, temperature=0, model_name=model_name)

    return prompt | llm | JsonOutputParser()


def web_search_chooser(switch_state):
    """
    Chooses the appropriate web search tool based on the `switch_state`.

    This function performs the following steps:
    1. Checks the `web_search_switch` attribute of the `switch_state`.
    2. Returns the corresponding web search tool.

    If `switch_state.web_search_switch` is "Tavily":
        - Uses the TavilySearchResults tool to perform web searches.
        - Required modules: tavily-python

    If `switch_state.web_search_switch` is "Google":
        - Uses the GoogleSearchAPIWrapper tool to perform web searches.
        - Required modules: google-api-python-client>=2.100.0, langchain-google-community
        - Sets up the required environment variables for Google Custom Search API.

    Args:
        switch_state (object): An object that contains the `web_search_switch` attribute to choose the web search tool.

    Returns:
        object: An instance of the chosen web search tool, or 'None' if no valid switch is provided.

    Required modules: tavily-python, google-api-python-client>=2.100.0, langchain-google-community
    """

    if switch_state.web_search_switch == "Tavily":
        # Tavily web search
        # Required modules: tavily-python
        from langchain_community.tools.tavily_search import TavilySearchResults

        return TavilySearchResults(k=3)

    elif switch_state.web_search_switch == "Google":
        # Google web search
        # Required modules: google-api-python-client>=2.100.0, langchain-google-community
        import os
        from langchain_google_community import GoogleSearchAPIWrapper
        from langchain_core.tools import Tool

        return Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=GoogleSearchAPIWrapper(k=3).run,
    )
    else:
        return 'None'


def wikipedia_search_creator():
    """
    Creates a tool for searching Wikipedia.

    This function sets up and returns a Wikipedia search tool using the WikipediaAPIWrapper.

    Returns:
        WikipediaQueryRun: An instance of the WikipediaQueryRun tool configured with the WikipediaAPIWrapper.

    Required modules: wikipedia, langchain_community
    """

    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun

    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


def pubmed_search_creator():
    """
    Creates a tool for searching PubMed.

    This function sets up and returns a PubMed search tool using the PubMedAPIWrapper.

    Returns:
        PubMedAPIWrapper: An instance of the PubMedAPIWrapper tool configured with the specified number of top results.

    Required modules: xmltodict, langchain_community
    """

    from langchain_community.utilities.pubmed import PubMedAPIWrapper

    return PubMedAPIWrapper(top_k_results=5)


### Control flow ###
"""
    Creates a workflow control system for processing user queries.

    This function defines a workflow control system for processing user queries. It consists of several components:

    1. State Definition: The GraphState class defines the state of the workflow, which includes various attributes such as switch states, question, generation, web search results, and documents.

    2. Nodes: There are several nodes in the workflow, each responsible for a specific task:
        - route_question: Determines whether to route the question for further processing or skip directly to retrieval or web search based on the switch state.
        - retrieve: Retrieves relevant documents based on the user question using an ensemble retriever.
        - generate: Generates a response to the user question using a RAG (Retrieval-Augmented Generation) model.
        - web_search: Conducts a web search for additional information related to the user question.
        - wikipedia_search: Searches Wikipedia for relevant articles related to the user question.
        - pubmed_search: Searches PubMed for scientific articles related to the user question.

    3. Conditional Edge: The workflow includes conditional edges to determine the flow of execution based on certain conditions. For example, after retrieval, it decides whether to continue with web search or Wikipedia search based on the web search switch state.

    4. Workflow Creation: The create_workflow function defines the entire workflow by adding nodes, defining conditional entry points, and establishing edges between nodes.

    Returns:
        StateGraph: An instance of StateGraph representing the compiled workflow.

    Required modules: langgraph
    """

from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document

# State

class GraphState(TypedDict):

    switch_state: object
    question_router: object
    ensemble_retriever: object
    rag_chain: object
    web_search_tool: object
    wikipedia: object
    pubmed: object
    question: str
    generation: str
    web_search: str
    documents: List[str]


# Nodes, edges

# Routing - Conditional >ENTRY< point
# If routing switched on by the user then route the question else skip routing and goes to websearch
def route_question(state):

    if state['switch_state'].routing_switch:
        print("---ROUTE QUESTION---")
        source = state['question_router'].invoke({"question": state["question"]})
        if source["datasource"] == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source["datasource"] == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
    else:
        return "retrieve"
    
# Retriever - Node
def retrieve(state):

    if state['switch_state'].retriever_switch == True:
        print("---RETRIEVE---")
        question = state["question"]
        documents = state['ensemble_retriever'].invoke(question)
        
        return {"documents": documents, "question": question}
    
# Generator - Node
def generate(state):

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = state['rag_chain'].invoke({"context": documents, "question": question})

    return {"documents": state["documents"], "question": question, "generation": generation}

# Web Search - Node
def web_search(state):
    print("---WEB SEARCH---")

    question = state["question"]
    documents = state["documents"]

    if state['switch_state'].web_search_switch == "Tavily":
        print("---WEB SEARCH TAVILY---")
        docs = state['web_search_tool'].invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])

    elif state['switch_state'].web_search_switch == "Google":
        print("---WEB SEARCH GOOGLE---")
        web_results = state['web_search_tool'].run(question)

    else:
        return {"documents": documents, "question": question}
    
    web_results = Document(page_content=web_results)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
        
    return {"documents": documents, "question": question}

# Wikipedia Search - Node
def wikipedia_search(state):
    question = state["question"]
    documents = state["documents"]

    if state['switch_state'].wikipedia_switch:
        print("---WIKIPEDIA SEARCH---")
        docs = state['wikipedia'].run(question)
        wikipedia_results = Document(page_content=docs)
        if documents is not None:
            documents.append(wikipedia_results)
        else:
            documents = [wikipedia_results]
            
    return {"documents": documents, "question": question}

# Pubmed Search - Node
def pubmed_search(state):
    question = state["question"]
    documents = state["documents"]

    if state['switch_state'].pubmed_switch:
        print("---PUBMED SEARCH---")
        docs = state['pubmed'].load(question)
        docs = '\n\n'.join(doc['Summary'] for doc in docs)
        pubmed_results = Document(page_content=docs)
        if documents is not None:
            documents.append(pubmed_results)
        else:
            documents = [pubmed_results]
            
    return {"documents": documents, "question": question}

# Decision websearch or wikipedia search - Conditional edge
def decide_to_continue(state):
    if state['switch_state'].web_search_switch != "None":
        return "websearch"
    else:
        return "wikipedia_search"
    
# Creates the nodes and builds the workflow
def create_workflow():
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("wikipedia_search", wikipedia_search)  # generatae
    workflow.add_node("pubmed_search", pubmed_search)  # generatae


    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
            "retrieve": "retrieve"
        },
    )
    workflow.add_conditional_edges(
        "retrieve",
        decide_to_continue,
        {
            "websearch": "websearch",
            "wikipedia_search": "wikipedia_search",
        },
    )
    workflow.add_edge("websearch", "wikipedia_search")
    workflow.add_edge("wikipedia_search", "pubmed_search")
    workflow.add_edge("pubmed_search", "generate")
    workflow.add_edge("generate", END)


    ### Compile ###
    return workflow.compile()