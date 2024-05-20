import pprint
from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate as PromptTemplate2
from langchain_community.tools.tavily_search import TavilySearchResults
from get_embedding import get_embedding_function
from langgraph.graph import END, StateGraph

load_dotenv()

DISCORD_BOT_ID = os.getenv('DISCORDBOT_ID')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DISCORD_CHANNEL = os.getenv('DISCORD_CHANNEL')
LLM_MODEL = os.getenv('LLM_MODEL')
CHROMA_PATH = os.getenv('CHROMA_PATH')
DATA_PATH  = os.getenv('DATA_PATH')
RAG_PROMPT_TEMPLATE = os.getenv('RAG_PROMPT_TEMPLATE')
retrieval_grader = None

def retrieve(state):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    question = state["question"]
    #results = db.similarity_search_with_score(question, k=4)
    #context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    #prompt_template = PromptTemplate2.from_template(RAG_PROMPT_TEMPLATE)
    #prompt = prompt_template.format(context=context_text, question=question)

    retriever = db.as_retriever(search_kwargs={'k': 10})
   
    print("---RETRIEVE---")
    documents = retriever.invoke(question)
    print(documents)
    #print("---Getting sources---")
    #sources = [doc.metadata.get("id", None) for doc, _score in documents]
    #print("---Getting score---")
    #score = [_score for _score in documents]
    #print(score)
    #print("---Getting formatted_response---")
    #formatted_response = f"Response: {documents}\nSources: {sources}\nScore: {score}"
    #print(formatted_response)
    print("---RETRIEVE END---")
    return {"documents": documents, "question": question}

def generate(state):
    print("---GENERATE RESPONSE---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = PromptTemplate2(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are **CastorBot** from **TeamNico** team, a French Discord assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Keep the answer concise in french.
        Add Links in the answer when necessary.
        Links are supported by wrapping the text in square brackets and the link in parenthesis, e.g [Example](https://example.com).
        If the subject of the conversation is about **TeamNico**, add this emoji <:teamNico:547934533182423050> to the response.
        Si ca parle de l'histoire de naruto regarde les réponses sur ce ste : https://naruto.fandom.com/fr/wiki/Histoire_de_Naruto
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    print(generation)
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION1---")
    llm = ChatOllama(model=LLM_MODEL, format="json", temperature=0)

    prompt = PromptTemplate2(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | llm | JsonOutputParser()
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION2---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        print("---GRADE: SCORE---")
        print(grade)
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            #web_search = "Yes"
            web_search = "No"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

def route_question(state):

    llm = ChatOllama(model=LLM_MODEL, format="json", temperature=0)

    prompt = PromptTemplate2(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        french user question to a vectorstore or web search. Use the vectorstore for questions . You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
        no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    print("---decide_to_generate---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    llm = ChatOllama(model=LLM_MODEL, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate2(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()

    llm2 = ChatOllama(model=LLM_MODEL, format="json", temperature=0)

    # Prompt
    prompt2 = PromptTemplate2(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt2 | llm2 | JsonOutputParser()

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
def fixSyntaxQuestion(state):
    print("--fix syntaxe question --")
    llm = ChatOllama(model=LLM_MODEL, format="json", temperature=0)

    prompt = PromptTemplate2(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are french language expert.
        Rewrite the sentence by correcting the spelling and syntax errors except for URLs.
        If a word contains one or more capital letters, do not change it. 
        You must only rewrite the sentence word for word without rephrasing it
        Return the a JSON with a single key 'datasource' and 
        no premable or explanation. Question to rewrite: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    question = state["question"]

    question_fixer = prompt | llm | JsonOutputParser()
    result = question_fixer.invoke({"question": question})
    question = result["datasource"]
    print(question)
    print(result["datasource"])
    return {"question": question}


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]

workflow = StateGraph(GraphState)

# Define the nodes

#workflow.add_node("fixSyntaxQuestion", fixSyntaxQuestion)  # web search
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae

# Build graph
#workflow.set_entry_point("fixSyntaxQuestion")
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": END,
        #"not useful": "websearch",
    },
)

def queryQuestion(question):
    app = workflow.compile()
    inputs = {"question": question}
    for output in app.stream(inputs):
        print("looping")
        value = output
        #for key, value in output.items():
        #    pprint(f"Finished running: {key}:")
    #pprint(value["generation"])
    print(value)
    return value["generate"]["generation"]
    #return value["generation"]