import langchain_core as lc
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma(persist_directory="../data/chroma_db", embedding_function=embeddings)

# nombre de documents à récupérer --> a changer pour un reranking
retriever = vectorstore.as_retriever(search_kwargs={"k": 14})

# Cette fonction prend les documents trouvés et les transforme en un seul bloc de texte
# J'inclut l'URL dans le texte pour que le LLM puisse la citer
def format_docs(docs):
    formatted_content = []
    for doc in docs:
        content = f"Source: {doc.metadata.get('source', 'Inconnue')}\nContenu: {doc.page_content}"
        formatted_content.append(content)
    return "\n\n---\n\n".join(formatted_content)

# LE PROMPT
template_text = '''
Tu es un assistant visant à guider les utilisateurs. Utilise le contexte ci-dessous.

Contexte (incluant les sources) :
{context}

Instructions :
1. Fais un court résumé.
2. Fournis le lien vers la documentation (trouvé dans le contexte sous "Source").
3. Réponse concise en français.

Format :
{{user: <question>
assistant: <réponse + lien>}}

Question de l'utilisateur : {user_input} 
'''

prompt = PromptTemplate(
    input_variables=["context", "user_input"],
    template=template_text
)

llm = ChatOllama(
    model="llama3",
    temperature=0,
    base_url="http://localhost:11434"
)

# CHAÎNE RAG
rag_chain = (
    # Préparation des entrées
    # "context" est rempli par : Question -> Retriever -> Documents -> format_docs
    # "user_input" est rempli par : Question (passée directement)
    {"context": retriever | format_docs, "user_input": RunnablePassthrough()} 
    | 
    # Le Prompt
    prompt 
    | 
    # Le LLM
    llm 
    | 
    # Nettoyage de la sortie (String pur)
    StrOutputParser()
)

question = "Quelle est la procédure relative à la location d'espace ?"
print(f"Question : {question}\n")

# TOP 20 DOCUMENTS
# Cela permet de voir si la réponse se trouve dans les documents récupérés
top_docs = vectorstore.similarity_search(question, k=20)
for i, doc in enumerate(top_docs):
    source = doc.metadata.get("source", "N/A")
    clean_content = doc.page_content.replace("\n", " ")[:150]
    print(f"[{i+1}] Source: {source}")
    print(f"      Extrait: {clean_content}...\n")

print("-" * 40 + "\n")

reponse = rag_chain.invoke(question)
print(reponse)