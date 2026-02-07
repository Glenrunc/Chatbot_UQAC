from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
template_text = '''
Tu es un assistant visant à guider les utilisateurs et à répondre directement à leurs questions. Pour ce faire tu utiliseras le contexte donnée ci-contre.
Contexte : {context}
Tu devras lui donner un cours résumé ainsi que directement le lien vers la documentation contenue dans ton contexte.
La réponse doit être concise et en français uniquement.
Utilise le format suivant :
{{user: <question de l'utilisateur>
assistant: <réponse concise en français avec lien vers la documentation>}}

Question de l'utilisateur : {user_input} 
'''

Prompt = PromptTemplate(
    input_variables = ["context", "user_input"],
    template = template_text
)

llm = ChatOllama(
    model="llama3",
    temperature=0,#invente rien
    max_tokens=512,#pour qu'il blablate pas trop
    base_url="http://localhost:11434"
)

chain = Prompt|llm # | c'est un pipe pour chainer le prompt au llm 

reponse = chain.invoke({
    "context": "LangChain permet de connecter des LLM à des sources de données.Doc: https://python.langchain.com",
    "user_input": "A quoi sert LangChain ?"
})

print(reponse.content)