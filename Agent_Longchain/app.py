import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LangChain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Assistente de Restaurante
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Você é um assistente de restaurante.

Funções:
- Receber pedidos
- Sugerir pratos
- Explicar ingredientes
- Confirmar escolhas
- Perguntar preferências

Primeiro você deve perguntar:
1. Quantas pessoas vão pedir?
2. Se já conhecem o cardápio?
3. Se querem recomendações da casa?
"""
    ),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# LLM 
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"  
)

chain = prompt | llm

store = {}
#obter o histórico da sessão
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

#interação com o assistente
def iniciar_assistente_restaurante():
    print("🍽️ Bem-vindo ao Assistente de Restaurante! Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Cliente: ")

        if pergunta.lower() in ["sair", "exit"]:
            print("👋 Obrigado pela visita!")
            break

        resposta = chain_with_history.invoke(
            {"input": pergunta},
            config={"configurable": {"session_id": "mesa01"}}
        )

        print("Assistente:", resposta.content, "\n")

# execução do assistente
if __name__ == "__main__":
    iniciar_assistente_restaurante()
