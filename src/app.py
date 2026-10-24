import streamlit as st

from rag_pipeline import LLM_PROVIDER, ROBOS, abrir_banco, gerar_resposta


st.set_page_config(page_title="Robotics RAG", page_icon="🤖")
st.title("🤖 Assistente de documentação dos robôs")
# a busca (embeddings + Chroma) é sempre local; só a geração da resposta usa o LLM escolhido
st.caption(f"Busca local (Chroma) + resposta gerada por {LLM_PROVIDER.capitalize()} a partir da documentação oficial.")

# seletor de robô: rótulo mostrado -> valor do filtro (None = todos)
OPCOES = {"Todos": None, **{nome: chave for chave, nome in ROBOS.items()}}
escolha = st.sidebar.selectbox("Robô", list(OPCOES))
if st.sidebar.button("Limpar conversa"):
    st.session_state.historico = []

# carrega o banco uma vez, já na abertura da página
with st.spinner("Carregando o banco vetorial..."):
    abrir_banco()


def mostrar_fontes(fontes):
    with st.expander(f"Fontes usadas ({len(fontes)} trechos)"):
        for doc, score in fontes:
            m = doc.metadata
            st.markdown(f"**{ROBOS[m['robot']]}** · `{m['source']}` · pág. {m['page_number']} · score {score:.2f}")
            st.text(doc.page_content)


# histórico fica no session_state para sobreviver às recargas do Streamlit
if "historico" not in st.session_state:
    st.session_state.historico = []

for msg in st.session_state.historico:
    with st.chat_message(msg["papel"]):
        st.markdown(msg["texto"])
        if msg.get("fontes"):
            mostrar_fontes(msg["fontes"])

pergunta = st.chat_input("Pergunte algo sobre o Go2, o Misty II ou o Alpha Mini")
if pergunta:
    st.session_state.historico.append({"papel": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando na documentação e gerando a resposta..."):
            # erros da API (sem chave, 429, sem internet) já voltam como aviso na resposta,
            # junto com os trechos que a busca encontrou, então o app não trava
            resposta, fontes = gerar_resposta(pergunta, OPCOES[escolha])
        st.markdown(resposta)
        if fontes:
            mostrar_fontes(fontes)

    st.session_state.historico.append({"papel": "assistant", "texto": resposta, "fontes": fontes})
