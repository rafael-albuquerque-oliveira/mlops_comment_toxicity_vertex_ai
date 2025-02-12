import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# URL da API hospedada (substitua pela URL correta do Flask)
API_URL = "https://https://toxic-comments-api-vertex-678895434688.us-central1.run.app/predict"

def prever_toxicidade(texto):
    """Envia um texto para a API Flask e retorna as previsões de toxicidade."""
    resposta = requests.post(API_URL, json={"text": texto})

    if resposta.status_code == 200:
        return resposta.json()
    else:
        return {"erro": "Falha ao obter previsão"}

# Configuração da Interface no Streamlit
st.title("🔍 Classificação de Comentários Tóxicos V1.0")
st.markdown("Digite um comentário para analisar seu nível de toxicidade.")

# Entrada do usuário
entrada_usuario = st.text_area("📝 Comentário:", "")

# Botão para análise
if st.button("📊 Analisar"):
    if entrada_usuario:
        resultado = prever_toxicidade(entrada_usuario)

        if "erro" in resultado:
            st.error(resultado["erro"])
        else:
            # Exibir textos original e traduzido
            st.markdown("### 📄 Resultados da Análise")
            st.markdown(f"**🗣️ Texto Original:** {resultado['original_text']}")
            st.markdown(f"**🌎 Tradução:** {resultado['translated_text']}")

            # Obter previsões e arredondar valores
            previsoes = resultado["prediction"]
            mapeamento_nomes = {
                "identity_hate": "Ódio Identitário",
                "insult": "Insulto",
                "obscene": "Obsceno",
                "severe_toxic": "Severamente Tóxico",
                "threat": "Ameaça",
                "toxic": "Tóxico"
            }
            
            previsoes_renomeadas = {mapeamento_nomes.get(k, k): round(v, 2) for k, v in previsoes.items()}

            # Exibir tabela de previsões
            st.markdown("### 🔢 Níveis de Toxicidade")
            df = pd.DataFrame.from_dict(previsoes_renomeadas, orient="index", columns=["Probabilidade"])
            df.reset_index(inplace=True)
            df.columns = ["Categoria", "Probabilidade"]
            st.dataframe(df)

            # Criar gráfico de barras
            st.markdown("### 📊 Visualização Gráfica")
            fig, ax = plt.subplots()
            ax.barh(df["Categoria"], df["Probabilidade"], color="crimson")
            ax.set_xlabel("Probabilidade de Toxicidade")
            ax.set_title("Classificação de Toxicidade do Comentário")

            # Exibir gráfico no Streamlit
            st.pyplot(fig)
    else:
        st.warning("⚠️ Por favor, insira um comentário antes de analisar.")
