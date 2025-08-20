import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import os
from typing import List

# Importações atualizadas do LangChain e Pydantic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from pydantic import BaseModel, Field
from langchain import hub # Importação para usar prompts da comunidade

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# O "CÉREBRO" DO AGENTE: Função de identificação de arquivos
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
def identificar_arquivo(nome_arquivo, arquivo_bytes):
    """Identifica o tipo de arquivo com base em suas colunas ou nome."""
    try:
        df_preview = pd.read_excel(arquivo_bytes, nrows=5, engine='openpyxl')
        cols = {str(col).strip().upper() for col in df_preview.columns}
        
        if 'TITULO DO CARGO' in cols and 'DESC. SITUACAO' in cols: return "ATIVOS"
        if 'DIAS DE FÉRIAS' in cols: return "FERIAS"
        if 'DATA DEMISSÃO' in cols and 'COMUNICADO DE DESLIGAMENTO' in cols: return "DESLIGADOS"
        if any('DIAS UTEIS' in col for col in cols): return "DIAS_UTEIS"
        if 'VALOR' in cols and any('ESTADO' in col for col in cols): return "VALORES"
        if 'CADASTRO' in cols and 'VALOR' in cols: return "EXTERIOR"
        
        nome_upper = nome_arquivo.upper()
        if 'APRENDIZ' in nome_upper: return "APRENDIZ"
        if 'ESTÁGIO' in nome_upper or 'ESTAGIO' in nome_upper: return "ESTAGIO"
        if 'AFASTAMENTO' in nome_upper: return "AFASTAMENTOS"

        return "DESCONHECIDO"
    except Exception:
        return "INVALIDO"

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# AS FERRAMENTAS DO AGENTE (COM A LÓGICA ATUALIZADA)
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=

@tool
def aplicar_regras_de_exclusao(ativos_key: str, aprendiz_key: str = None, estagio_key: str = None, exterior_key: str = None, afastamentos_key: str = None) -> str:
    """
    Recebe as chaves dos DataFrames de funcionários ativos e de exclusão. Filtra a base de ativos
    e salva o resultado na memória com a chave 'ELEGIVEIS'. Retorna uma mensagem de sucesso.
    """
    dfs = st.session_state.get('dfs', {})
    if ativos_key not in dfs: return "Erro: Chave de DataFrame de ativos não encontrada na memória."
    
    df_ativos = dfs[ativos_key]
    df_elegiveis = df_ativos.copy()
    matriculas_para_excluir = set()

    if aprendiz_key and dfs.get(aprendiz_key) is not None: matriculas_para_excluir.update(dfs[aprendiz_key]['MATRICULA'].tolist())
    if estagio_key and dfs.get(estagio_key) is not None: matriculas_para_excluir.update(dfs[estagio_key]['MATRICULA'].tolist())
    if exterior_key and dfs.get(exterior_key) is not None: matriculas_para_excluir.update(dfs[exterior_key]['Cadastro'].tolist())
    if afastamentos_key and dfs.get(afastamentos_key) is not None: matriculas_para_excluir.update(dfs[afastamentos_key]['MATRICULA'].tolist())

    diretores = df_elegiveis[df_elegiveis['TITULO DO CARGO'].str.contains("DIRETOR", case=False)]
    if not diretores.empty: matriculas_para_excluir.update(diretores['MATRICULA'].tolist())
        
    df_elegiveis = df_elegiveis[~df_elegiveis['MATRICULA'].isin(list(matriculas_para_excluir))]
    
    st.session_state.dfs['ELEGIVEIS'] = df_elegiveis
    return "DataFrame de funcionários elegíveis foi criado e salvo com a chave 'ELEGIVEIS'."

@tool
def calcular_dias_de_beneficio(elegiveis_key: str, dias_uteis_key: str, ferias_key: str = None, desligados_key: str = None) -> str:
    """
    Calcula os dias de benefício a pagar. Usa o DataFrame de elegíveis (chave 'ELEGIVEIS') e outros
    DataFrames de regras. Salva o resultado na memória com a chave 'DIAS_CALCULADOS'.
    """
    dfs = st.session_state.get('dfs', {})
    if elegiveis_key not in dfs or dias_uteis_key not in dfs: return "Erro: Chaves de DataFrames essenciais não encontradas."

    df_elegiveis = dfs[elegiveis_key]
    df_dias_uteis = dfs[dias_uteis_key]
    df_ferias = dfs.get(ferias_key)
    df_desligados = dfs.get(desligados_key)

    df_dias_uteis_clean = df_dias_uteis.copy()
    df_dias_uteis_clean.columns = ['Sindicato', 'Dias_Uteis_Mes']
    df_calculo = pd.merge(df_elegiveis, df_dias_uteis_clean, on='Sindicato', how='left')

    if df_desligados is not None and not df_desligados.empty:
        df_desligados.columns = df_desligados.columns.str.strip()
        df_desligados['DATA DEMISSÃO'] = pd.to_datetime(df_desligados['DATA DEMISSÃO'])

    def calcular(colaborador):
        dias_a_pagar = colaborador.get('Dias_Uteis_Mes', 0)
        if df_ferias is not None:
            ferias_colaborador = df_ferias[df_ferias['MATRICULA'] == colaborador['MATRICULA']]
            if not ferias_colaborador.empty: dias_a_pagar -= ferias_colaborador['DIAS DE FÉRIAS'].iloc[0]
        if df_desligados is not None:
            desligado_colaborador = df_desligados[df_desligados['MATRICULA'] == colaborador['MATRICULA']]
            if not desligado_colaborador.empty:
                data_desligamento = desligado_colaborador['DATA DEMISSÃO'].iloc[0]
                if data_desligamento.day <= 15: return 0
                else: return np.busday_count('2025-05-01', data_desligamento.strftime('%Y-%m-%d'))
        return max(0, dias_a_pagar)

    df_calculo['Dias_A_Pagar'] = df_calculo.apply(calcular, axis=1)
    st.session_state.dfs['DIAS_CALCULADOS'] = df_calculo
    return "DataFrame com dias calculados foi criado e salvo com a chave 'DIAS_CALCULADOS'."

@tool
def calcular_valores_finais(dias_calculados_key: str, valores_key: str) -> str:
    """
    Calcula os valores monetários finais. Usa o DataFrame com dias calculados (chave 'DIAS_CALCULADOS').
    Esta é a última ferramenta a ser usada. Ela salva o resultado final na memória com a chave 'RESULTADO_FINAL'
    e retorna uma mensagem de sucesso.
    """
    dfs = st.session_state.get('dfs', {})
    if dias_calculados_key not in dfs or valores_key not in dfs: return "Erro: Chaves de DataFrames essenciais não encontradas."

    df_com_dias_pagos = dfs[dias_calculados_key]
    df_valores = dfs[valores_key]
    
    df_valores_clean = df_valores.copy()
    df_valores_clean.columns = ['Estado', 'VALOR']
    df_valores_clean.dropna(inplace=True)
    
    df_final = df_com_dias_pagos.copy()
    df_final['Sigla_Estado'] = df_final['Sindicato'].str.extract(r'\b(SP|RS|RJ|PR)\b')
    mapa_estados = {'SP': 'São Paulo', 'RS': 'Rio Grande do Sul', 'RJ': 'Rio de Janeiro', 'PR': 'Paraná'}
    df_final['Estado'] = df_final['Sigla_Estado'].map(mapa_estados)
    
    df_final = pd.merge(df_final, df_valores_clean, on='Estado', how='left')
    df_final['Valor_Total_VR'] = df_final['Dias_A_Pagar'] * df_final['VALOR']
    df_final['Custo_Empresa'] = df_final['Valor_Total_VR'] * 0.80
    df_final['Desconto_Profissional'] = df_final['Valor_Total_VR'] * 0.20
    
    df_layout_final = df_final[['MATRICULA', 'EMPRESA', 'Valor_Total_VR', 'Custo_Empresa', 'Desconto_Profissional']]
    
    st.session_state.dfs['RESULTADO_FINAL'] = df_layout_final
    return "Cálculo finalizado com sucesso. O resultado está pronto para ser exibido."

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# INTERFACE DO STREAMLIT E LÓGICA DO AGENTE
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=

st.set_page_config(layout="wide", page_title="Agente de IA para Análise de VR")
st.title("🧠 Agente de IA para Automação de VR/VA (com LangChain)")

# Inicialização da memória do agente
if 'dfs' not in st.session_state: st.session_state.dfs = {}
if 'arquivos_processados_log' not in st.session_state: st.session_state.arquivos_processados_log = {}

# Carregando a chave de API
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Chave de API do Google não configurada. Por favor, crie o arquivo .streamlit/secrets.toml com sua GOOGLE_API_KEY.")
    st.stop()

# Uploader de arquivos
uploaded_files = st.file_uploader("Carregue um arquivo .zip com as planilhas, ou envie os arquivos individualmente.", type=["xlsx", "zip"], accept_multiple_files=True, key="file_uploader")

if uploaded_files:
    st.session_state.dfs, st.session_state.arquivos_processados_log = {}, {}
    arquivos_para_processar = []
    for file in uploaded_files:
        if file.name.endswith('.zip'):
            with zipfile.ZipFile(file, 'r') as z:
                for filename in z.namelist():
                    if filename.endswith('.xlsx') and not filename.startswith('__MACOSX'):
                        arquivos_para_processar.append((filename, io.BytesIO(z.read(filename))))
        elif file.name.endswith('.xlsx'):
            arquivos_para_processar.append((file.name, file))

    for nome, arquivo in arquivos_para_processar:
        tipo = identificar_arquivo(nome, arquivo)
        st.session_state.arquivos_processados_log[nome] = tipo
        if tipo not in ["DESCONHECIDO", "INVALIDO"]:
            st.session_state.dfs[tipo] = pd.read_excel(arquivo, engine='openpyxl')

# Painel de Diagnóstico e Dashboard de Status
with st.expander("Clique aqui para ver o Painel de Diagnóstico do Agente"):
    if not st.session_state.arquivos_processados_log:
        st.write("Nenhum arquivo processado ainda.")
    else:
        st.write("O agente analisou os seguintes arquivos:")
        for nome, tipo in st.session_state.arquivos_processados_log.items():
            if tipo == "DESCONHECIDO": st.warning(f"**{nome}** -> ❓ Classificado como: **{tipo}**. O agente não reconheceu este arquivo.")
            elif tipo == "INVALIDO": st.error(f"**{nome}** -> ☠️ Classificado como: **{tipo}**. Não parece ser um arquivo Excel válido.")
            else: st.success(f"**{nome}** -> ✔️ Classificado como: **{tipo}**.")
st.header("Status dos Arquivos")
arquivos_obrigatorios = {"ATIVOS", "DIAS_UTEIS", "VALORES"}
arquivos_opcionais = { "APRENDIZ", "ESTAGIO", "EXTERIOR", "AFASTAMENTOS", "FERIAS", "DESLIGADOS"}
obrigatorios_ok = arquivos_obrigatorios.issubset(st.session_state.dfs.keys())
col1, col2 = st.columns(2)
col1.subheader("Obrigatórios")
for key in sorted(list(arquivos_obrigatorios)):
    if key in st.session_state.dfs: col1.success(f"✔️ {key}: Carregado")
    else: col1.error(f"❌ {key}: Faltando")
col2.subheader("Opcionais (Recomendados)")
for key in sorted(list(arquivos_opcionais)):
    if key in st.session_state.dfs: col2.success(f"✔️ {key}: Carregado")
    else: col2.warning(f"⚠️ {key}: Não fornecido")

# Lógica de Processamento com LangChain
if obrigatorios_ok:
    st.success("✅ Arquivos obrigatórios carregados. O agente está pronto para trabalhar.")
    if st.button("Executar Agente de IA"):
        
        with st.spinner("O agente está pensando e executando o plano..."):
            try:
                tools = [aplicar_regras_de_exclusao, calcular_dias_de_beneficio, calcular_valores_finais]
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
                
                # CORREÇÃO: Usar um prompt validado da comunidade (LangChain Hub)
                prompt = hub.pull("hwchase17/structured-chat-agent")
                
                agent = create_structured_chat_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
                
                st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                
                chaves_disponiveis = list(st.session_state.dfs.keys())
                task = f"""
                Sua tarefa é calcular o benefício de Vale Refeição. As chaves dos DataFrames disponíveis na memória são: {chaves_disponiveis}.
                Execute as seguintes etapas em ordem:
                1. Use a ferramenta 'aplicar_regras_de_exclusao' com as chaves apropriadas. A chave principal é 'ATIVOS'.
                2. Use a ferramenta 'calcular_dias_de_beneficio' com a chave do resultado da etapa 1 ('ELEGIVEIS').
                3. Use a ferramenta 'calcular_valores_finais' com a chave do resultado da etapa 2 ('DIAS_CALCULADOS').
                Confirme quando o processo estiver completo.
                """
                
                response = agent_executor.invoke({"input": task}, {"callbacks": [st_callback]})
                
                st.success("Agente concluiu a tarefa!")
                
                resultado_final_df = st.session_state.dfs.get('RESULTADO_FINAL')
                
                if resultado_final_df is not None:
                    st.dataframe(resultado_final_df)
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        resultado_final_df.to_excel(writer, index=False, sheet_name='VR_Final_IA')
                    st.download_button("📥 Baixar Planilha Final", output.getvalue(), "VR_MENSAL_calculado_pelo_agente_IA.xlsx")
                else:
                    st.error("O agente finalizou, mas não conseguiu gerar o DataFrame final. Verifique o log de pensamento acima.")

            except Exception as e:
                st.error(f"Ocorreu um erro durante a execução do agente: {e}")
else:
    st.info("Aguardando o carregamento dos arquivos obrigatórios para habilitar o agente.")
