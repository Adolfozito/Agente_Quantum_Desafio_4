import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile

# =====================================================================================
# O "CÉREBRO" DO AGENTE: Função de identificação
# =====================================================================================
def identificar_arquivo(nome_arquivo, arquivo_bytes):
    try:
        # Adicionamos engine='openpyxl' para garantir a leitura correta
        df_preview = pd.read_excel(arquivo_bytes, nrows=5, engine='openpyxl')
        
        cols = {str(col).strip().upper() for col in df_preview.columns}
        
        # Regras de identificação aprimoradas
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
    except Exception as e:
        # Para depuração, podemos imprimir o erro no terminal
        print(f"Erro ao ler o arquivo {nome_arquivo}: {e}")
        return "INVALIDO"
# =====================================================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# =====================================================================================
def processar_arquivos(arquivos_identificados):
    df_ativos = pd.read_excel(arquivos_identificados['ATIVOS'])
    df_elegiveis = df_ativos.copy()
    matriculas_para_excluir = set()
    if 'APRENDIZ' in arquivos_identificados:
        df_aprendiz = pd.read_excel(arquivos_identificados['APRENDIZ'])
        matriculas_para_excluir.update(df_aprendiz['MATRICULA'].tolist())
    if 'ESTAGIO' in arquivos_identificados:
        df_estagio = pd.read_excel(arquivos_identificados['ESTAGIO'])
        matriculas_para_excluir.update(df_estagio['MATRICULA'].tolist())
    if 'EXTERIOR' in arquivos_identificados:
        df_exterior = pd.read_excel(arquivos_identificados['EXTERIOR'])
        matriculas_para_excluir.update(df_exterior['Cadastro'].tolist())
    if 'AFASTAMENTOS' in arquivos_identificados:
        df_afastamentos = pd.read_excel(arquivos_identificados['AFASTAMENTOS'])
        matriculas_para_excluir.update(df_afastamentos['MATRICULA'].tolist())
    diretores = df_elegiveis[df_elegiveis['TITULO DO CARGO'].str.contains("DIRETOR", case=False)]
    if not diretores.empty:
        matriculas_para_excluir.update(diretores['MATRICULA'].tolist())
    df_elegiveis = df_elegiveis[~df_elegiveis['MATRICULA'].isin(list(matriculas_para_excluir))]
    df_dias_uteis = pd.read_excel(arquivos_identificados['DIAS_UTEIS'], skiprows=1)
    df_dias_uteis.columns = ['Sindicato', 'Dias_Uteis_Mes']
    df_calculo = pd.merge(df_elegiveis, df_dias_uteis, on='Sindicato', how='left')
    df_ferias = pd.read_excel(arquivos_identificados['FERIAS']) if 'FERIAS' in arquivos_identificados else pd.DataFrame(columns=['MATRICULA', 'DIAS DE FÉRIAS'])
    df_desligados = pd.read_excel(arquivos_identificados['DESLIGADOS']) if 'DESLIGADOS' in arquivos_identificados else pd.DataFrame(columns=['MATRICULA', 'DATA DEMISSÃO'])
    if not df_desligados.empty:
        df_desligados.columns = df_desligados.columns.str.strip()
        df_desligados['DATA DEMISSÃO'] = pd.to_datetime(df_desligados['DATA DEMISSÃO'])
    def calcular_dias_pagaveis(colaborador):
        dias_a_pagar = colaborador['Dias_Uteis_Mes']
        ferias_colaborador = df_ferias[df_ferias['MATRICULA'] == colaborador['MATRICULA']]
        if not ferias_colaborador.empty:
            dias_a_pagar -= ferias_colaborador['DIAS DE FÉRIAS'].iloc[0]
        desligado_colaborador = df_desligados[df_desligados['MATRICULA'] == colaborador['MATRICULA']]
        if not desligado_colaborador.empty:
            data_desligamento = desligado_colaborador['DATA DEMISSÃO'].iloc[0]
            if data_desligamento.day <= 15: return 0
            else: return np.busday_count('2025-05-01', data_desligamento.strftime('%Y-%m-%d'))
        return max(0, dias_a_pagar)
    df_calculo['Dias_A_Pagar'] = df_calculo.apply(calcular_dias_pagaveis, axis=1)
    df_valores = pd.read_excel(arquivos_identificados['VALORES'])
    df_valores.columns = ['Estado', 'VALOR']
    df_valores.dropna(inplace=True)
    df_calculo['Sigla_Estado'] = df_calculo['Sindicato'].str.extract(r'\b(SP|RS|RJ|PR)\b')
    mapa_estados = {'SP': 'São Paulo', 'RS': 'Rio Grande do Sul', 'RJ': 'Rio de Janeiro', 'PR': 'Paraná'}
    df_calculo['Estado'] = df_calculo['Sigla_Estado'].map(mapa_estados)
    df_final = pd.merge(df_calculo, df_valores, on='Estado', how='left')
    df_final['Valor_Total_VR'] = df_final['Dias_A_Pagar'] * df_final['VALOR']
    df_final['Custo_Empresa'] = df_final['Valor_Total_VR'] * 0.80
    df_final['Desconto_Profissional'] = df_final['Valor_Total_VR'] * 0.20
    df_layout_final = df_final[['MATRICULA', 'EMPRESA', 'Valor_Total_VR', 'Custo_Empresa', 'Desconto_Profissional']]
    return df_layout_final

# =====================================================================================
# INTERFACE DO USUÁRIO (STREAMLIT) - VERSÃO 4
# =====================================================================================
st.set_page_config(layout="wide")
st.title("🤖 Agente de Automação de VR/VA - Grupo Quantum - I2A2")

if 'arquivos_identificados' not in st.session_state:
    st.session_state.arquivos_identificados = {}
if 'arquivos_processados_log' not in st.session_state:
    st.session_state.arquivos_processados_log = {}

uploaded_files = st.file_uploader(
    "Carregue um arquivo .zip com todas as planilhas, ou envie os arquivos individualmente.",
    type=["xlsx", "zip"],
    accept_multiple_files=True,
    key="file_uploader" # Adiciona uma chave para resetar
)

if uploaded_files:
    # Resetar estado anterior ao carregar novos arquivos
    st.session_state.arquivos_identificados = {}
    st.session_state.arquivos_processados_log = {}

    arquivos_para_processar = []
    for file in uploaded_files:
        if file.type == "application/zip":
            with zipfile.ZipFile(file, 'r') as z:
                for filename in z.namelist():
                    if filename.endswith('.xlsx') and not filename.startswith('__MACOSX'):
                        arquivos_para_processar.append((filename, io.BytesIO(z.read(filename))))
        else:
            arquivos_para_processar.append((file.name, file))

    for nome, arquivo in arquivos_para_processar:
        tipo = identificar_arquivo(nome, arquivo)
        st.session_state.arquivos_processados_log[nome] = tipo # Log de diagnóstico
        if tipo not in ["DESCONHECIDO", "INVALIDO"]:
            st.session_state.arquivos_identificados[tipo] = arquivo

# --- NOVO: Painel de Diagnóstico ---
with st.expander("Clique aqui para ver o Painel de Diagnóstico do Agente"):
    if not st.session_state.arquivos_processados_log:
        st.write("Nenhum arquivo processado ainda.")
    else:
        st.write("O agente analisou os seguintes arquivos:")
        for nome, tipo in st.session_state.arquivos_processados_log.items():
            if tipo == "DESCONHECIDO":
                st.warning(f"**{nome}** -> ❓ Classificado como: **{tipo}**. O agente não reconheceu este arquivo.")
            elif tipo == "INVALIDO":
                st.error(f"**{nome}** -> ☠️ Classificado como: **{tipo}**. Não parece ser um arquivo Excel válido.")
            else:
                st.success(f"**{nome}** -> ✔️ Classificado como: **{tipo}**.")

# --- Dashboard de Status ---
st.header("Status dos Arquivos")
arquivos_obrigatorios = {"ATIVOS": "Base de ativos", "DIAS_UTEIS": "Tabela de dias úteis", "VALORES": "Tabela de valores"}
arquivos_opcionais = { "APRENDIZ": "Exclusão de aprendizes", "ESTAGIO": "Exclusão de estagiários", "EXTERIOR": "Exclusão de colab. no exterior", "AFASTAMENTOS": "Colaboradores afastados", "FERIAS": "Controle de férias", "DESLIGADOS": "Controle de desligamentos" }
obrigatorios_ok = all(key in st.session_state.arquivos_identificados for key in arquivos_obrigatorios)

col1, col2 = st.columns(2)
col1.subheader("Obrigatórios")
for key, desc in arquivos_obrigatorios.items():
    if key in st.session_state.arquivos_identificados:
        col1.success(f"✔️ {desc}: Carregado")
    else:
        col1.error(f"❌ {desc}: Faltando")
col2.subheader("Opcionais (Recomendados)")
for key, desc in arquivos_opcionais.items():
    if key in st.session_state.arquivos_identificados:
        col2.success(f"✔️ {desc}: Carregado")
    else:
        col2.warning(f"⚠️ {desc}: Não fornecido")

# --- Lógica de Processamento ---
if obrigatorios_ok:
    st.success("✅ Arquivos obrigatórios carregados. O agente está pronto para trabalhar.")
    if st.button("Executar Análise e Gerar Planilha"):
        with st.spinner("O agente está trabalhando..."):
            try:
                resultado_final = processar_arquivos(st.session_state.arquivos_identificados)
                st.toast('Análise concluída com sucesso!', icon='✅')
                st.success("Análise concluída!")
                st.dataframe(resultado_final.head(10))
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    resultado_final.to_excel(writer, index=False, sheet_name='VR_Final')
                st.download_button("📥 Baixar Planilha Final", output.getvalue(), "VR_MENSAL_calculado_pelo_agente_Quantum.xlsx")
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
else:
    st.info("Aguardando o carregamento dos arquivos obrigatórios para habilitar a análise.")