import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from typing import Optional
import re
from datetime import date, datetime
import holidays

# Importações atualizadas do LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# =====================================================================================
# O "CÉREBRO" DO AGENTE: Função de identificação de ficheiros
# =====================================================================================
def identificar_arquivo(nome_arquivo, arquivo_bytes):
    """Identifica o tipo de ficheiro com base nas suas colunas ou nome."""
    try:
        # Primeiro, tentar ler normalmente
        df_preview = pd.read_excel(arquivo_bytes, nrows=10, engine='openpyxl')
        
        # Se a primeira linha contém títulos, tentar ler a partir da linha 2
        if df_preview.iloc[0].astype(str).str.contains('BASE DIAS UTEIS', case=False, na=False).any():
            df_preview = pd.read_excel(arquivo_bytes, header=1, nrows=5, engine='openpyxl')
            if 'SINDICADO' in df_preview.columns or 'SINDICATO' in df_preview.columns:
                return "DIAS_UTEIS"
        
        cols = {str(col).strip().upper() for col in df_preview.columns}
        
        if 'TITULO DO CARGO' in cols and 'DESC. SITUACAO' in cols: return "ATIVOS"
        if 'DIAS DE FÉRIAS' in cols: return "FERIAS"
        if 'DATA DEMISSÃO' in cols and 'COMUNICADO DE DESLIGAMENTO' in cols: return "DESLIGADOS"
        if 'SINDICADO' in cols or ('SINDICATO' in cols and 'DIAS UTEIS' in cols): return "DIAS_UTEIS"
        if 'VALOR' in cols and any('ESTADO' in col for col in cols): return "VALORES"
        if 'CADASTRO' in cols and 'VALOR' in cols: return "EXTERIOR"
        
        nome_upper = nome_arquivo.upper()
        if 'APRENDIZ' in nome_upper: return "APRENDIZ"
        if 'ESTÁGIO' in nome_upper or 'ESTAGIO' in nome_upper: return "ESTAGIO"
        if 'AFASTAMENTO' in nome_upper: return "AFASTAMENTOS"
        if 'ADMISSÃO' in nome_upper: return "ADMITIDOS"
        if 'DIAS UTEIS' in nome_upper or 'BASE DIAS' in nome_upper: return "DIAS_UTEIS"

        return "DESCONHECIDO"
    except Exception as e:
        st.write(f"Erro ao identificar arquivo {nome_arquivo}: {e}")
        return "INVALIDO"

def identificar_casos_especiais(df, mes_referencia, ano_referencia):
    """
    Usa lógica de pandas para identificar rapidamente funcionários que
    precisam de uma análise mais detalhada da IA.
    Retorna o DataFrame com uma nova coluna 'Motivo_Analise_IA'.
    """
    df['Motivo_Analise_IA'] = ''
    
    # Regra 1: Pagamento zerado (excluindo demitidos na 1a quinzena)
    demitido_1a_quinzena = (pd.to_datetime(df.get('DATA DEMISSÃO')).dt.month == mes_referencia) & \
                           (pd.to_datetime(df.get('DATA DEMISSÃO')).dt.day <= 15) & \
                           (df.get('COMUNICADO DE DESLIGAMENTO', '').str.upper() == 'OK')
                           
    pagamento_zerado_injustificado = (df['Dias_A_Pagar'] == 0) & (~demitido_1a_quinzena)
    df.loc[pagamento_zerado_injustificado, 'Motivo_Analise_IA'] += 'Pagamento zerado; '

    # Regra 2: Admitidos no mês de referência
    admitidos_no_mes = (pd.to_datetime(df.get('Admissão')).dt.month == mes_referencia) & \
                       (pd.to_datetime(df.get('Admissão')).dt.year == ano_referencia)
    df.loc[admitidos_no_mes, 'Motivo_Analise_IA'] += 'Admissão recente; '
    
    # Regra 3: Desligados no mês de referência
    desligados_no_mes = (pd.to_datetime(df.get('DATA DEMISSÃO')).dt.month == mes_referencia) & \
                        (pd.to_datetime(df.get('DATA DEMISSÃO')).dt.year == ano_referencia)
    df.loc[desligados_no_mes, 'Motivo_Analise_IA'] += 'Desligamento recente; '
    
    # Regra 4: Dados importantes ausentes que afetam o cálculo
    if 'sindicato_ausente' in df.columns and df['sindicato_ausente'].any():
        df.loc[df['sindicato_ausente'], 'Motivo_Analise_IA'] += 'Sindicato ausente; '
        
    if 'VALOR DIÁRIO VR' in df.columns and (df['VALOR DIÁRIO VR'] == 0).any():
        df.loc[df['VALOR DIÁRIO VR'] == 0, 'Motivo_Analise_IA'] += 'Valor diário zerado; '

    return df

# =====================================================================================
# NOVA FUNÇÃO DEDICADA PARA CARREGAR DIAS ÚTEIS
# =====================================================================================
def carregar_dias_uteis(arquivo_bytes):
    """
    Carrega e processa a planilha de dias úteis de forma robusta.
    Retorna um dicionário com {Sindicato: Dias}.
    """
    try:
        # Tenta ler a partir da segunda linha (header=1), que é o caso mais comum
        df = pd.read_excel(arquivo_bytes, header=1, engine='openpyxl')
        
        # Se a primeira coluna não tiver um nome (por causa de um título na primeira linha)
        # os cabeçalhos podem estar errados. Tentamos ler de novo.
        if 'Unnamed: 0' in df.columns:
             df = pd.read_excel(arquivo_bytes, header=0, engine='openpyxl')

        # Padronizar nomes das colunas (remove espaços, põe em maiúsculas)
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Renomear SINDICADO para SINDICATO, se existir
        if 'SINDICADO' in df.columns:
            df.rename(columns={'SINDICADO': 'SINDICATO'}, inplace=True)

        # Verificação final e crítica das colunas necessárias
        if 'SINDICATO' not in df.columns or 'DIAS UTEIS' not in df.columns:
            raise ValueError("A planilha de dias úteis precisa ter as colunas 'SINDICATO' e 'DIAS UTEIS'.")

        # Limpa dados inválidos
        df.dropna(subset=['SINDICATO', 'DIAS UTEIS'], inplace=True)
        df['DIAS UTEIS'] = pd.to_numeric(df['DIAS UTEIS'], errors='coerce')
        df.dropna(subset=['DIAS UTEIS'], inplace=True)
        df['SINDICATO'] = df['SINDICATO'].astype(str)
        
        # Cria e retorna o dicionário
        dias_uteis_dict = dict(zip(df['SINDICATO'], df['DIAS UTEIS'].astype(int)))
        return dias_uteis_dict

    except Exception as e:
        st.error(f"Erro Crítico ao processar a planilha de dias úteis: {e}")
        return None

# =====================================================================================
# FUNÇÕES DE VALIDAÇÃO E TRATAMENTO DE DADOS
# =====================================================================================

def validar_e_corrigir_dados(df, tipo_arquivo):
    """Valida e corrige dados inconsistentes em um DataFrame."""
    df_limpo = df.copy()
    
    # O TRATAMENTO ESPECIAL PARA 'DIAS_UTEIS' FOI REMOVIDO DAQUI
    
    # Limpar nomes das colunas
    df_limpo.columns = [str(col).strip() for col in df_limpo.columns]
    
    # Garantir que MATRICULA seja numérica
    if 'MATRICULA' in df_limpo.columns:
        df_limpo['MATRICULA'] = pd.to_numeric(df_limpo['MATRICULA'], errors='coerce')
        df_limpo = df_limpo.dropna(subset=['MATRICULA'])
        df_limpo['MATRICULA'] = df_limpo['MATRICULA'].astype(int)
    elif 'Cadastro' in df_limpo.columns:  # Para arquivo EXTERIOR
        df_limpo['Cadastro'] = pd.to_numeric(df_limpo['Cadastro'], errors='coerce')
        df_limpo = df_limpo.dropna(subset=['Cadastro'])
        df_limpo['Cadastro'] = df_limpo['Cadastro'].astype(int)
    
    # Tratar datas
    colunas_data = ['Admissão', 'DATA DEMISSÃO', 'Data de Admissão', 'Data Demissão']
    for col in colunas_data:
        if col in df_limpo.columns:
            df_limpo[col] = pd.to_datetime(df_limpo[col], errors='coerce', dayfirst=True)
    
    # Tratar campos de texto
    colunas_texto = ['TITULO DO CARGO', 'DESC. SITUACAO', 'Sindicato', 'COMUNICADO DE DESLIGAMENTO']
    for col in colunas_texto:
        if col in df_limpo.columns:
            df_limpo[col] = df_limpo[col].astype(str).str.strip()
            df_limpo[col] = df_limpo[col].replace('nan', pd.NA)
    
    # Validar e corrigir dias de férias
    if 'DIAS DE FÉRIAS' in df_limpo.columns:
        df_limpo['DIAS DE FÉRIAS'] = pd.to_numeric(df_limpo['DIAS DE FÉRIAS'], errors='coerce')
        df_limpo['DIAS DE FÉRIAS'] = df_limpo['DIAS DE FÉRIAS'].clip(lower=0, upper=31)
        df_limpo['DIAS DE FÉRIAS'] = df_limpo['DIAS DE FÉRIAS'].fillna(0)
    
    # Validar valores monetários
    if 'VALOR' in df_limpo.columns:
        df_limpo['VALOR'] = pd.to_numeric(df_limpo['VALOR'], errors='coerce')
        df_limpo['VALOR'] = df_limpo['VALOR'].fillna(0)
    
    return df_limpo

def obter_feriados_brasil(ano, estado=None):
    """Retorna os feriados nacionais e estaduais do Brasil para um ano específico."""
    feriados_br = holidays.Brazil(years=ano, state=estado)
    return list(feriados_br.keys())

def consolidar_matriculas(dfs):
    """Consolida todas as matrículas de todos os arquivos em uma base única."""
    st.write("🔄 **Passo 1: Consolidando todas as matrículas...**")
    
    # Lista de arquivos que contêm matrículas
    arquivos_com_matricula = ["ATIVOS", "ADMITIDOS", "DESLIGADOS", "FERIAS", "APRENDIZ", "ESTAGIO", "AFASTAMENTOS"]
    
    # Validar e limpar dados de cada arquivo
    dfs_validados = {}
    for key, df in dfs.items():
        dfs_validados[key] = validar_e_corrigir_dados(df, key)
    
    # Coletar todas as matrículas únicas
    todas_matriculas = set()
    for key in arquivos_com_matricula:
        if key in dfs_validados and 'MATRICULA' in dfs_validados[key].columns:
            matriculas = dfs_validados[key]['MATRICULA'].dropna().unique()
            todas_matriculas.update(matriculas)
            st.write(f"   - {key}: {len(matriculas)} matrículas")
    
    # Também incluir matrículas do arquivo EXTERIOR (coluna Cadastro)
    if "EXTERIOR" in dfs_validados and 'Cadastro' in dfs_validados["EXTERIOR"].columns:
        matriculas_exterior = dfs_validados["EXTERIOR"]['Cadastro'].dropna().unique()
        todas_matriculas.update(matriculas_exterior)
        st.write(f"   - EXTERIOR: {len(matriculas_exterior)} matrículas")
    
    # Criar DataFrame consolidado
    master_df = pd.DataFrame(list(todas_matriculas), columns=['MATRICULA'])
    master_df['MATRICULA'] = master_df['MATRICULA'].astype(int)
    
    st.success(f"✅ **Consolidação concluída: {len(master_df)} matrículas únicas encontradas**")
    
    return master_df, dfs_validados

def aplicar_joins_sequenciais(master_df, dfs_validados):
    """
    Aplica joins sequenciais e AGORA TAMBÉM AGREGA NOTAS DE COLUNAS SEM CABEÇALHO.
    """
    st.write("🔗 **Passo 2: Aplicando joins sequenciais e capturando notas...**")
    
    prioridade_merge = ["ATIVOS", "ADMITIDOS", "DESLIGADOS", "FERIAS"]
    df_consolidado = master_df.copy()
    df_consolidado['Notas_Nao_Estruturadas'] = ''

    for arquivo in prioridade_merge:
        if arquivo in dfs_validados:
            df_fonte = dfs_validados[arquivo].copy()
            
            # Coletar notas de colunas "Unnamed"
            colunas_unnamed = [col for col in df_fonte.columns if 'unnamed' in str(col).lower()]
            if colunas_unnamed:
                st.write(f"   - 📝 Encontradas colunas de notas em '{arquivo}'")
                # Cria uma coluna de notas temporária no df_fonte
                df_fonte['temp_notas'] = df_fonte[colunas_unnamed].astype(str).agg(' | '.join, axis=1)
                # Limpa strings vazias ou de 'nan'
                df_fonte['temp_notas'] = df_fonte['temp_notas'].str.replace(r'(\s*\|\s*)*(nan|None)(\s*\|\s*)*', '', regex=True).str.strip(' |')
                
                # Junta as notas com a base consolidada
                df_consolidado = pd.merge(df_consolidado, df_fonte[['MATRICULA', 'temp_notas']], on='MATRICULA', how='left')
                df_consolidado['Notas_Nao_Estruturadas'] += df_consolidado['temp_notas'].fillna('') + ' '
                df_consolidado.drop(columns=['temp_notas'], inplace=True)

            # Evitar duplicação de colunas, exceto a chave e as já existentes
            colunas_para_merge = ['MATRICULA']
            for col in df_fonte.columns:
                if col not in df_consolidado.columns and col != 'temp_notas' and col not in colunas_unnamed:
                    colunas_para_merge.append(col)
            
            if len(colunas_para_merge) > 1:
                 df_consolidado = pd.merge(df_consolidado, df_fonte[colunas_para_merge], on='MATRICULA', how='left')
                 st.write(f"   - Merged com {arquivo}: {len(colunas_para_merge)-1} colunas adicionadas")
    
    df_consolidado['Notas_Nao_Estruturadas'] = df_consolidado['Notas_Nao_Estruturadas'].str.strip()
    st.success(f"✅ **Base consolidada criada com {len(df_consolidado)} registros**")
    return df_consolidado

# =====================================================================================
# FERRAMENTAS DO AGENTE DE IA
# =====================================================================================

@tool
def analisar_funcionario_ia(dados_funcionario: str, motivo_analise: str, notas_nao_estruturadas: Optional[str] = None) -> str:
    """
    Analisa dados de um funcionário que foi pré-selecionado como um caso especial.
    Usa o motivo da análise e notas não estruturadas para gerar observações inteligentes.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0,
            convert_system_message_to_human=True
        )
        
        prompt = f"""
        Você é um analista de RH especialista. Analise os dados de um funcionário que foi sinalizado como um caso especial.
        Seu objetivo é gerar uma observação CONCISA e útil para a planilha de Vale Refeição.

        **Motivo pelo qual este funcionário foi sinalizado para análise:**
        {motivo_analise}

        **Dados do Funcionário:**
        {dados_funcionario}
        """

        if notas_nao_estruturadas and notas_nao_estruturadas.strip():
            prompt += f"""
        **Notas Manuais Encontradas na Planilha (informação crucial e não estruturada):**
        "{notas_nao_estruturadas}"
        """

        prompt += """
        **Sua Tarefa:**
        Com base em TODAS as informações (motivo, dados e especialmente as notas manuais, se houver), gere uma observação curta (máximo 150 caracteres) que resuma a situação ou a ação necessária.
        - Se as notas manuais explicarem o motivo (ex: "Pagamento zerado" e nota "Funcionário de licença"), use essa informação.
        - Se não houver nada relevante a adicionar, retorne "SEM_OBSERVACAO".

        Exemplos de boas observações:
        - "Admissão em 15/05. Cálculo proporcional ok."
        - "Pagamento zerado devido a licença não remunerada (ver nota)."
        - "Desligado em 20/05. Comunicado OK. Pagamento proporcional."
        - "Sindicato não localizado, valor padrão SP aplicado."
        """
        
        response = llm.invoke(prompt)
        observacao = response.content.strip()
        
        if len(observacao) > 150:
            observacao = observacao[:147] + "..."
            
        return observacao if observacao != "SEM_OBSERVACAO" else ""
        
    except Exception as e:
        return f"Erro na análise IA: {str(e)[:50]}"

@tool
def processar_calculo_vr() -> str:
    """
    Executa o processo completo de cálculo do Vale Refeição com validação de dados,
    consolidação de matrículas, aplicação de regras de exclusão e cálculo detalhado.
    A análise por IA para casos especiais é opcional.
    """
    dfs = st.session_state.get('dfs', {})
    reference_date = st.session_state.get('reference_date', date(2025, 5, 1))
    
    # <-- MUDANÇA: Verifica se a análise de IA está habilitada
    ai_enabled = st.session_state.get('ai_analysis_enabled', False)
    
    st.write("🚀 **Iniciando processamento completo do Vale Refeição**")
    st.write(f"🤖 **Análise com IA para Casos Especiais:** {'ATIVADA' if ai_enabled else 'DESATIVADA'}")
    st.write("=" * 60)
    
    # --- PASSO 1: CONSOLIDAÇÃO DE MATRÍCULAS ---
    master_df, dfs_validados = consolidar_matriculas(dfs)
    
    # --- PASSO 2: JOINS SEQUENCIAIS E CAPTURA DE NOTAS ---
    df_consolidado = aplicar_joins_sequenciais(master_df, dfs_validados)
    
    # --- PASSO 3: APLICAÇÃO DAS REGRAS DE EXCLUSÃO (sem alterações) ---
    st.write("❌ **Passo 3: Aplicando regras de exclusão...**")
    # (O código do Passo 3 permanece o mesmo)
    matriculas_para_excluir = set()
    detalhes_exclusao = []
    if 'TITULO DO CARGO' in df_consolidado.columns:
        diretores = df_consolidado[df_consolidado['TITULO DO CARGO'].str.contains("DIRECTOR|DIRETOR", case=False, na=False)]
        if not diretores.empty:
            matriculas_diretores = set(diretores['MATRICULA'].tolist())
            matriculas_para_excluir.update(matriculas_diretores)
            detalhes_exclusao.append(f"Diretores: {len(matriculas_diretores)} matrículas")
    if 'DESC. SITUACAO' in df_consolidado.columns:
        situacoes_excluir = ["Atestado", "Auxílio Doença", "Licença Maternidade", "Licença Paternidade", "Afastamento", "Suspensão"]
        afastados = df_consolidado[df_consolidado['DESC. SITUACAO'].isin(situacoes_excluir)]
        if not afastados.empty:
            matriculas_afastados = set(afastados['MATRICULA'].tolist())
            matriculas_para_excluir.update(matriculas_afastados)
            detalhes_exclusao.append(f"Afastados: {len(matriculas_afastados)} matrículas")
    if "APRENDIZ" in dfs_validados and 'MATRICULA' in dfs_validados["APRENDIZ"].columns:
        matriculas_aprendiz = set(dfs_validados["APRENDIZ"]['MATRICULA'].dropna().unique())
        matriculas_para_excluir.update(matriculas_aprendiz)
        detalhes_exclusao.append(f"Aprendizes: {len(matriculas_aprendiz)} matrículas")
    if "ESTAGIO" in dfs_validados and 'MATRICULA' in dfs_validados["ESTAGIO"].columns:
        matriculas_estagio = set(dfs_validados["ESTAGIO"]['MATRICULA'].dropna().unique())
        matriculas_para_excluir.update(matriculas_estagio)
        detalhes_exclusao.append(f"Estagiários: {len(matriculas_estagio)} matrículas")
    if "EXTERIOR" in dfs_validados and 'Cadastro' in dfs_validados["EXTERIOR"].columns:
        matriculas_exterior = set(dfs_validados["EXTERIOR"]['Cadastro'].dropna().unique())
        matriculas_para_excluir.update(matriculas_exterior)
        detalhes_exclusao.append(f"Exterior: {len(matriculas_exterior)} matrículas")
    if "AFASTAMENTOS" in dfs_validados and 'MATRICULA' in dfs_validados["AFASTAMENTOS"].columns:
        matriculas_afastamentos = set(dfs_validados["AFASTAMENTOS"]['MATRICULA'].dropna().unique())
        matriculas_para_excluir.update(matriculas_afastamentos)
        detalhes_exclusao.append(f"Afastamentos: {len(matriculas_afastamentos)} matrículas")
    df_elegiveis = df_consolidado[~df_consolidado['MATRICULA'].isin(matriculas_para_excluir)].copy()
    st.write(f"   - Total de exclusões: {len(matriculas_para_excluir)} matrículas")
    for detalhe in detalhes_exclusao: st.write(f"     • {detalhe}")
    st.success(f"✅ **Restaram {len(df_elegiveis)} funcionários elegíveis**")

    # --- PASSO 4: CONFIGURAÇÃO DO PERÍODO E FERIADOS (sem alterações) ---
    st.write("📅 **Passo 4: Configurando período de referência e feriados...**")
    ano_referencia = reference_date.year
    mes_referencia = reference_date.month
    mes_inicio = pd.to_datetime(f'{ano_referencia}-{mes_referencia:02d}-01')
    mes_fim = mes_inicio + pd.offsets.MonthEnd(0)
    feriados_nacionais = obter_feriados_brasil(ano_referencia)
    feriados_sp = obter_feriados_brasil(ano_referencia, 'SP')
    feriados_rj = obter_feriados_brasil(ano_referencia, 'RJ')
    feriados_rs = obter_feriados_brasil(ano_referencia, 'RS')
    feriados_pr = obter_feriados_brasil(ano_referencia, 'PR')
    todos_feriados = list(set(feriados_nacionais + feriados_sp + feriados_rj + feriados_rs + feriados_pr))
    feriados_periodo = [f for f in todos_feriados if mes_inicio <= pd.to_datetime(f) <= mes_fim]
    st.write(f"   - Período: {mes_inicio.strftime('%d/%m/%Y')} a {mes_fim.strftime('%d/%m/%Y')}")
    st.write(f"   - Feriados no período: {len(feriados_periodo)}")

    # --- PASSO 5: USAR DIAS ÚTEIS DA PLANILHA BASE (lógica já opcional, sem alterações) ---
    # (O código do Passo 5 permanece o mesmo)
    usar_dias_uteis_base = False
    dias_uteis_por_sindicato = {}
    calculation_mode = st.session_state.get('calculation_mode', 'Calcular dinamicamente (Padrão)')
    if calculation_mode == "Usar planilha 'Base dias uteis.xlsx'" and "DIAS_UTEIS" in st.session_state.dfs:
        st.write("📋 **Passo 5: Processando planilha 'Base dias uteis.xlsx'...**")
        arquivo_dias_uteis_info = next((f for f in st.session_state.file_uploader if "dias uteis" in f.name.lower() or "base dias" in f.name.lower()), None)
        if arquivo_dias_uteis_info:
            dias_uteis_por_sindicato = carregar_dias_uteis(arquivo_dias_uteis_info)
            if dias_uteis_por_sindicato:
                usar_dias_uteis_base = True
                st.success(f"   - ✅ Planilha de dias úteis carregada com sucesso para {len(dias_uteis_por_sindicato)} sindicatos.")
            else:
                st.warning("⚠️ Planilha 'Base dias uteis.xlsx' não pôde ser processada. Usando cálculo dinâmico.")
        else:
             st.warning("⚠️ Planilha 'Base dias uteis.xlsx' não encontrada nos arquivos. Usando cálculo dinâmico.")
    else:
        st.write("📋 **Passo 5: Usando cálculo dinâmico de dias úteis.**")


    # --- PASSO 6: CÁLCULO DOS DIAS ---
    st.write("🧮 **Passo 6: Calculando dias de benefício...**")
    
    def calcular_dias_trabalhados(funcionario):
        """Calcula os dias trabalhados com a NOVA LÓGICA DE FÉRIAS."""
        
        # Se a base de dias úteis for usada, a lógica de férias é um simples desconto
        if usar_dias_uteis_base and 'Sindicato' in funcionario and pd.notna(funcionario['Sindicato']):
            sindicato_func = str(funcionario['Sindicato']).strip()
            for sind_base, dias_base in dias_uteis_por_sindicato.items():
                if sind_base.upper() in sindicato_func.upper():
                    # Lógica simples de desconto, pois a base já define o total
                    dias_ferias = funcionario.get('DIAS DE FÉRIAS', 0)
                    return max(0, dias_base - dias_ferias)
        
        # --- CÁLCULO DINÂMICO COM NOVA LÓGICA DE FÉRIAS ---
        data_admissao = pd.to_datetime(funcionario.get('Admissão', pd.NaT), errors='coerce')
        data_demissao = pd.to_datetime(funcionario.get('DATA DEMISSÃO', pd.NaT), errors='coerce')
        comunicado_ok = str(funcionario.get('COMUNICADO DE DESLIGAMENTO', '')).strip().upper() == 'OK'
        dias_ferias = funcionario.get('DIAS DE FÉRIAS', 0)
        if pd.isna(dias_ferias): dias_ferias = 0
        
        # Período base do mês
        inicio_calculo_base = mes_inicio
        fim_calculo = mes_fim
        
        # <-- NOVA LÓGICA DE FÉRIAS: Desconta dias corridos do início do mês
        if dias_ferias > 0:
            # O trabalho só começa após o término das férias
            # Adiciona (dias-1) ao dia 1 para achar o último dia de férias
            # e +1 para achar o primeiro dia de trabalho
            inicio_apos_ferias = mes_inicio + pd.Timedelta(days=dias_ferias)
            inicio_calculo_base = max(inicio_calculo_base, inicio_apos_ferias)

        # Ajustar por data de admissão (pega o maior entre início pós-férias e admissão)
        if pd.notna(data_admissao):
            inicio_calculo = max(inicio_calculo_base, data_admissao)
        else:
            inicio_calculo = inicio_calculo_base

        # Ajustar por data de demissão
        if pd.notna(data_demissao):
            if comunicado_ok and data_demissao.day <= 15 and data_demissao.month == mes_referencia:
                return 0  # Não paga nada
            fim_calculo = min(data_demissao, mes_fim)
        
        # Validação final do período
        if inicio_calculo > fim_calculo:
            return 0
        
        dias_uteis = np.busday_count(
            inicio_calculo.strftime('%Y-%m-%d'),
            (fim_calculo + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            holidays=[f.strftime('%Y-%m-%d') for f in feriados_periodo]
        )

        # A LÓGICA DE SUBTRAÇÃO NO FINAL FOI REMOVIDA
        return int(max(0, dias_uteis))

    df_elegiveis['Dias_A_Pagar'] = df_elegiveis.apply(calcular_dias_trabalhados, axis=1)

    # --- PASSO 7: ANÁLISE IA (AGORA OPCIONAL) ---
    # Este bloco inteiro só será executado se o toggle estiver ligado
    if ai_enabled:
        st.write("🤖 **Passo 7: Identificando casos especiais para análise com IA...**")
        df_elegiveis = identificar_casos_especiais(df_elegiveis, mes_referencia, ano_referencia)
        df_para_analise = df_elegiveis[df_elegiveis['Motivo_Analise_IA'] != ''].copy()
        total_a_analisar = len(df_para_analise)
        st.write(f"   - {total_a_analisar} de {len(df_elegiveis)} funcionários selecionados para análise detalhada.")
        
        observacoes_ia = {}
        if total_a_analisar > 0:
            progress_bar = st.progress(0, text=f"Analisando {total_a_analisar} casos especiais...")
            for idx, (matricula, funcionario) in enumerate(df_para_analise.iterrows()):
                dados_formatados = f"- Matrícula: {funcionario.get('MATRICULA', 'N/A')}\n- Cargo: {funcionario.get('TITULO DO CARGO', 'N/A')}\n- Situação: {funcionario.get('DESC. SITUACAO', 'N/A')}\n- Admissão: {funcionario.get('Admissão', 'N/A')}\n- Demissão: {funcionario.get('DATA DEMISSÃO', 'N/A')}\n- Dias Calculados: {funcionario.get('Dias_A_Pagar', 'N/A')}"
                motivo = funcionario['Motivo_Analise_IA']
                notas = funcionario.get('Notas_Nao_Estruturadas', '')
                try:
                    observacao = analisar_funcionario_ia.invoke({"dados_funcionario": dados_formatados, "motivo_analise": motivo, "notas_nao_estruturadas": notas})
                    observacoes_ia[matricula] = observacao
                except Exception:
                    observacoes_ia[matricula] = "Erro na chamada da IA."
                progress_bar.progress((idx + 1) / total_a_analisar, text=f"Analisando {idx+1}/{total_a_analisar}...")
        
        df_elegiveis['Observacao_IA'] = df_elegiveis.index.map(observacoes_ia).fillna('')
    else:
        # Se a IA estiver desligada, cria a coluna vazia para evitar erros
        st.write("🤖 **Passo 7: Análise com IA desativada.**")
        df_elegiveis['Observacao_IA'] = ''


    # --- PASSO 8 E 9 (sem alterações) ---
    # O código dos Passos 8 e 9 permanece o mesmo
    st.write("💰 **Passo 8: Calculando valores do benefício...**")
    if "VALORES" not in dfs_validados:
        st.error("❌ Arquivo de valores não encontrado!")
        return "Erro: Arquivo VALORES não encontrado"
    df_valores = dfs_validados["VALORES"].copy()
    df_valores.columns = ['Estado', 'VALOR DIÁRIO VR']
    df_valores = df_valores.dropna()
    def mapear_sindicato_estado(sindicato_text):
        if not isinstance(sindicato_text, str): return 'São Paulo'
        sindicato_upper = sindicato_text.upper()
        if 'SP' in sindicato_upper or 'SÃO PAULO' in sindicato_upper: return 'São Paulo'
        elif 'RJ' in sindicato_upper or 'RIO DE JANEIRO' in sindicato_upper: return 'Rio de Janeiro'
        elif 'RS' in sindicato_upper or 'RIO GRANDE DO SUL' in sindicato_upper: return 'Rio Grande do Sul'
        elif 'PR' in sindicato_upper or 'PARANÁ' in sindicato_upper: return 'Paraná'
        else: return 'São Paulo'
    df_final = df_elegiveis.copy()
    df_final['sindicato_ausente'] = df_final['Sindicato'].isna()
    df_final['Sindicato'] = df_final['Sindicato'].fillna('SINDPD SP - SIND.TRAB.EM PROC DADOS E EMPR.EMP...')
    df_final['Estado'] = df_final['Sindicato'].apply(mapear_sindicato_estado)
    df_final = pd.merge(df_final, df_valores, on='Estado', how='left')
    df_final['VALOR DIÁRIO VR'] = df_final['VALOR DIÁRIO VR'].fillna(0)
    df_final['TOTAL'] = df_final['Dias_A_Pagar'] * df_final['VALOR DIÁRIO VR']
    df_final['Custo empresa'] = df_final['TOTAL'] * 0.80
    df_final['Desconto profissional'] = df_final['TOTAL'] * 0.20
    st.write("📋 **Passo 9: Formatando resultado final...**")
    def gerar_observacao_completa(row):
        obs_padrao = []
        obs_ia = row.get('Observacao_IA', '')
        if row['sindicato_ausente']: obs_padrao.append('Sindicato não informado; atribuído SP por padrão')
        if row['VALOR DIÁRIO VR'] == 0: obs_padrao.append('Valor diário não encontrado para o estado')
        if row['Dias_A_Pagar'] == 0 and pd.notna(row.get('DATA DEMISSÃO')):
            if str(row.get('COMUNICADO DE DESLIGAMENTO', '')).strip().upper() == 'OK':
                obs_padrao.append('Desligado até dia 15 com comunicado OK')
        todas_obs = obs_padrao + ([obs_ia] if obs_ia and obs_ia.strip() else [])
        return '; '.join(todas_obs)
    df_final['OBS GERAL'] = df_final.apply(gerar_observacao_completa, axis=1)
    df_final['Competência'] = f"{mes_referencia:02d}/{ano_referencia}"
    if 'Admissão' in df_final.columns:
        df_final['Admissão'] = pd.to_datetime(df_final['Admissão'], errors='coerce').dt.strftime('%d/%m/%Y')
    colunas_finais = ['MATRICULA', 'Admissão', 'Sindicato', 'Competência', 'Dias_A_Pagar', 'VALOR DIÁRIO VR', 'TOTAL', 'Custo empresa', 'Desconto profissional', 'OBS GERAL']
    colunas_existentes = [col for col in colunas_finais if col in df_final.columns]
    layout_final = df_final[colunas_existentes].copy()
    mapeamento_colunas = {'MATRICULA': 'Matricula', 'Admissão': 'Admissão', 'Sindicato': 'Sindicato do Colaborador', 'Competência': 'Competência', 'Dias_A_Pagar': 'Dias', 'VALOR DIÁRIO VR': 'VALOR DIÁRIO VR', 'TOTAL': 'TOTAL', 'Custo empresa': 'Custo empresa', 'Desconto profissional': 'Desconto profissional', 'OBS GERAL': 'OBS GERAL'}
    layout_final = layout_final.rename(columns={k: v for k, v in mapeamento_colunas.items() if k in layout_final.columns})
    layout_final = layout_final.sort_values('Matricula').reset_index(drop=True)
    st.session_state.dfs['RESULTADO_FINAL'] = layout_final
    st.write("=" * 60)
    st.success(f"🎉 **PROCESSAMENTO CONCLUÍDO!**")
    st.write(f"📊 **Resumo final:**")
    st.write(f"   - Funcionários processados: {len(layout_final)}")
    st.write(f"   - Valor total calculado: R$ {layout_final['TOTAL'].sum():,.2f}")
    
    return f"✅ Cálculo finalizado! {len(layout_final)} funcionários processados, valor total: R$ {layout_final['TOTAL'].sum():,.2f}"

# =====================================================================================
# INTERFACE DO STREAMLIT E LÓGICA DO AGENTE
# =====================================================================================

st.set_page_config(layout="wide", page_title="Agente de IA para Análise de VR - Versão Melhorada")
st.title("🧠 Agente de IA Dinâmico para Automação de VR/VA (Versão Melhorada com Gemini-2.5-Flash)")

if 'dfs' not in st.session_state: 
    st.session_state.dfs = {}
if 'arquivos_processados_log' not in st.session_state: 
    st.session_state.arquivos_processados_log = {}

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("❌ Chave de API do Google não configurada nos secrets do Streamlit.")
    st.info("Configure a chave GOOGLE_API_KEY nos secrets para usar o modelo Gemini-2.5-Flash")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Defina o Mês de Referência")
    reference_date = st.date_input(
        "Mês e Ano para o Cálculo",
        value=date(2025, 5, 1),
        min_value=date(2020, 1, 1),
        max_value=date(2030, 12, 31),
        format="DD/MM/YYYY"
    )
    st.session_state.reference_date = reference_date
    
    st.subheader("2. Carregue os Ficheiros")
    uploaded_files = st.file_uploader(
        "Carregue um ficheiro .zip ou múltiplos ficheiros .xlsx",
        type=["xlsx", "zip"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    st.subheader("3. Escolha o Método de Cálculo")
    st.radio(
        "Como calcular os dias a pagar?",
        ('Calcular dinamicamente (Padrão)', "Usar planilha 'Base dias uteis.xlsx'"),
        key='calculation_mode',
        help="Se a planilha 'Base dias uteis.xlsx' for fornecida e esta opção selecionada, os dias dela terão prioridade."
    )

    st.subheader("4. Análise com IA")
    st.toggle(
        "Ativar Análise com IA para Casos Especiais",
        key='ai_analysis_enabled',
        value=False,  # Inicia desligado por padrão
        help="Se ativado, a IA analisará funcionários com dados inconsistentes ou situações atípicas. Pode ser lento e consumir cotas da API."
    )


    # Informações sobre arquivos esperados
    with st.expander("📋 Arquivos Esperados"):
        st.write("**Obrigatórios:**")
        st.write("- Funcionários ativos")
        st.write("- Funcionários admitidos")
        st.write("- Funcionários desligados")
        st.write("- Tabela de valores por estado")
        st.write("")
        st.write("**Opcionais:**")
        st.write("- Base dias uteis.xlsx (prioridade)")
        st.write("- Férias dos funcionários")
        st.write("- Lista de aprendizes")
        st.write("- Lista de estagiários")
        st.write("- Funcionários no exterior")
        st.write("- Afastamentos gerais")

if uploaded_files:
    st.session_state.dfs, st.session_state.arquivos_processados_log = {}, {}
    arquivos_para_processar = []
    
    # Processar uploads
    for file in uploaded_files:
        if file.name.endswith('.zip'):
            with zipfile.ZipFile(file, 'r') as z:
                for filename in z.namelist():
                    if filename.endswith('.xlsx') and not filename.startswith('__MACOSX'):
                        arquivos_para_processar.append((filename, io.BytesIO(z.read(filename))))
        elif file.name.endswith('.xlsx'):
            arquivos_para_processar.append((file.name, file))
    
    # Identificar e carregar arquivos
    for nome, arquivo in arquivos_para_processar:
        tipo = identificar_arquivo(nome, arquivo)
        st.session_state.arquivos_processados_log[nome] = tipo
        if tipo not in ["DESCONHECIDO", "INVALIDO"]:
            try:
                # Tratamento especial para arquivos com header na linha 2
                if tipo == "DIAS_UTEIS":
                    # Tentar carregar normalmente primeiro
                    df_test = pd.read_excel(arquivo, nrows=2, engine='openpyxl')
                    if df_test.iloc[0].astype(str).str.contains('BASE DIAS UTEIS', case=False, na=False).any():
                        # Header está na linha 2
                        st.session_state.dfs[tipo] = pd.read_excel(arquivo, header=1, engine='openpyxl')
                    else:
                        # Header está na linha 1
                        st.session_state.dfs[tipo] = pd.read_excel(arquivo, engine='openpyxl')
                else:
                    st.session_state.dfs[tipo] = pd.read_excel(arquivo, engine='openpyxl')
            except Exception as e:
                st.error(f"Erro ao carregar {nome}: {e}")

with col2:
    with st.expander("🔍 Painel de Diagnóstico do Agente"):
        if not st.session_state.arquivos_processados_log: 
            st.write("Nenhum ficheiro processado.")
        else:
            st.write("**Análise dos arquivos enviados:**")
            for nome, tipo in st.session_state.arquivos_processados_log.items():
                if tipo == "DESCONHECIDO": 
                    st.warning(f"**{nome}** -> ❓ Classificado como: **{tipo}**")
                elif tipo == "INVALIDO": 
                    st.error(f"**{nome}** -> ☠️ Classificado como: **{tipo}**")
                else: 
                    st.success(f"**{nome}** -> ✅ Classificado como: **{tipo}**")
                    
                    # Mostrar preview dos dados para DIAS_UTEIS
                    if tipo == "DIAS_UTEIS" and tipo in st.session_state.dfs:
                        df_preview = st.session_state.dfs[tipo].head(3)
                        st.write(f"Preview de {nome}:")
                        st.dataframe(df_preview, use_container_width=True)

    st.subheader("📊 Status dos Ficheiros Necessários")
    arquivos_obrigatorios = {"ATIVOS", "VALORES", "ADMITIDOS", "DESLIGADOS"}
    arquivos_opcionais = {"APRENDIZ", "ESTAGIO", "EXTERIOR", "AFASTAMENTOS", "FERIAS", "DIAS_UTEIS"}
    obrigatorios_ok = arquivos_obrigatorios.issubset(st.session_state.dfs.keys())
    
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown("**Obrigatórios**")
        for key in sorted(list(arquivos_obrigatorios)):
            if key in st.session_state.dfs: 
                count = len(st.session_state.dfs[key])
                st.success(f"✅ {key} ({count} registros)")
            else: 
                st.error(f"❌ {key}")
                
    with status_col2:
        st.markdown("**Opcionais**")
        for key in sorted(list(arquivos_opcionais)):
            if key in st.session_state.dfs: 
                count = len(st.session_state.dfs[key])
                st.success(f"✅ {key} ({count} registros)")
                if key == "DIAS_UTEIS":
                    st.info("🎯 Dias úteis serão priorizados!")
            else: 
                st.warning(f"⚠️ {key}")

# Seção principal de execução
if obrigatorios_ok:
    st.success("✅ **Ficheiros obrigatórios carregados. O agente está pronto para trabalhar!**")
    
    # Informações sobre o processamento
    st.info(f"📅 **Processando para competência: {reference_date.strftime('%m/%Y')}**")
    
    if "DIAS_UTEIS" in st.session_state.dfs:
        st.info("🎯 **Modo Especializado:** Utilizará a planilha 'Base dias uteis.xlsx' para cálculos precisos")
    
    if st.button("🚀 Executar Agente de IA", type="primary", use_container_width=True):
        with st.spinner("🤖 O agente Gemini-2.5-Flash está analisando e processando..."):
            try:
                # Configurar ferramentas e modelo
                tools = [processar_calculo_vr, analisar_funcionario_ia]
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    temperature=0,
                    convert_system_message_to_human=True
                )
                
                # Criar agente
                prompt = hub.pull("hwchase17/structured-chat-agent")
                agent = create_structured_chat_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(
                    agent=agent, 
                    tools=tools, 
                    verbose=True, 
                    handle_parsing_errors=True,
                    max_iterations=3
                )
                
                # Callback para mostrar progresso
                st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                
                # Tarefa para o agente
                task = f"""
                Execute o cálculo completo do Vale Refeição para a competência {reference_date.strftime('%m/%Y')}.
                Use a ferramenta 'processar_calculo_vr' para fazer todos os cálculos necessários.
                O sistema já tem todos os arquivos carregados e validados.
                Certifique-se de aplicar todas as regras de exclusão e usar a análise de IA quando apropriado.
                """
                
                # Executar agente
                response = agent_executor.invoke(
                    {"input": task}, 
                    {"callbacks": [st_callback]}
                )
                
                st.success("🎉 **Agente concluiu o processamento com sucesso!**")
                
                # Mostrar resultados
                resultado_final_df = st.session_state.dfs.get('RESULTADO_FINAL')
                
                if resultado_final_df is not None:
                    st.subheader("📋 Resultado Final")
                    
                    # Métricas principais
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total de Funcionários", len(resultado_final_df))
                    with col2:
                        st.metric("Valor Total VR", f"R$ {resultado_final_df['TOTAL'].sum():,.2f}")
                    with col3:
                        st.metric("Custo Empresa", f"R$ {resultado_final_df['Custo empresa'].sum():,.2f}")
                    with col4:
                        st.metric("Desconto Funcionários", f"R$ {resultado_final_df['Desconto profissional'].sum():,.2f}")
                    
                    # Tabela de resultados
                    st.dataframe(resultado_final_df, use_container_width=True)
                    
                    # Download da planilha
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        sheet_name = f"VR_{reference_date.strftime('%m_%Y')}"
                        resultado_final_df.to_excel(writer, index=False, sheet_name=sheet_name)
                        
                        # Formatação
                        workbook  = writer.book
                        worksheet = writer.sheets[sheet_name]
                        
                        # Formato monetário
                        money_format = workbook.add_format({'num_format': 'R$ #,##0.00'})
                        
                        # Aplicar formato nas colunas de valor
                        colunas_valor = ['F', 'G', 'H', 'I']  # VALOR DIÁRIO VR, TOTAL, Custo empresa, Desconto profissional
                        for col in colunas_valor:
                            worksheet.set_column(f'{col}:{col}', 18, money_format)
                        
                        # Ajustar largura das colunas
                        worksheet.set_column('A:A', 10)  # Matricula
                        worksheet.set_column('B:B', 12)  # Admissão
                        worksheet.set_column('C:C', 30)  # Sindicato
                        worksheet.set_column('D:D', 12)  # Competência
                        worksheet.set_column('E:E', 8)   # Dias
                        worksheet.set_column('J:J', 40)  # OBS GERAL

                    st.download_button(
                        label=f"📥 Baixar Planilha Final VR {reference_date.strftime('%m/%Y')}",
                        data=output.getvalue(),
                        file_name=f"VR_MENSAL_{reference_date.strftime('%m.%Y')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    # Análise adicional
                    with st.expander("📊 Análise Detalhada"):
                        st.write("**Distribuição por Estado:**")
                        if 'Sindicato do Colaborador' in resultado_final_df.columns:
                            # Mapear sindicatos para estados para análise
                            def extrair_estado(sindicato):
                                if pd.isna(sindicato):
                                    return 'Não informado'
                                sindicato_upper = str(sindicato).upper()
                                if 'SP' in sindicato_upper: return 'São Paulo'
                                elif 'RJ' in sindicato_upper: return 'Rio de Janeiro'
                                elif 'RS' in sindicato_upper: return 'Rio Grande do Sul'
                                elif 'PR' in sindicato_upper: return 'Paraná'
                                else: return 'Outros'
                            
                            resultado_final_df['Estado_Analise'] = resultado_final_df['Sindicato do Colaborador'].apply(extrair_estado)
                            analise_estado = resultado_final_df.groupby('Estado_Analise').agg({
                                'Matricula': 'count',
                                'TOTAL': 'sum'
                            }).rename(columns={'Matricula': 'Funcionários', 'TOTAL': 'Valor Total'})
                            
                            st.dataframe(analise_estado)
                        
                        st.write("**Funcionários com Observações Especiais:**")
                        funcionarios_com_obs = resultado_final_df[
                            resultado_final_df['OBS GERAL'].str.len() > 0
                        ][['Matricula', 'OBS GERAL']]
                        
                        if not funcionarios_com_obs.empty:
                            st.dataframe(funcionarios_com_obs)
                        else:
                            st.info("Nenhum funcionário com observações especiais.")
                
                else:
                    st.error("❌ O agente finalizou, mas não gerou o resultado final. Verifique os logs acima.")
                    
            except Exception as e:
                st.error(f"❌ **Erro durante a execução do agente:** {str(e)}")
                st.write("**Detalhes do erro:**")
                st.exception(e)
                
else:
    st.info("📋 **Aguardando o carregamento dos ficheiros obrigatórios para habilitar o agente.**")
    
    missing_files = arquivos_obrigatorios - set(st.session_state.dfs.keys())
    if missing_files:
        st.write("**Arquivos em falta:**")
        for missing in missing_files:
            st.write(f"- {missing}")

# Footer
st.markdown("---")
st.markdown("🤖 **Powered by Gemini-2.5-Flash** | 📊 **Análise Inteligente de VR/VA**")