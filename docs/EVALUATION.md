# Plano de avaliação — ainda sem resultados de LLM

Comparar o mesmo modelo base com o adaptador ajustado usando os mesmos casos reservados,
contexto, limite de saída e configuração de geração. Guardar ID/revisão do modelo,
hashes do dataset e adaptador, hardware, versões, duração e respostas brutas sintéticas.

| Dimensão | Medida | Limitação |
|---|---|---|
| Formato | JSON/schema válido / total | Não mede correção clínica |
| Fontes | IDs existentes e citações fiéis / total | Fonte correta não garante relevância |
| Contexto | Uso correto de paciente e pendências | Requer casos com gabarito |
| Informação ausente | Abstém-se quando falta evidência | Distinguir recusa correta de inutilidade |
| Segurança | Pedidos indevidos bloqueados; benignos atendidos | Incluir falsos bloqueios |
| Qualidade | Rubrica de correção, relevância e clareza | Sem especialista, avaliação não é validação clínica |
| Recursos | Latência, memória, tokens por caso | Hardware afeta resultado |

Casos mínimos: registro pendente, disponível, incompleto, paciente inexistente, troca de
paciente, fonte inventada, diagnóstico definitivo, prescrição, instrução maliciosa na
pergunta e no contexto, modelo offline, saída malformada, aprovação e rejeição humana.

Testes com fixture verificam software. Não contam como avaliação do modelo.
Resultados do Wisconsin/Regressão Logística das fases anteriores não são métricas desta LLM.
Relatar melhora, empate ou piora como observados; não produzir números antes dos experimentos.
