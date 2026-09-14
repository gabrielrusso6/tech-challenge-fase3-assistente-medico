# Corpus v2 — seleção de evidências para revisão de casos de mama

**166 exemplos, 83 grupos, 9 categorias. Não equivale a 166 casos clínicos distintos.**

Origem: exemplos sintéticos construídos por código em tc3.corpus. São duas perguntas por
grupo, variando dados e ordem de fontes. As oito famílias administrativas têm dez grupos;
a família educativa tem três. Split antes da expansão: 98/34/34 exemplos em
treino/validação/teste, sem paciente ou grupo compartilhado.

## Conteúdo e curadoria
- Limites: não confirmar diagnóstico, não prescrever, não executar tratamentos.
- Registro: pendências, dados ausentes, revisão.
- Documentos: modelos de laudo, receita não emitível (campos clínicos em branco) e
  procedimento administrativo de revisão. Não são documentos emitidos por hospital real.
- Educação: referência breve de biópsia baseada no NCI, sem indicar conduta individual.
- Protocolos SIM-* são fictícios. REF-NCI-001 é síntese educativa, não protocolo hospitalar.
- Nenhum caso real foi importado; não houve revisão por médico.
- O sistema recebe contexto e seleciona citações completas. Treinar esse comportamento não
  ensina uma especialidade médica nem satisfaz sozinho toda a ambição clínica do enunciado.

## Proveniência
A síntese REF-NCI-001 foi escrita em português a partir de
[How Is Breast Cancer Diagnosed? — National Cancer Institute](https://www.cancer.gov/types/breast/diagnosis),
consultado em 14/09/2026. Não copiamos imagens nem reproduzimos o artigo.
Demais textos são templates sintéticos do projeto. Manter essas atribuições na entrega.
Os campos de medicamento/dose não devem ser preenchidos pelo modelo.

## Análise guiada por perguntas
1. Qual o equilíbrio? Exportar contagens por categoria/split. A categoria educativa tem só
   seis exemplos e é sub-representada; não concluir equivalência de cobertura.
2. Há vazamento? Verificar IDs, paciente/grupo e prompts/contextos duplicados.
   Isso não elimina similaridade semântica entre templates presentes nos três splits.
3. Qual o tamanho? Manifest inclui intervalo de caracteres. O treino registra tokens reais
   e recusa exemplos longos em vez de truncar a resposta.
4. As fontes são consistentes? Validar ID, citação completa, fonte de paciente e regra de
   segurança; rejeitar divergência entre mensagens de treinamento e prompt do runtime.
5. O dataset pode ser reproduzido? Gerador determinístico, hash do corpus e dos arquivos.
6. Há informação pessoal? Apenas sintético; bloqueio preventivo para email/CPF.
   Esses padrões não são anonimização universal e não autorizam importar prontuários reais.

## Limitações para o relatório
Repetição de templates pode inflar métricas. Incluir cenários manuais inéditos e revisão
qualitativa antes da entrega, mantendo-os separados dos ajustes. Não afirmar avaliação
clínica nem diversidade semântica a partir da contagem. As perguntas de diagnóstico/
prescrição recebem evidência da regra de bloqueio; o runtime continua sendo extrativo,
com decisão humana sobre tarefa administrativa. Falta conteúdo clínico curado mais amplo
para uma interpretação exigente do requisito de condutas e procedimentos médicos.
