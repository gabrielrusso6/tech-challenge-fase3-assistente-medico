# Dados iniciais e perguntas exploratórias

## Origem e uso

Dados criados para este protótipo: 3 pacientes fictícios, 4 protocolos administrativos
simulados e 12 exemplos textuais de FAQ. Não vieram de hospital real nem dos 569 casos
do Wisconsin Dataset. Não há prontuários reais, nomes, telefones nem informação clínica
validada. Os protocolos SIM-* não são diretrizes médicas. Não há receitas ou laudos
clínicos completos nesta versão. O corpus ainda não atende sozinho ao requisito de treino.

## Perguntas que orientam a análise

1. Há cobertura equilibrada de limites, pendências, ausência de informação e revisão?
   Contagens por categoria são geradas em analysis.json; falta ampliar cobertura clínica.
2. Quais fontes concentram os exemplos? Há casos sem referência?
   Contagem por fonte e validação de referência não vazia. Conferência semântica é manual.
3. Quanto variam os comprimentos? Precisaremos truncar exemplos?
   Min/mediana/max de palavras. Tokens devem ser medidos com tokenizer do modelo escolhido.
4. Há duplicatas e vazamento entre splits?
   Normalização NFKC, espaços e caixa; rejeição de perguntas duplicadas e grupos cruzados.
   A separação inicial tem apenas 6/3/3 exemplos. Algumas intenções são semelhantes entre
   splits: esta amostra é um teste de formato, não um benchmark independente confiável.
5. Há dados pessoais?
   Apenas dados sintéticos aceitos; redator preventivo de email e CPF. Não detecta todo tipo
   de PII. Reutilizar com dados reais exigiria processo adicional e revisão.

## Expansão necessária

- Criar cenários independentes de paciente/protocolo; atribuir splits antes das paráfrases.
- Registrar origem, versão e licença de cada conteúdo médico externo.
- Revisor clínico ou limitação explícita; não chamar conteúdo gerado de protocolo hospitalar real.
- Evitar usar o conjunto de teste para escolher hiperparâmetros e prompts repetidamente.
- Medir similaridade semântica e auditar contaminação. Não usar multiplicação de templates
  para afirmar diversidade que não existe.
- Criar corpus alinhado ao formato final de resposta. Os FAQs textuais atuais não correspondem
  ao schema extrativo do runtime; NÃO treinar e conectar diretamente sem alinhar os formatos.
