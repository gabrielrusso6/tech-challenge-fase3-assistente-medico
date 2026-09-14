# Estado verificável — atualização de 14/09/2026

## Evidências anteriores confirmadas
- Repositório publicado: gabrielrusso6/tech-challenge-fase3-assistente-medico.
- Corrigido fechamento de SQLite no commit 6b7ad2b.
- CI desse commit passou nas seis combinações de OS/Python.
- Usuário executou pip check sem conflitos e todos os testes passaram no Windows 3.12.10.
- CLI fixture chegou a approved, registrando fontes, uma pendência e decisão humana.
- Exportação do seed original 6/3/3 executada pelo usuário.

## Incremento atual
- Corpus alinhado com prompt/runtime; gerador, manifest e checagem de integridade.
- Fine-tuning LoRA com máscara de prompt e artefatos de execução.
- Backend local para modelo base e adaptador PEFT; integração ao LangChain.
- Avaliação e comparação antes/depois, sem resultados preenchidos.
- Testes de corpus, tokenização/máscara, comparação e teste de treino LoRA minúsculo.
- CI ampliado. Consultar Actions para o resultado desta versão; não inferir sucesso pelo código.

## Ainda depende de execução e revisão
- Piloto e treino do modelo base no Mac M4 24 GB; memória e tempo reais.
- Avaliação das gerações com o adaptador treinado e casos independentes.
- Distribuir pesos e testar download/execução por terceiro.
- Expandir/curar conteúdo clínico: dataset atual prioriza revisão extrativa e segurança.
- Relatório técnico final, revisão de limitações e vídeo até 15 minutos.
- Docker e autenticação real não foram validados; identidade de revisor é demonstrativa.

Documentos de v0.1 registram o estado histórico e não devem ser usados como status atual.
Veja TRAINING.md e DATA_CARD_V2.md. Nenhum adaptador médico foi treinado nesta preparação.
