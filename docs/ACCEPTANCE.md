# Critérios de aceite — Fase 3

Primeiro incremento, 10/09/2026. Prazo informado: 14/09/2026.
Autor: Gabriel Augusto Russo. Projeto individual.

| Requisito | Evidência prevista | Estado atual |
|---|---|---|
| Python modular | src/tc3 e pacote instalável | Código implementado; instalação externa pendente |
| Consultar dados estruturados | SQLite parametrizado, isolamento por paciente | Testado com biblioteca padrão |
| LangChain | Runnable para fixture e pipeline prompt/model/parser para Ollama | Implementado; execução pendente |
| LangGraph | load, generate, review, blocked; interrupt/resume | Implementado; testes de integração pendentes de execução |
| Revisão humana | Pausa real; aprovar/rejeitar; persistência da decisão | Implementado; grafo não executado neste ambiente |
| Logs e fontes | IDs, versões, hashes, motivo de bloqueio, decisão | SQLite testado; fluxo completo pendente |
| Fine-tuning real | Script, dataset curado, configuração, logs, adaptador | Pendente; nenhum treinamento realizado |
| Protocolos, FAQs e modelos de documentos médicos | Corpus com origem/licença e revisão | Apenas seed administrativo sintético; insuficiente para entrega |
| Avaliação antes/depois | Casos inéditos, mesmos critérios, resultados brutos | Plano definido; resultados inexistentes |
| CI | Matriz Linux/Mac/Windows e Python 3.11/3.12 | YAML criado; não publicado nem executado |
| Documentação | README, riscos, comandos, relatório final | README inicial; relatório final pendente |
| Vídeo até 15 min | Treino, caso, fluxo, logs, revisão | Pendente |

## Feedbacks anteriores incorporados

- Testes de comportamento e integração, não só inspeção de strings de prompt.
- CI por push/PR e artefatos de execução; sem inventar deploy que não existe.
- EDA guiada por perguntas: distribuição, lacunas, duplicação, fontes e separação.
- Análise de correlação será apropriada ao dado. Se integrar o classificador antigo,
  discutir correlações das medidas e limitações, sem confundir correlação com causalidade.
- Não reutilizar o fallback da Fase 2 que afirma baixa probabilidade em todos os casos.
- Se integrar a Fase 2, persistir imputador/scaler/modelo juntos e executar preprocessamento
  dentro dos folds de validação. A integração do classificador é opcional, não foi feita.

## Prioridades até a entrega

1. Resolver instalação, executar toda a suíte e publicar o repositório para ativar CI.
2. Validar modelo pequeno e estratégia de ajuste no Mac M4 24 GB, com piloto curto.
3. Expandir corpus com fontes médicas verificadas, FAQs, laudos/modelos de documentos,
   respostas de recusa e ausência; separar por cenários antes de gerar paráfrases.
4. Treinar, salvar adaptador e verificar carregamento em ambiente de inferência documentado.
5. Ampliar respostas além da extração conservadora desta base e avaliar base versus ajustado.
6. Congelar código/dados/modelo, documentar resultados reais e gravar o vídeo.

Não marcar fine-tuning como concluído porque existe integração com Ollama.
Não marcar segurança clínica como validada porque testes de schema passaram.
