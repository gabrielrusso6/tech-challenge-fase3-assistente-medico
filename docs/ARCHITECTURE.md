# Decisões iniciais

## D1 — Base portátil, treinamento separado

SQLite + Python + LangChain/LangGraph formam o runtime. Nenhum import de MLX/CUDA é exigido
para a fixture. Treinamento no Mac M4 24 GB será avaliado por piloto antes de escolher
modelo/quantização/adaptador. Não prometer que um adaptador de um backend funciona em outro.

## D2 — Falhar com estado explícito

Modelo indisponível ou saída inválida leva a blocked. Não usar uma resposta determinística
no lugar de uma resposta de LLM sem indicar isso. Fixture é selecionada explicitamente.

## D3 — Primeira saída conservadora

Schema contém apenas ação review_records e citações exatas. Fonte desconhecida, texto
adicional e ação de prescrição não passam. Isso restringe a geração, mas NÃO garante
relevância nem validação clínica. O modelo ainda precisa ser avaliado com perguntas reais.

## D4 — Banco controlado

SQL é escrita pela aplicação e recebe parâmetros. A LLM não recebe conexão de banco nem
executa comandos. Paciente é carregado por ID; fontes são versionadas e têm hash no log.

## D5 — Aprovação simulada

interrupt/resume do LangGraph exige decisão explícita com booleano e identificação de
revisor. A decisão persiste no SQLite. Não há login nem prova de credencial do revisor.
Checkpoint em memória serve à sessão CLI; persistência durável será avaliada se o escopo
exigir retomada entre processos. Nenhum exame ou tratamento é executado.

## D6 — CI antes da entrega

Workflow de testes em três sistemas, resultados XML e wheel como artefatos. Não treinar
modelo em cada push. Treinamento terá configuração e registro de execução separados.

## D7 — Continuidade sem dependência desnecessária

Manter tema de mama e referências às fases anteriores. Classificador numérico é integração
opcional, posterior aos requisitos obrigatórios de fine-tuning e assistente. Se integrado,
exigir as 30 features, pipeline completo persistido e nunca interpretar score como diagnóstico.
