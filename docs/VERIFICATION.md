# Evidência de verificação do incremento 0.1.0

Data: 10/09/2026. Ambiente disponível: Linux, Python 3.12.14.

## Executado de fato

| Verificação | Resultado |
|---|---|
| `PYTHONPATH=src python -m unittest discover -s tests -p test_core.py -v` | 15 testes aprovados |
| `PYTHONPATH=src python -m tc3.dataset` | 12 registros preparados; exportação JSONL concluída |
| `python -m compileall -q src tests` | Sucesso; valida apenas sintaxe/compilação |

A análise gerada está em `reports/seed_analysis.json`. Não representa resultado de LLM.

## Não executado

- Instalação limpa de LangChain/LangGraph/pytest: interrompida pela autorização de rede.
  O retorno do ambiente foi `network approval was cancelled before a decision was returned`.
  Não houve contorno da restrição. O ambiente não tinha esses pacotes pré-instalados.
- Testes `test_workflow.py`: implementados, mas dependem dos pacotes acima.
- CLI com grafo, integração Ollama, Docker e matriz GitHub Actions.
- Treinamento, exportação, carregamento de adaptador e avaliação clínica.

Não existe log de sucesso de CI, URL de repositório da Fase 3, modelo treinado ou resultado
de benchmark nesta versão. Não interpretar testes de SQLite como validação ponta a ponta.

## Próxima verificação necessária

Em máquina com internet, instalar conforme README, executar `pip check`, `pytest -v` e
`tc3 --mode fixture`. Confirmar aprovação e rejeição com intervenção humana. Depois validar
Ollama com modelo explícito e somente então iniciar o piloto de fine-tuning.
