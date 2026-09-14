"""
dcforest: Random Forest rastreável + descoberta de Denial Constraints (LIMA)
aplicada a datasets da UCI.

Pipeline (ver `dcforest.pipeline.run_uci_experiment`):
    1. `dcforest.data_loading`   -> busca o dataset na UCI e prepara X/y
    2. `dcforest.forest`         -> treina a TraceableRandomForestClassifier
    3. `dcforest.evaluation`     -> calcula accuracy/precision/recall/F1
    4. `dcforest.dc_discovery`   -> roda o LIMA sobre x_train e gera as DCs
"""
