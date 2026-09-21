"""Alert-volume and score-distribution drift monitoring (Phase 7A). No
database, MLflow, Docker, or network access -- operates on in-memory
windows of alert_evidence-shaped data; a real caller would build these
windows from `alert_evidence` rows (Phase 7B), fixture-only this phase.
"""
