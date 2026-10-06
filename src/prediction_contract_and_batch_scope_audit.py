"""Create a compact, reader-auditable prediction-time and batch-scope audit."""
from pathlib import Path
import json, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"

def main():
    z = np.load(DATA / "cell_panel.npz", allow_pickle=True)
    labels = np.asarray(z["labels"], str)
    source = np.asarray(z["source_row"], np.int64)
    # The source metadata are the only non-expression fields used by the model.
    # The control reference is frozen before model fitting and is therefore an
    # analysis-time object, not an endpoint-derived predictor.
    availability = {
        "identity_features": {"available": "before response measurement", "source": "perturbation identity/guide annotation", "used": True},
        "reactome_scores": {"available": "before response measurement", "source": "identity-derived annotation", "used": True},
        "component_indicators": {"available": "before response measurement", "source": "identity tokens", "used": True},
        "gemgroup": {"available": "at prediction request from assay metadata", "source": "observation-level gemgroup label", "used": True, "unseen_level_supported": False},
        "control_reference": {"available": "before fitting and scoring", "source": "frozen CONTROL cells with even source-row index", "used": "target centering only", "n_cells": 3207},
        "endpoint_expression": {"available": "only after response measurement", "used": False},
        "library_size": {"available": "endpoint measurement", "used": False},
        "guide_count": {"available": "endpoint measurement/quality table", "used": False},
        "response_derived_statistics": {"available": "after response measurement", "used": False},
    }
    audit = json.loads((OUT / "batch_availability_audit.json").read_text())
    intersections = np.asarray(audit["pairwise_identity_intersections"], int)
    batch_scope = {
        "protocol": "known-gemgroup within-Norman scope audit",
        "n_cells": int(len(labels)), "n_noncontrol_cells": int(np.sum(labels != "CONTROL")),
        "n_batches": int(audit["n_batches"]), "n_identities": 236,
        "minimum_pairwise_identity_intersection": int(intersections.min()),
        "all_batches_observed": True,
        "leave_one_batch_out_candidate": "undefined without a declared encoding for an unseen one-hot level",
        "valid_fallback_evaluated": False,
        "interpretation": "The candidate is estimable only when the queried gemgroup is one of the eight observed levels. This is a scope limitation, not evidence for transfer to a new batch.",
        "source_row_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
    }
    (OUT / "prediction_time_contract_audit.json").write_text(json.dumps({"availability": availability, "batch_scope": batch_scope}, indent=2))
    print(json.dumps({"path": str(OUT / "prediction_time_contract_audit.json"), "minimum_pairwise_identity_intersection": int(intersections.min())}, indent=2))

if __name__ == "__main__":
    main()
