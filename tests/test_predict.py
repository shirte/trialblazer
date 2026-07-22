from __future__ import annotations
import multiprocessing as mp
import os
from multiprocessing.queues import Queue
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from rdkit.Chem import Mol
from trialblazer import Trialblazer


def _predict(mols: list[Mol]) -> pd.DataFrame:
    with TemporaryDirectory() as tmpdir:
        # write molecules in temporary file
        with Path(f"{tmpdir}/test.smiles").open("w") as f:
            f.write("your_id,SMILES\n")
            f.writelines(f"id{i},{mol}\n" for i, mol in enumerate(mols))
        tb = Trialblazer(input_file=f"{tmpdir}/test.smiles")
        tb.run()
        return tb.get_dataframe()


def _predict_in_process(
    mols: list[Mol],
    queue: Queue[pd.DataFrame],
) -> None:
    queue.put(_predict(mols))


def _predict_with_seed(mols: list[Mol], seed: int) -> pd.DataFrame:
    context = mp.get_context("spawn")
    queue = context.Queue()
    previous_seed = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        process = context.Process(
            target=_predict_in_process,
            args=(mols, queue),
        )
        process.start()
    finally:
        if previous_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = previous_seed

    result = queue.get()
    process.join()
    assert process.exitcode == 0
    return result


def test_prediction_score_stays_consistent_across_runs():
    example_mol = (
        # one molecule written across two lines because of length
        "[H]OC1([H])C([H])(n2c([H])nc3c(N([H])[H])nc([H])nc32)OC([H])"
        "(C([H])([H])OP(=O)(O[H])O[H])C1([H])OP(=O)(O[H])O[H]"
    )

    # We run the prediction twice with different seeds to check if the
    # predictions stay identical.
    df_prediction1 = _predict_with_seed([example_mol], seed=1)
    df_prediction2 = _predict_with_seed([example_mol], seed=2)

    prediction1 = df_prediction1[df_prediction1["id"] == "id0"].iloc[0][
        "prediction"
    ]
    prediction2 = df_prediction2[df_prediction2["id"] == "id0"].iloc[0][
        "prediction"
    ]
    assert prediction1 == prediction2


def test_knn_score_stays_consistent_in_batch():
    batch_mols = [
        "[H]Oc1c([H])c(Cl)c([H])c([H])c1Cl",
        "[H]OC(=O)C(=O)C([H])(O[H])C([H])(O[H])C(=O)C([H])([H])O[H]",
        "[H]OC([H])([H])C1(C([H])([H])[H])OP(=O)(O[H])OP(=O)(O[H])OC([H])([H])C1([H])O[H]",
        "[H]OC(=O)C(O[H])=C([H])C([H])=C([H])C(=O)C([H])([H])[H]",
        "[H]OC(=O)C(=O)C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H]",
        "[H]OC(=O)C(=O)C([H])([H])C([H])([H])C([H])([H])C(=O)O[H]",
    ]

    example_mol = (
        # one molecule written across two lines because of length
        "[H]OC1([H])C([H])(n2c([H])nc3c(N([H])[H])nc([H])nc32)OC([H])"
        "(C([H])([H])OP(=O)(O[H])O[H])C1([H])OP(=O)(O[H])O[H]"
    )

    df1 = _predict([example_mol])
    df2 = _predict([*batch_mols, example_mol])

    # Check if the knn score stays the same when adding other molecules.
    # The example molecule has id "id0" in df1 and "id6" in df2.
    knn_score1 = df1[df1["id"] == "id0"].iloc[0]["3_nearest_neighbor_score"]
    knn_score2 = df2[df2["id"] == "id6"].iloc[0]["3_nearest_neighbor_score"]
    assert knn_score1 == knn_score2
