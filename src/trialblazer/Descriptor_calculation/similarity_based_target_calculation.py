import numpy as np
import pandas as pd
from FPSim2 import FPSim2Engine
from rdkit import Chem

def target_features_preprocess(
    chembl_data,
    preprocessed_chembl_data,
    outputFolder,
    remove_multi_components=False,
):
    if remove_multi_components:
        removed_multicom_chembl = preprocessed_chembl_data[
            ~preprocessed_chembl_data.id.str.contains("x")
        ]
    else:
        removed_multicom_chembl = preprocessed_chembl_data
    chembl_data = chembl_data.rename(columns={"chembl_id": "id"})
    chembl_data["id"] = chembl_data["id"].astype(str)
    merged_data = removed_multicom_chembl.merge(
        chembl_data[["id", "target_id"]],
        how="left",
        on="id",
    )
    merged_data["MolWithoutStereo"] = merged_data["preprocessedSmiles"].apply(
        Chem.MolFromSmiles,
    )
    merged_data["SmilesWithoutStereo"] = merged_data["MolWithoutStereo"].apply(
        lambda x: Chem.MolToSmiles(x, isomericSmiles=False),
    )
    target_label_smpl = merged_data[["target_id", "SmilesWithoutStereo"]].dropna(
        subset=["target_id"],
    )
    target_id_list, preprocessed_target_unique_smiles, fpe = generate_h5file(
        target_label_smpl,
        outputFolder,
    )
    return target_id_list, preprocessed_target_unique_smiles, fpe


def tanimoto_similarity_calculation(
    fpe,
    query_dataset,
    split_data=False,
    num_splits=None,
    similarity_threshold=0,
):
    results_append = []
    my_smi_append = []
    if split_data:
        if not num_splits:
            msg = "num_splits must be provided when split_data is True."
            raise ValueError(
                msg,
            )
        gen = np.array_split(query_dataset, num_splits)
        for _index, batch in generator_yield_gen(gen):
            for my_smi in generator_yield_smiles(batch):
                results = fpe.similarity(
                    my_smi,
                    similarity_threshold,
                    n_workers=1,
                )
                results_append.append(results)
                my_smi_append.append(my_smi)
    else:
        for my_smi in generator_yield_smiles(query_dataset):
            results = fpe.similarity(
                my_smi,
                similarity_threshold,
                n_workers=1,
            )
            results_append.append(results)
            my_smi_append.append(my_smi)
    return pd.DataFrame(
        {
            "my_smi": my_smi_append,
            "similarity_results": results_append,
        },
    )


def generate_h5file(preprocessed_target, outputFolder):
    target_id_list = list(preprocessed_target["target_id"].unique())
    target_id_list = [x for x in target_id_list if str(x) != "nan"]
    preprocessed_target_unique_smiles = (
        preprocessed_target.groupby("SmilesWithoutStereo")["target_id"]
        .apply(list)
        .reset_index()
    )
    preprocessed_target_unique_smiles["SmilesWithoutStereo"].tolist()

    fpe = FPSim2Engine(outputFolder)
    return target_id_list, preprocessed_target_unique_smiles, fpe


def generator_yield_gen(gen):
    yield from enumerate(gen)


def generator_yield_smiles(dataset):
    yield from dataset["SmilesForDropDu"]
