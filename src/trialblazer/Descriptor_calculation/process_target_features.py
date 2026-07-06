from rdkit import Chem
from rdkit.Chem import AllChem
from .process_similarity_results import binarize_similarity_value
from .process_similarity_results import organize_similarity_results
from .process_similarity_results import separate_similarity_results
from .similarity_based_target_calculation import (
    tanimoto_similarity_calculation,
)
from .similarity_based_target_calculation import target_features_preprocess
import pandas as pd


def process_target_features(
    output_path_temp_save,
    similarity_threshold=0.75,
    training_data=False,
    assay_median_value=None,
    target_prepocessed=None,
    output_h5=None,
    training_target_list=None,
    fpe=None,
    preprocessed_target_unique_smiles=None,
    query_data=None,
):
    if training_data:
        (
            target_id_list,
            preprocessed_target_unique_smiles,
            fpe,
        ) = target_features_preprocess(
            assay_median_value,
            target_prepocessed,
            output_h5,
        )
        data_for_similarity = query_data
        target_ids = target_id_list
    else:
        if (
            training_target_list is None
            or fpe is None
            or preprocessed_target_unique_smiles is None
            or query_data is None
        ):
            msg = "For test mode, training_target_list, fpe, preprocessed_target_unique_smiles, and query_data are required"
            raise ValueError(
                msg,
            )
        data_for_similarity = query_data
        target_ids = training_target_list

    results_whole = tanimoto_similarity_calculation(
        fpe,
        data_for_similarity,
        similarity_threshold=similarity_threshold,
    )  # kernal crush

    separate_similarity_results(
        results_whole,
        target_ids,
        preprocessed_target_unique_smiles,
        output_path_temp_save,
        training_data=training_data,
    )  # kernal crush

    features, target_list = organize_similarity_results(
        target_ids,
        output_path_temp_save,
        training_data=training_data,
    )

    features = features.rename(columns={"smi": "SmilesForDropDu"})

    if not training_data:
        pass
    binarize_list, binarized_target_remain = binarize_similarity_value(
        features,
        target_list,
        similarity_threshold,
    )

    if training_data:
        return (
            binarize_list,
            binarized_target_remain,
            preprocessed_target_unique_smiles,
            fpe,
        )
    return binarize_list, binarized_target_remain


def add_morgan_fingerprints(df, morgan_cols, smiles_column="SmilesForDropDu"):
    result_df = df.copy()
    result_df["Molecule"] = result_df[smiles_column].apply(Chem.MolFromSmiles)
    result_df["fp"] = result_df.Molecule.apply(get_morgan2)
    result_df[morgan_cols] = result_df["fp"].to_list()
    return result_df


def get_morgan2(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
