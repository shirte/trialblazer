import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def separate_similarity_results(
    results,
    target_id_list,
    preprocessed_target_unique_smiles,
    output_path_temp_save,
    training_data=False,
    start_index=0,
) -> None:
    default_value = -1
    for i, array in tqdm(
        generator_separate_results(results, start_index=start_index),
    ):
        smi = results.my_smi.iloc[i]
        temp_save = pd.DataFrame()
        dict_save = []
        smi_save = []
        sim_dict = dict.fromkeys(target_id_list, default_value)
        for value in array:
            if training_data:
                p = preprocessed_target_unique_smiles.iloc[value[0]]["target_id"]
            else:
                p = preprocessed_target_unique_smiles.iloc[value[0]]["target_id"]
            if any(item in target_id_list for item in p):
                for target in p:
                    sim_dict[target] = max(sim_dict[target], value[1])
        dict_save.append(sim_dict)
        smi_save.append(smi)
        temp_save["smi"] = smi_save
        temp_save["dict"] = dict_save
        temp_save.to_csv(
            Path(output_path_temp_save.name) / f"temp_save_{i}.csv",
            sep="|",
        )


def organize_similarity_results(
    target_id_list,
    output_path_temp_save,
    training_data=False,
):
    whole_df = read_results_from_file(output_path_temp_save)
    parsed_dicts = []
    for value in tqdm(whole_df["dict"]):
        parsed = eval(value, {"__builtins__": {}, "nan": float("nan"), "np": np})
        parsed_dicts.append(parsed)
    whole_df["dict_con"] = parsed_dicts
    all_lists = []
    target_cols_found = set()
    for value_dict in whole_df["dict_con"]:
        if training_data:
            values = list(value_dict.values())
            all_lists.append(values)
        else:
            values = []
            for target in target_id_list:
                if target in value_dict:
                    target_cols_found.add(target)
                    values.append(value_dict[target])
            all_lists.append(values)
    if training_data:
        whole_df[target_id_list] = all_lists
        return whole_df, target_id_list

    # ensure that the order of the columns is the same as the order of the
    # target_id_list
    ordered_target_cols = [
        target for target in target_id_list if target in target_cols_found
    ]
    whole_df[ordered_target_cols] = all_lists
    return whole_df, ordered_target_cols


def read_results_from_file(output_temp_save):
    whole_df = pd.DataFrame()
    for file in tqdm(os.listdir(output_temp_save.name)):
        filename = Path(output_temp_save.name) / file
        df1 = pd.read_csv(filename, sep="|")
        whole_df = pd.concat([whole_df, df1], ignore_index=True)
    return whole_df


def binarize_similarity_value(dataframe, target_list, threshold):
    dataframe[target_list] = dataframe[target_list].apply(
        lambda row: row.map(lambda x: threshold_binarize(x, threshold)),
    )
    return target_list, dataframe


def remove_tested_inactive_targets(
    binarized_inactive_target,
    binarized_query_targets,
):
    filtered_rows = []
    for i, targets in binarized_inactive_target.iterrows():
        inactive_targets = set(targets[targets == "1"].index)
        active_targets = binarized_query_targets.iloc[i, :]
        active_target_indices = set(
            active_targets[active_targets == "1"].index,
        )
        common_targets = inactive_targets & active_target_indices
        if common_targets:
            active_targets[list(common_targets)] = "0"
        filtered_rows.append(active_targets)
    filtered_df = pd.DataFrame(filtered_rows)
    filtered_df["SmilesForDropDu"] = binarized_query_targets["SmilesForDropDu"]
    target_list = filtered_df.filter(regex="^(CHEMBL*)").columns.tolist()
    return filtered_df, target_list


def remove_invariant_target(sanity_checked_dataframe, target_list):
    unique_counts = sanity_checked_dataframe[target_list].nunique()
    unique_counts[unique_counts == 1]
    target_binarize_list = unique_counts[unique_counts != 1].index.to_list()
    binarized_target_remain = sanity_checked_dataframe[target_binarize_list]
    binarized_target_remain["SmilesForDropDu"] = sanity_checked_dataframe[
        "SmilesForDropDu"
    ]
    return binarized_target_remain, target_binarize_list


def threshold_binarize(x, threshold) -> Optional[str]:
    if x >= threshold:
        return "1"
    if x < threshold:
        return "0"
    return None


def generator_separate_results(results, start_index=0):
    yield from enumerate(
        results.similarity_results[start_index:],
        start=start_index,
    )
