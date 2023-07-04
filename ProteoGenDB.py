import os
import io
import re
import shutil
import sys
import time
import yaml
import warnings
import pandas as pd
import numpy as np
import multiprocessing
import requests
import logging
import json
import platform
from pyarrow import feather
from requests.exceptions import ConnectionError
from argparse import ArgumentParser
from datetime import datetime
from colorlog import ColoredFormatter
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from time import sleep
from tqdm import tqdm


global start_time

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(asctime)s %(message)s%(reset)s"
LOGFORMAT_WEL = "%(log_color)s%(message)s%(reset)s"

stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(ColoredFormatter(LOGFORMAT_WEL))

log = logging
log.getLogger().setLevel(logging.INFO)
log.getLogger("requests").setLevel(logging.WARNING)
log.getLogger().addHandler(stream)

# Suppress SettingWithCopyWarning & PerformanceWarning
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def print_welcome():
    if platform.system() == 'Windows':
        os.system("color")
    log.info(r"""
  _____           _              _____            _____  ____  
 |  __ \         | |            / ____|          |  __ \|  _ \ 
 | |__) | __ ___ | |_ ___  ___ | |  __  ___ _ __ | |  | | |_) |
 |  ___/ '__/ _ \| __/ _ \/ _ \| | |_ |/ _ \ '_ \| |  | |  _ < 
 | |   | | | (_) | ||  __/ (_) | |__| |  __/ | | | |__| | |_) |
 |_|   |_|  \___/ \__\___|\___/ \_____|\___|_| |_|_____/|____/                              
""")
    log.info(" Niko Pinter - https://github.com/npinter/ProteoGenDB")
    log.info(" v1.1.1 \n")


def multi_process(func, input_df, unit, *args):
    with multiprocessing.Manager() as manager:
        df_status = manager.dict()
        progress_bar_value = manager.dict()

        # set number of parallel jobs
        if config_yaml["num_jobs"] == 0:
            num_jobs = multiprocessing.cpu_count()
        else:
            num_jobs = config_yaml["num_jobs"]

        if config_yaml["num_job_chunks"] == 0 or config_yaml["num_job_chunks"] == num_jobs:
            num_job_chunks = num_jobs
        else:
            num_job_chunks = config_yaml["num_job_chunks"]

        # setup all jobs
        df_max = 0
        df_step = int(np.ceil(len(input_df) / num_jobs))

        jobs = []

        # append progress bar job
        progress_bar_process = multiprocessing.Process(target=progress_bar,
                                                       args=(len(input_df), progress_bar_value, unit))

        # append sliced df to job list
        for job_id in range(0, num_jobs):
            df_min = df_max
            df_max += df_step

            if df_max > len(input_df):
                df_max = len(input_df)

            if not df_min == df_max:
                df_slice = input_df[df_min:df_max]

                process = multiprocessing.Process(target=globals()[func],
                                                  args=(df_slice,
                                                        df_status,
                                                        job_id,
                                                        progress_bar_value,
                                                        args))
                jobs.append(process)

        progress_bar_process.start()

        for cj in range(0, num_jobs, num_job_chunks):
            chunk_jobs = jobs[cj:cj+num_job_chunks]

            # start jobs
            for j in chunk_jobs:
                j.start()

            jobs_done = False
            jobs_status_dict = {}
            while not jobs_done:
                for j in chunk_jobs:
                    if j.is_alive():
                        jobs_status_dict[j.name] = True
                    else:
                        jobs_status_dict[j.name] = False

                if not all(value for value in jobs_status_dict.values()):
                    jobs_done = True

            # validate if all jobs are finished
            for j in chunk_jobs:
                j.join()

        progress_bar_process.join()

        # job outputs need to be sorted by job id
        sorted_dict = {k: df_status[k] for k in sorted(df_status)}

        df_out = pd.concat(sorted_dict.values(), ignore_index=True)

        df_status.clear()

    return df_out


def progress_bar(max_value, p_bar_val, unit):
    for bar_val in tqdm(range(max_value),
                        unit=" {}".format(unit),
                        bar_format='{percentage:1.0f}%|{bar:10}{r_bar}'):
        p_bar_val_sum = sum(p_bar_val.values())
        if bar_val > p_bar_val_sum:
            sleep(1)


def read_yaml(yaml_file):
    with open(yaml_file) as yaml_stream:
        return yaml.safe_load(yaml_stream)


def read_fasta(fasta_file, database):
    # create fasta data frame
    with open(fasta_file) as fasta:
        ids = []
        gids = []
        seqs = []
        name = []
        desc = []
        fasta_df_temp = pd.DataFrame()

        if database == "uniprot":
            for seq_record in SeqIO.parse(fasta, "fasta"):
                ids.append([seq_record.id.split("|")[1] if "|" in seq_record.id else seq_record.id][0])
                seqs.append(seq_record.seq)
                name.append([seq_record.id.split("|")[2] if "|" in seq_record.id else seq_record.id][0])
                desc.append(seq_record.description)
            fasta_df_temp["Identifier"] = ids
            fasta_df_temp["Sequence"] = seqs
            fasta_df_temp["Name"] = name
            fasta_df_temp["Description"] = desc
        elif database == "galaxy":
            pids = []
            gids = []
            gname = []
            for seq_record in SeqIO.parse(fasta, "fasta"):
                header_split = seq_record.description.split("|")
                pids.append(header_split[1])  # Protein ID
                gids.append(header_split[4])  # Gene ID
                gname.append(header_split[5])  # Gene name
                seqs.append(seq_record.seq)  # SAAV protein sequence
                desc.append(header_split[6])  # Description

            # Get UniProtIDs from EnsemblIDs
            log.info("Requesting UniProtIDs with ENSEMBL ProteinIDs..")
            uniprot_lookup_df = get_uniprot_id(pids, ens_sub=True).rename({"FromID": "ProteinID",
                                                                           "ToID": "UniProtID"},
                                                                          axis=1)

            # fill dataframe
            fasta_df_temp["ProteinID"] = pids
            fasta_df_temp["UniProtID"] = multi_process("process_uniprot_ids", pids, "ids", uniprot_lookup_df)
            fasta_df_temp["GeneID"] = gids
            fasta_df_temp["GeneName"] = gname
            fasta_df_temp["Sequence"] = seqs
            fasta_df_temp["Description"] = desc

            # fill empty strings in UniProtID column
            fasta_df_temp["UniProtID"] = fasta_df_temp["UniProtID"].replace("", np.nan)

            # get missing UniProtIDs via the GeneID
            log.info("Requesting UniProtIDs with ENSEMBL GeneIDs..")
            fasta_df_temp_uniprot_mis = fasta_df_temp[fasta_df_temp["UniProtID"].isnull()]
            uniprot_lookup_genes = get_uniprot_id(fasta_df_temp_uniprot_mis["GeneID"].tolist(),
                                                  fmt_from="Ensembl",
                                                  ens_sub=True).rename({"FromID": "GeneID",
                                                                        "ToID": "UniProtID"},
                                                                       axis=1)
            uniprot_lookup_genes = uniprot_lookup_genes.drop_duplicates(subset=["GeneID"])

            # lookup UniProtIDs from GeneID df
            fasta_df_temp = fasta_df_temp.set_index("GeneID").combine_first(
                uniprot_lookup_genes.set_index("GeneID")
            ).reset_index()

            # fill empty strings in UniProtID column
            fasta_df_temp["UniProtID"] = fasta_df_temp["UniProtID"].replace(np.nan, "NoUniID")
        elif database == "ensembl_merge":
            for seq_record in SeqIO.parse(fasta, "fasta"):
                ids.append(seq_record.id)
                gids.append([seq_record.description.split("gene:")[1].split(" ", 1)[0]
                            if "gene:" in seq_record.description
                            else ""][0])
                seqs.append(seq_record.seq)
                name.append(seq_record.name.split(".")[0])

            fasta_df_temp["ProteinID"] = ids
            fasta_df_temp["StableID"] = name
            fasta_df_temp["GeneID"] = gids
            fasta_df_temp["Sequence"] = seqs
        elif database == "ensembl":
            for seq_record in SeqIO.parse(fasta, "fasta"):
                ids.append(seq_record.id)
                name.append(seq_record.name.split(".")[0])
                gids.append([seq_record.description.split()[1]
                             if "ENSP" in seq_record.description and seq_record.description != ""
                             else seq_record.description][0])
                seqs.append(seq_record.seq)

            fasta_df_temp["ProteinID"] = ids
            fasta_df_temp["StableID"] = name
            fasta_df_temp["GeneID"] = gids
            fasta_df_temp["Sequence"] = seqs

    return fasta_df_temp


def subset_fasta_db(fasta_db, exp_mat):
    fasta_df_temp = read_fasta(fasta_db, "uniprot")

    ex_mat_sep = pd.read_csv(exp_mat, sep=None, iterator=True, engine="python")
    ex_mat_sep_det = ex_mat_sep._engine.data.dialect.delimiter
    ex_mat_df_temp = pd.read_csv(exp_mat, sep=ex_mat_sep_det)
    id_subset = ex_mat_df_temp.iloc[:, 0]

    fasta_subset = fasta_df_temp[fasta_df_temp["Identifier"].isin(id_subset.tolist())]

    return fasta_subset


def write_fasta(df, out_path, time_str, fasta_type, in_src):
    with open(os.path.join(out_path, "{}_{}_{}.fasta".format(time_str, fasta_type, in_src)), "w") as fasta_file:
        SeqIO.write(df, fasta_file, "fasta")


def read_tso(tso_path):
    with open(tso_path, "r") as tso_input:
        tso_split = tso_input.read().split("[Small Variants]")[1][3:]
        tso_df = pd.read_csv(io.StringIO(tso_split), sep="\t")

    # extract NM ID
    tso_df["ProteinID"] = tso_df["P-Dot Notation"].str.extract(r"(NP_\d+\.\d+).*")

    # extract p. notation
    # substitution: Ala102Val |[A-z]{3}
    # fs mutation: Ser378PhefsTer6 |[A-z]{8}\d+
    # inframe insertion: Ala767_Val769dup |_[A-z]{3}\d+dup
    # stop gained: Arg661Ter |Ter
    # inframe deletion: Pro129del |del
    tso_df["VariantPos3"] = tso_df["P-Dot Notation"].str.extract(r"p\.\(([A-z]{3}\d+(?:[A-z]{3}|Ter|del))\)")

    aa_dict = {
        'Cys': 'C', 'Asp': 'D', 'Ser': 'S', 'Gln': 'Q', 'Lys': 'K',
        'Ile': 'I', 'Pro': 'P', 'Thr': 'T', 'Phe': 'F', 'Asn': 'N',
        'Gly': 'G', 'His': 'H', 'Leu': 'L', 'Arg': 'R', 'Trp': 'W',
        'Ala': 'A', 'Val': 'V', 'Glu': 'E', 'Tyr': 'Y', 'Met': 'M',
        'Ter': '*', 'del': '-'
    }
    aa_regex = "|".join(aa_dict.keys())

    # translate into one-letter code and change to list
    tso_df["VariantPos"] = tso_df["VariantPos3"].str.replace(
        aa_regex, lambda x: aa_dict[x.group()], regex=True
    ).apply(lambda x: [x])

    # drop na
    tso_df = tso_df[~tso_df["VariantPos3"].isna()].reset_index(drop=True)

    # get protein sequence from NCBI as SeqRecord
    tso_df["Sequence"] = multi_process("get_ncbi_fasta", tso_df["ProteinID"].tolist(), "ids")

    return tso_df


def get_uniprot_id(ids, fmt_from="Ensembl_Protein", fmt_to="UniProtKB", ens_sub=False, split_str="_"):
    response = []
    base_url = "https://rest.uniprot.org/idmapping"
    entrez_ids = [entrez_id.split(split_str)[0] for entrez_id in ids]

    ids_per_batch = 100000

    # remove duplicates from list
    entrez_ids = list(dict.fromkeys(entrez_ids))

    # add ENSEMBL IDs current subversion
    if ens_sub:
        if entrez_ids[0][:4] == "ENSP":
            ensembl_lookup = read_fasta(config_yaml["fasta_ensembl"], "ensembl").drop(
                ["Sequence", "GeneID"],
                axis=1)
            ensembl_lookup = ensembl_lookup[~ensembl_lookup["StableID"].duplicated(keep="last")]
            entrez_ids_df = pd.DataFrame(entrez_ids, columns=["StableID"])
            entrez_ids_match = entrez_ids_df.merge(ensembl_lookup,
                                                   on='StableID',
                                                   how='left')
            entrez_ids_match = entrez_ids_match[~entrez_ids_match["ProteinID"].isnull()]
            entrez_ids = entrez_ids_match["ProteinID"].tolist()
        elif entrez_ids[0][:4] == "ENSG":
            ensembl_lookup = read_fasta(config_yaml["fasta_ensembl"], "ensembl").drop(
                ["Sequence", "ProteinID", "StableID"],
                axis=1)
            ensembl_lookup["StableID"] = ensembl_lookup["GeneID"].str.split(".").str[0]
            ensembl_lookup = ensembl_lookup[~ensembl_lookup["StableID"].duplicated(keep="last")]
            entrez_ids_df = pd.DataFrame(entrez_ids, columns=["StableID"])
            entrez_ids_match = entrez_ids_df.merge(ensembl_lookup,
                                                   on='StableID',
                                                   how='left')
            entrez_ids_match = entrez_ids_match[~entrez_ids_match["GeneID"].isnull()]
            entrez_ids = entrez_ids_match["GeneID"].tolist()

    # send REST API request
    for entrez_id in range(0, len(entrez_ids), ids_per_batch):
        entrez_ids_tmp = ",".join(entrez_ids[entrez_id:entrez_id + ids_per_batch])
        data = {
            'from': fmt_from,
            'to': fmt_to,
            'ids': entrez_ids_tmp
        }
        key_error = True
        while key_error:
            try:
                response.append(json.loads(requests.post("{}/run".format(base_url), data=data).text)["jobId"])
                key_error = False
            except KeyError:
                log.error("UniProt connection error.. retry..")
                sleep(10)

    result_df = pd.DataFrame()
    uni_data = None

    # get REST API results
    err_json = True
    err_result = True

    for uni_job in response:
        while err_result:
            while err_json:
                try:
                    uni_data = json.loads(requests.get("{}/stream/{}".format(base_url, uni_job)).text)
                    err_json = False
                except json.decoder.JSONDecodeError:
                    log.error("Request failed.. retry..")
                    sleep(1)
            try:
                result_df = pd.concat([result_df, pd.DataFrame(uni_data["results"])])
                err_result = False
            except KeyError:
                log.error("Waiting for results..")
                err_json = True
                sleep(5)

    result_df = result_df.rename(columns={"from": "FromID", "to": "ToID"}).drop_duplicates(subset=["FromID"],
                                                                                           keep="first")

    log.info("{}/{} ({}%) ENSEMBL IDs were mapped to a UniProt ID".format(
        len(result_df),
        len(entrez_ids),
        round(len(result_df)/(len(entrez_ids)/100), 2))
    )

    return result_df


def get_ncbi_fasta(pids, df_status, jid, p_bar_val, *_):
    base_url = "https://www.ncbi.nlm.nih.gov/search/api/download-sequence/?db=protein&id={pid}&filename={pid}"
    seq_records = []
    for i_pid, pid in enumerate(pids):
        # ToDo: check if temp database entry exists if not get from NCBI and write to disk
        prot_seq_rec = SeqIO.read(io.StringIO(requests.get(base_url.format(pid=pid)).text), "fasta").seq
        seq_records.append(prot_seq_rec)
        p_bar_val[jid] = i_pid

    seq_records_df = pd.Series(seq_records)

    df_status[jid] = seq_records_df


def get_annotation_data(pids, df_status, jid, p_bar_val, *args):
    api_name = args[0][0]
    api_url = args[0][1]
    annot_folder = args[0][2]

    api_annot_path = os.path.join(annot_folder)

    # load Feather storage
    feather_filename = "{}_data.feather".format(api_name)
    feather_path = os.path.join(api_annot_path, feather_filename)

    feather_temp_filename = "{}_data_{}.temp.feather".format(api_name, jid)
    feather_temp_path = os.path.join(api_annot_path, feather_temp_filename)

    try:
        data = pd.read_feather(feather_path, columns=["pid", "data"])
    except FileNotFoundError:
        data = pd.DataFrame(columns=["pid", "data"])
        feather.write_feather(data, feather_path)

    data_new = pd.DataFrame(columns=["pid", "data"])
    result_data = []
    feather_save = False

    # get pid data from Feather or update it via API
    for i_pid, pid in enumerate(pids):
        if pid in data["pid"].values:
            result = data[data['pid'] == pid]
            result_data.append(result["data"].iloc[0])
        else:
            con_err = True
            while con_err:
                try:
                    api_data = requests.get(api_url.format(pid=pid)).json()
                    con_err = False
                except ConnectionError:
                    log.error("UniProt REST API issue.. retry..")
                    sleep(1)
            data_new = pd.concat([data_new, pd.DataFrame({'pid': [pid], 'data': [api_data]})], ignore_index=True)
            result_data.append(api_data)
            feather_save = True

        p_bar_val[jid] = i_pid

    result_data_df = pd.Series(result_data)

    if feather_save:
        feather.write_feather(data_new, feather_temp_path)

    df_status[jid] = result_data_df


def get_variant_info(pids, df_status, jid, p_bar_val, *_):
    var_df = pd.DataFrame()

    for i_pid, pid in enumerate(pids):
        variant_data = pid

        if "features" in variant_data:
            if variant_data["features"] is not None:
                for variant in variant_data["features"]:
                    var_dict = {}
                    if "ftId" in variant:
                        var_dict["DL_ftID"] = variant["ftId"]
                    else:
                        var_dict["DL_ftID"] = None
                    var_dict["DL_UniProtID"] = pid["accession"]
                    if "cosmic curated" in str(variant["xrefs"]):
                        for xref in variant["xrefs"]:
                            if xref["name"] == "cosmic curated":
                                var_dict["DL_CosmicID"] = xref["id"]
                                break
                    else:
                        var_dict["DL_CosmicID"] = None

                    if "ClinVar" in str(variant["xrefs"]):
                        for xref in variant["xrefs"]:
                            if xref["name"] == "ClinVar":
                                var_dict["DL_ClinVarID"] = xref["id"]
                                break
                    else:
                        var_dict["DL_ClinVarID"] = None

                    if "mutatedType" in variant:
                        var_seq_mutatedtype = variant["mutatedType"]
                    else:
                        var_seq_mutatedtype = None

                    var_dict["DL_VariantPosStr"] = "{}{}{}".format([variant["wildType"] if "wildType" in variant else "?"][0],
                                                                   variant["begin"],
                                                                   var_seq_mutatedtype)

                    var_dict["DL_VariantPos"] = [var_dict["DL_VariantPosStr"]]

                    if variant["begin"] != variant["end"]:
                        var_dict["DL_VariantPosStr"] = ""
                        var_dict["DL_VariantPos"] = []

                    if "populationFrequencies" in variant:
                        if not variant["populationFrequencies"] is None:
                            for pf in variant["populationFrequencies"]:
                                if pf["populationName"] == "MAF":
                                    var_dict["DL_MAF"] = pf["frequency"]
                                    break
                    else:
                        var_dict["DL_MAF"] = None

                    var_dict["DL_sig_patho"] = np.nan
                    var_dict["DL_sig_likely_patho"] = np.nan
                    var_dict["DL_sig_likely_benign"] = np.nan
                    var_dict["DL_sig_benign"] = np.nan
                    var_dict["DL_sig_uncertain"] = np.nan
                    var_dict["DL_sig_conflict"] = np.nan

                    if "clinicalSignificances" in variant:
                        if not variant["clinicalSignificances"] is None:
                            for cs in variant["clinicalSignificances"]:
                                if cs["type"] == "Likely pathogenic":
                                    var_dict["DL_sig_likely_patho"] = ";".join(cs["sources"])
                                elif cs["type"] == "Pathogenic":
                                    var_dict["DL_sig_patho"] = ";".join(cs["sources"])
                                elif cs["type"] == "Variant of uncertain significance":
                                    var_dict["DL_sig_uncertain"] = ";".join(cs["sources"])
                                elif cs["type"] == "Benign":
                                    var_dict["DL_sig_benign"] = ";".join(cs["sources"])
                                elif cs["type"] == "Likely benign":
                                    var_dict["DL_sig_likely_benign"] = ";".join(cs["sources"])
                                elif cs["type"] == "Conflicting interpretations of pathogenicity":
                                    var_dict["DL_sig_conflict"] = ";".join(cs["sources"])

                    if "association" in variant:
                        if not variant["association"] is None:
                            var_dict["DL_disease_association"] = "1"
                        else:
                            var_dict["DL_disease_association"] = "0"
                    else:
                        var_dict["DL_disease_association"] = "0"

                    var_df = pd.concat([var_df, pd.DataFrame(pd.Series(var_dict)).transpose()])

                p_bar_val[jid] = i_pid

    df_status[jid] = var_df


def map_variant_info(dl_df, df_status, jid, p_bar_val, *args):
    input_df = args[0][0]
    min_pep_len = args[0][1]
    max_pep_len = args[0][2]

    input_df_temp = pd.DataFrame()

    for index, var in dl_df.iterrows():
        prot_temp_df = input_df.loc[
            (input_df.UniProtID.isin([var.DL_UniProtID]))]
        var_temp_df = prot_temp_df.loc[(prot_temp_df.VariantPos.isin(var.DL_VariantPos))]

        if len(var_temp_df) > 0 and var_temp_df.VarSeqCleave.values[0] is not None:
            if min_pep_len <= len(var_temp_df.VarSeqCleave.values[0]) <= max_pep_len:
                var_temp_df = pd.concat([var_temp_df.reset_index(drop=True).drop(
                    ["VarSeq"], axis=1),
                    pd.DataFrame(var).transpose().reset_index(drop=True)],
                    axis=1
                )

                input_df_temp = pd.concat([input_df_temp, var_temp_df])

        p_bar_val[jid] = index

    df_status[jid] = input_df_temp


def annotate_variant_info(input_df, left_join, annot_data_path, min_pep_len, max_pep_len):
    log.info(f"Collect disease information for {left_join}s..")

    # drop Sequence (probably lowers RAM usage)
    input_df.drop(["Sequence"], axis=1, inplace=True)

    input_df = input_df.explode('VariantPos')

    def get_var_seq_cleave(row):
        pos = row['VariantPos']
        cleave_dict = row['VarSeqCleave'][0]
        return cleave_dict.get(pos, '')

    # apply function to each row
    input_df['VarSeq'] = input_df.apply(get_var_seq_cleave, axis=1)
    input_df['VarSeqCleave'] = input_df.apply(get_var_seq_cleave, axis=1)

    # drop duplicates
    input_df.drop_duplicates(subset=["UniProtID", "VariantPos", "VarSeqCleave"], keep="first", inplace=True)

    # load Feather storage
    feather_filename = "{}_data.feather".format("ebi")
    feather_path = os.path.join(annot_data_path, feather_filename)

    # create annotation folder
    if not os.path.exists(annot_data_path):
        os.mkdir(annot_data_path)

    try:
        feather_data = pd.read_feather(feather_path)
    except FileNotFoundError:
        feather_data = pd.DataFrame(columns=["pid", "data"])
        feather.write_feather(feather_data, feather_path)

    annotate_data = multi_process("get_annotation_data",
                                  input_df.UniProtID.drop_duplicates().to_list(),
                                  left_join,
                                  "ebi",
                                  "https://www.ebi.ac.uk/proteins/api/variation/{pid}",
                                  annot_data_path)

    # concatenate requested variant data and combine with Feather database
    temp_feather_files = []

    for file_name in os.listdir(annot_data_path):
        if file_name.endswith(".temp.feather"):
            temp_feather_files.append(os.path.join(annot_data_path, file_name))

    # save new annotations to Feather database
    if len(temp_feather_files) > 0:
        temp_feather_dfs = pd.DataFrame()
        for temp_feather_file in temp_feather_files:
            temp_feather_dfs = pd.concat([temp_feather_dfs, pd.read_feather(temp_feather_file)])
            os.remove(temp_feather_file)

        feather_data = pd.concat([feather_data, temp_feather_dfs])

        feather.write_feather(feather_data, feather_path)

    log.info(f"Extract variant information for {left_join}s..")
    disease_lookup_df = multi_process("get_variant_info",
                                      annotate_data,
                                      left_join)

    # remove duplicates with UniProtID and VarPos
    disease_lookup_df = disease_lookup_df.drop_duplicates(
        subset=["DL_UniProtID", "DL_VariantPosStr"])

    log.info(f"Map variant information..")
    input_df = multi_process("map_variant_info",
                             disease_lookup_df,
                             "Variants",
                             input_df,
                             min_pep_len,
                             max_pep_len)

    # keep only columns for FASTA header
    input_df_drop_list = ["DL_UniProtID", "DL_CosmicID", "DL_VariantPos", "DL_VariantPosStr"]
    input_df.drop(input_df_drop_list, axis=1, inplace=True)

    # replace nan with NA
    input_df.replace({np.nan: "NA"}, inplace=True)

    # rename columns
    input_df.rename(columns=lambda x: x.replace("DL_", ""), inplace=True)

    return input_df


def process_uniprot_ids(ids, df_status, jid, p_bar_val, *args):
    lookup_df = args[0][0].reset_index(drop=True)

    uids = []
    out_df = pd.DataFrame()

    lookup_df["ProteinID"] = lookup_df.ProteinID.str[:15]

    for i_pid, pid in enumerate(ids):
        uid = lookup_df["UniProtID"].to_numpy()[lookup_df["ProteinID"].to_numpy() == pid[:15]]
        uids.append([uid[0] if len(uid) > 0 else ""][0])
        p_bar_val[jid] = i_pid

    out_df["UniProtID"] = uids

    df_status[jid] = out_df["UniProtID"]


def annotate_saavs(fasta_df, cfg):
    log.info("Annotate SAAVs from Galaxy Workflow..")

    # extract SAAVs position from ProteinID
    fasta_df_temp = fasta_df
    fasta_df_temp["VariantPos"] = fasta_df["ProteinID"].str.split(pat="_").str[-1]

    # logic if ProteinID == VariantPos --> isoform/None
    fasta_df_temp.loc[(fasta_df_temp.VariantPos == fasta_df_temp.ProteinID), 'VariantPos'] = None

    # split iso and saavs
    fasta_df_var = fasta_df_temp[fasta_df_temp.VariantPos.notnull()]
    # fasta_df_iso = fasta_df_temp[fasta_df_temp.VariantPos.isnull()]

    # multiple SAAVs to list
    fasta_df_var["VariantPos"] = fasta_df_var["VariantPos"].str.split(pat=".")

    # generate variant sequences
    fasta_df_var = multi_process("process_saavs", fasta_df_var, "variants", cfg)

    return fasta_df_var


def process_saavs(df, df_status, jid, p_bar_val, *args):
    cfg = args[0][0]

    df["VarSeq"] = None
    df["VarSeqCleave"] = None
    var_seq_len = cfg["var_seq_length"]

    for pos in df.itertuples():
        var_seq_temp = {}
        var_seq_cleave_temp = {}
        for p in pos.VariantPos:
            if not p[-1].isnumeric():
                # substitution, termination, deletion
                aa_pos = int(p[1:-1])-1
                seq_pos_last = len(pos.Sequence)
                pos_start = [aa_pos - var_seq_len if aa_pos - var_seq_len > 0 else 0][0]
                pos_end = [aa_pos + var_seq_len + 1 if aa_pos + var_seq_len + 1 < seq_pos_last else seq_pos_last][0]
                pre_x = ["#" * (2 * var_seq_len + 1 - (pos_end-pos_start))
                         if pos_start == 0 else ""][0]
                post_x = ["#" * (2 * var_seq_len + 1 - (pos_end-pos_start))
                          if aa_pos + var_seq_len + 1 > seq_pos_last else ""][0]
                var_seq_temp[p] = pre_x + pos.Sequence[pos_start:pos_end] + post_x
                var_seq_cleave_temp[p] = cleave_peptide(var_seq_temp[p], cfg)
        df["VarSeq"].loc[pos.Index] = [var_seq_temp]
        df["VarSeqCleave"].loc[pos.Index] = [var_seq_cleave_temp]

        # update progress bar value process-wise
        p_bar_val[jid] = pos.Index

    # drop everything with invalid var seq
    df = df[df["VarSeq"].str[0] != {}]

    df_status[jid] = df


def process_mutation(df, df_status, jid, p_bar_val, *_):

    df["SequenceMut"] = None

    for i_seq, seq in df.iterrows():
        var_pos = int(seq.VariantPos[0][1:-1])
        var_sub = seq.VariantPos[0][-1]

        if not var_sub == "-" and not var_sub == "*":
            df["SequenceMut"].loc[i_seq] = seq.Sequence[:var_pos - 1] + var_sub + seq.Sequence[var_pos:]
        elif var_sub == "*":
            df["SequenceMut"].loc[i_seq] = seq.Sequence[:var_pos - 1]
        elif var_sub == "-":
            df["SequenceMut"].loc[i_seq] = seq.Sequence[:var_pos - 1] + seq.Sequence[var_pos:]

        # update progress bar
        p_bar_val[jid] = i_seq

    df = df.drop(columns=["Sequence"]).rename(columns={"SequenceMut": "Sequence"})

    df_status[jid] = df


def cleave_peptide(var_dict, cfg):
    regex = re.compile(cfg["enzyme_specificity"])

    enz_spec = [(str(var_dict).index(enz_match) + 1, str(var_dict).index(enz_match) + len(enz_match), enz_match)
                for enz_match in regex.findall(str(var_dict))]

    enz_peptide = None

    for e in enz_spec:
        enz_pos_low, enz_pos_high, enz_group = e

        if enz_pos_low <= cfg["var_seq_length"] + 1 <= enz_pos_high:
            # remove # from N- or C- terminal peptides
            if "#" in enz_group:
                enz_group = enz_group.replace("#", "")

            enz_peptide = Seq(enz_group)

            return enz_peptide
        else:
            enz_peptide = None
    if not enz_spec:
        enz_peptide = None

    return enz_peptide


def filter_id_with_reference(input_df, cfg):
    input_df_noid = input_df[input_df["UniProtID"] == "NoUniID"]

    with open(cfg["reference_dataset"]) as ref:
        ref_df_sep = pd.read_table(ref, sep=None, iterator=True, engine="python")
        ref_df_sep_det = ref_df_sep._engine.data.dialect.delimiter
        ref_df = pd.read_table(ref, sep=ref_df_sep_det)

    ref_list = ref_df.iloc[:, 0].tolist()

    output_df_temp = input_df[input_df['UniProtID'].isin(ref_list)]
    if cfg["filter_seq_with_reference_add_no_ids"]:
        output_df = pd.concat([output_df_temp, input_df_noid])
    else:
        output_df = output_df_temp

    return output_df


def filter_seq_with_reference(input_df, df_status, jid, p_bar_val, *args):
    ref_df = args[0][0]
    output_folder = args[0][1]
    ref_df = ref_df.Sequence.apply(str).str
    temp_bool_list = []

    i_var_pep = 0
    for var_pep in input_df.itertuples():
        temp_var_seq_str = str(next(iter(var_pep.VarSeqCleave[0].values())))
        ref_contains_bool = ref_df.contains(temp_var_seq_str, regex=False).any()
        temp_bool_list.append(ref_contains_bool)

        p_bar_val[jid] = i_var_pep
        i_var_pep += 1

    bool_df = pd.Series(temp_bool_list, name="bools")
    output_df = input_df[~bool_df.values]  # boolean series needs to be inverted by ~

    h5_path_out = os.path.join(output_folder, "temp/filter_seq_{}.h5".format(jid))

    output_df.to_hdf(h5_path_out, key="filter_seq_{}".format(jid), mode='w')

    df_status[jid] = pd.Series(h5_path_out)


def convert_df_to_bio_list(pd_seq, seq_format, min_pep_len=None, max_pep_len=None, keep_dups=None):
    seq_records = []
    seq_dups = []
    var_header = None

    if seq_format == "galaxy" or seq_format == "cosmic" or seq_format == "tso":
        for i_var_pep, var_pep in pd_seq.VarSeqCleave.items():
            var_counter = 1
            var_seq_last = None
            var_pos_last = None

            for var_pos, var_seq in var_pep[0].items():
                if var_seq and min_pep_len <= len(var_seq) <= max_pep_len:
                    if var_seq == var_seq_last:
                        # this concatenates variants with the same peptide
                        # drop last element of list
                        seq_records.pop()
                        seq_dups.pop()
                        # append variant position to last one
                        var_pos = "_".join((var_pos_last, var_pos))
                        var_counter -= 1
                    if seq_format == "galaxy":
                        # FASTA header: sp|UniProtID_VariantPos|ENSEMBLID
                        var_header = "sp|{}_{}_{}|{}".format(
                            pd_seq.UniProtID.loc[i_var_pep],
                            var_pos,
                            var_counter,
                            pd_seq.ProteinID.loc[i_var_pep].split("_")[0])
                    elif seq_format == "cosmic":
                        # FASTA header: sp|UniProtID_VariantPos|COSMICID
                        var_header = "sp|{}_{}_{}|{}".format(
                            pd_seq.UniProtID.loc[i_var_pep],
                            var_pos,
                            var_counter,
                            "{}".format(pd_seq.CosmicID.loc[i_var_pep]))
                    elif seq_format == "tso":
                        # FASTA header: sp|UniProtID_VariantPos|RefSeq_Protein
                        var_header = "sp|{}_{}_{}|{}".format(
                            pd_seq.UniProtID.loc[i_var_pep],
                            var_pos,
                            var_counter,
                            pd_seq.ProteinID.loc[i_var_pep])
                    record = SeqRecord(var_seq, var_header, '', '')
                    seq_records.append(record)
                    seq_dups.append(str(var_seq))
                    var_counter += 1
                    var_seq_last = var_seq
                    var_pos_last = var_pos

        if not keep_dups:
            # this will keep only the first occurance of a peptide
            dup_ids = pd.DataFrame(seq_dups).sort_values(by=0).duplicated(keep="first").sort_index().tolist()
            seq_records = [rec for (rec, dup) in zip(seq_records, dup_ids) if not dup]

    elif seq_format == "ensembl":
        for i_ens_id, ens_id in pd_seq.iterrows():
            record = SeqRecord(ens_id.Sequence,
                               ens_id.ProteinID,
                               ens_id.StableID,
                               ens_id.GeneID)
            seq_records.append(record)
    elif seq_format == "uniprot":
        for i_prot_id, prot_id in pd_seq.iterrows():
            record = SeqRecord(seq=prot_id.Sequence,
                               id=[prot_id.Description.split(" ", 1)[0] if "|" in prot_id.Description else
                                   prot_id.Identifier][0],
                               name=prot_id.Name,
                               description=[prot_id.Description.split(" ", 1)[1] if "|" in prot_id.Description else
                                            prot_id.Description][0])
            seq_records.append(record)

    return seq_records


if __name__ == "__main__":
    start_time = time.time()
    timestamp_str = datetime.now().strftime("%y%m%d%H%M")

    # Add parser for config YAML input
    parser = ArgumentParser()
    parser.add_argument("-c", "--config",
                        dest="config_yaml",
                        default=None,
                        help="Path to config YAML file")
    arg_parser = parser.parse_args()

    # print welcome message
    print_welcome()

    # Reformat log output
    stream.setFormatter(ColoredFormatter(LOGFORMAT, "%H:%M:%S"))

    # load config YAML
    if arg_parser.config_yaml:
        config_yaml = read_yaml(arg_parser.config_yaml)
        output_path = os.path.dirname(arg_parser.config_yaml)
    else:
        log.error("ProteoGenDB config YAML not found..")
        sys.exit()

    if config_yaml["map_galaxy"]:
        # get fasta DB of proteogenomics Galaxy workflow
        fasta_proteogen = read_fasta(config_yaml["fasta_proteogen_path"], "galaxy")

        # annotate saavs
        proteogen_saavs = annotate_saavs(fasta_proteogen, config_yaml)

        # keep proteins which are present in reference dataset if provided
        if config_yaml["reference_dataset"] != "":
            log.info("Filter UniProt IDs with reference dataset..")
            proteogen_saavs = filter_id_with_reference(proteogen_saavs, config_yaml)

        # filter sequences against reference proteome
        if config_yaml["filter_seq_with_reference"]:
            log.info("Filter variant sequences with reference proteome..")
            if config_yaml["generate_subFASTA"] and config_yaml["reference_dataset"] != "":
                log.info("Generate subFASTA proteome..")
                fasta_proteome = subset_fasta_db(config_yaml["reference_proteome"],
                                                 config_yaml["reference_dataset"])
            else:
                fasta_proteome = read_fasta(config_yaml["reference_proteome"], "uniprot")

            # create temp folder
            if not os.path.exists(os.path.join(output_path, "temp")):
                os.mkdir(os.path.join(output_path, "temp"))

            proteogen_saavs_h5 = multi_process("filter_seq_with_reference",
                                               proteogen_saavs,
                                               "sequences",
                                               fasta_proteome,
                                               output_path)

            proteogen_saavs = pd.DataFrame()

            for h5_path in proteogen_saavs_h5:
                h5_path_key = os.path.splitext(os.path.basename(h5_path))[0]
                proteogen_saavs_var_temp = pd.read_hdf(h5_path,
                                                       h5_path_key)
                proteogen_saavs = pd.concat([proteogen_saavs, proteogen_saavs_var_temp])
        else:
            fasta_proteome = read_fasta(config_yaml["reference_proteome"], "uniprot")

        # add disease information
        if config_yaml["add_disease_info"]:
            proteogen_saavs_out = annotate_variant_info(proteogen_saavs,
                                                        "GeneID",
                                                        config_yaml["annotation_data"],
                                                        config_yaml["min_spec_pep_len"],
                                                        config_yaml["max_spec_pep_len"]).to_csv(
                os.path.join(output_path, "{}_disease_annotation_galaxy.tsv".format(timestamp_str)),
                sep="\t")

        # mapped SAAVS to FASTA
        proteogen_saavs_list = convert_df_to_bio_list(proteogen_saavs,
                                                      "galaxy",
                                                      config_yaml["min_spec_pep_len"],
                                                      config_yaml["max_spec_pep_len"],
                                                      config_yaml["keep_saav_dups_in_fasta"])

        # write to FASTA
        log.info("Save annotated Galaxy SAAVs as FASTA..")
        write_fasta(proteogen_saavs_list, output_path, timestamp_str, "SAAV_sequences", "galaxy")

        fasta_proteome_name = os.path.basename(config_yaml["reference_proteome"]).split(".")[0]
        fasta_output = convert_df_to_bio_list(fasta_proteome, "uniprot")
        fasta_combined_output = fasta_output + proteogen_saavs_list

        if config_yaml["generate_subFASTA"]:
            log.info("Save subFASTA proteome..")

            write_fasta(fasta_output, output_path, timestamp_str, "subFASTA", "galaxy")
            write_fasta(fasta_combined_output, output_path, timestamp_str, "subFASTA_SAAV", "galaxy")
        else:
            log.info("Save FASTA proteome..")
            write_fasta(fasta_combined_output, output_path, timestamp_str, "FASTA_SAAV", "galaxy")


    if config_yaml["map_cosmic"]:
        # get ensembl fasta
        log.info("Load ENSEMBL fasta..")
        fasta_ensembl = read_fasta(config_yaml["fasta_ensembl"], "ensembl")

        # load CosmicMutantExport.tsv
        # only load columns of interest
        log.info("Load CosmicMutantExport.tsv..")
        cosmic_coi = {
            "Accession Number": "str",
            "Sample name": "str",
            "ID_sample": "int32",
            "ID_tumour": "int32",
            "Primary site": "str",
            "Primary histology": "str",
            "Histology subtype 1": "str",
            "LEGACY_MUTATION_ID": "str",
            "Mutation AA": "str",
            "Resistance Mutation": "str",
            "Mutation somatic status": "str",
            "Pubmed_PMID": "int32",
            "ID_STUDY": "int16",
            "Sample Type": "str",
            "Tumour origin": "str",
            "HGVSP": "str"
        }
        cosmic_coi_keys = list(cosmic_coi.keys())

        cosmic_data = pd.read_table(config_yaml["cosmic_mutant_export"],
                                    usecols=cosmic_coi_keys,
                                    encoding="cp1252")
        if config_yaml["cosmic_primary_site_filter"]:
            cosmic_primary_site_set = None

            if not config_yaml["cosmic_primary_site_set"]:
                cosmic_primary_site_select = True

                # get all possible entries
                cosmic_data_primary_sites = cosmic_data["Primary site"].unique().tolist()

                # user input
                log.info("Please select a primary site:")
                cosmic_data_primary_sites_len = len(cosmic_data_primary_sites)
                for i in range(0, len(cosmic_data_primary_sites), 2):
                    if not i == cosmic_data_primary_sites_len-1 or not cosmic_data_primary_sites_len % 2 == 1:
                        log.info("\t{}. {} | {}. {}".format(i, cosmic_data_primary_sites[i],
                                                            i + 1, cosmic_data_primary_sites[i + 1]))
                    else:
                        log.info("\t{}. {}".format(i, cosmic_data_primary_sites[i]))
                while cosmic_primary_site_select:
                    cosmic_primary_site_set = input("Select ID: ")
                    try:
                        cosmic_primary_site_set = cosmic_data_primary_sites[int(cosmic_primary_site_set)]
                        cosmic_primary_site_select = False
                    except ValueError:
                        log.error("Error: Type in a number!")
                    except IndexError:
                        log.error("Error: ID not in list!")
            else:
                cosmic_primary_site_set = config_yaml["cosmic_primary_site_set"]

            cosmic_data = cosmic_data[cosmic_data["Primary site"] == cosmic_primary_site_set]

        cosmic_data_ensp = cosmic_data[cosmic_data.HGVSP.notnull()].filter(["LEGACY_MUTATION_ID",
                                                                            "Mutation AA",
                                                                            "HGVSP"])
        # Free memory
        del cosmic_data

        cosmic_data_ensp["ProteinID"] = cosmic_data_ensp["HGVSP"].str.split(":").str[0]
        cosmic_data_ensp["VariantPos"] = cosmic_data_ensp["Mutation AA"].str.split("p.").str[1]
        cosmic_data_ensp = cosmic_data_ensp.rename(columns={"LEGACY_MUTATION_ID": "CosmicID"})

        # Filter
        cosmic_data_ensp_filt = cosmic_data_ensp.filter(["CosmicID", "ProteinID", "VariantPos"])

        # deduplicate
        cosmic_data_ensp_filt = cosmic_data_ensp_filt.drop_duplicates()

        # Free memory
        del cosmic_data_ensp

        # drop all coding silent substitutions, deletions, insertions, duplications, frameshifts
        cosmic_data_ensp_filt = cosmic_data_ensp_filt[~cosmic_data_ensp_filt["VariantPos"].str.contains(
            "=|del|ins|dup|fs|ext|Sec|\\?",
            case=False)
        ]

        # move VariantPos values in list
        cosmic_data_ensp_filt["VariantPos"] = cosmic_data_ensp_filt["VariantPos"].apply(lambda x: [x])

        # add protein sequence
        cosmic_data_ensp_filt = cosmic_data_ensp_filt.merge(fasta_ensembl[["ProteinID", "Sequence"]],
                                                            on='ProteinID',
                                                            how='left')
        cosmic_data_ensp_filt_nan = cosmic_data_ensp_filt[cosmic_data_ensp_filt["Sequence"].isnull()]

        # get peptides
        log.info("Mutate ENSEMBL sequence with COSMIC mutations..")
        cosmic_data_ensp_filt_mut = multi_process("process_mutation",
                                                  cosmic_data_ensp_filt,
                                                  "COSMIC mutations",
                                                  config_yaml)
        log.info("Get COSMIC peptide variants..")
        cosmic_data_ensp_filt_var = multi_process("process_saavs",
                                                  cosmic_data_ensp_filt_mut,
                                                  "COSMIC variants",
                                                  config_yaml)

        # get UniProt IDs via ENSP ID
        log.info("Map UniProt IDs to ENSEMBL IDs..")
        cosmic_mutant_export_ensp_filt_ids = cosmic_data_ensp_filt["ProteinID"].tolist()
        cosmic_mutant_export_ensp_filt_ids_unique = cosmic_data_ensp_filt[
            ~cosmic_data_ensp_filt["ProteinID"].duplicated()
        ]["ProteinID"].tolist()

        uniprot_lookup = get_uniprot_id(cosmic_mutant_export_ensp_filt_ids_unique).rename({"FromID": "ProteinID",
                                                                                           "ToID": "UniProtID"},
                                                                                          axis=1)
        uniprot_lookup["ProteinID"].apply(lambda x: x[:15])

        # Free memory
        del cosmic_data_ensp_filt

        # get UniProt IDs via ENSP ID without subversion
        uniprot_nan = list(dict.fromkeys(cosmic_data_ensp_filt_var["ProteinID"][
                                             ~cosmic_data_ensp_filt_var["ProteinID"].isin(uniprot_lookup["ProteinID"])
                                         ].tolist()))
        uniprot_nan_split = [ensp_id.split(".", 1)[0] for ensp_id in uniprot_nan]
        uniprot_lookup_nan = get_uniprot_id(uniprot_nan_split).rename({"FromID": "ProteinID",
                                                                       "ToID": "UniProtID"},
                                                                      axis=1)

        # concat both lookup dataframes
        uniprot_lookup = pd.concat([uniprot_lookup, uniprot_lookup_nan])

        # map UniProt IDs with lookup table
        cosmic_data_ensp_filt_var["UniProtID"] = multi_process("process_uniprot_ids",
                                                               cosmic_mutant_export_ensp_filt_ids,
                                                               "ids",
                                                               uniprot_lookup)
        cosmic_data_ensp_filt_var["UniProtID"] = cosmic_data_ensp_filt_var["UniProtID"].replace("", "NoUniID")

        cosmic_data_ensp_filt_var_len = len(cosmic_data_ensp_filt_var["UniProtID"][
                                                cosmic_data_ensp_filt_var["UniProtID"] != "NoUniID"
                                                ])

        # keep proteins which are present in reference dataset if provided
        if config_yaml["reference_dataset"] != "":
            log.info("Filter UniProt IDs with reference dataset..")
            cosmic_data_ensp_filt_var = filter_id_with_reference(cosmic_data_ensp_filt_var, config_yaml)

        # filter sequences against reference proteome
        if config_yaml["filter_seq_with_reference"]:
            log.info("Filter variant sequences with reference proteome..")
            if config_yaml["generate_subFASTA"] and config_yaml["reference_dataset"] != "":
                log.info("Generate subFASTA proteome..")
                fasta_proteome = subset_fasta_db(config_yaml["reference_proteome"],
                                                 config_yaml["reference_dataset"])
            else:
                fasta_proteome = read_fasta(config_yaml["reference_proteome"], "uniprot")

            # create temp folder
            if not os.path.exists(os.path.join(output_path, "temp")):
                os.mkdir(os.path.join(output_path, "temp"))

            cosmic_data_ensp_filt_var_h5 = multi_process("filter_seq_with_reference",
                                                         cosmic_data_ensp_filt_var,
                                                         "sequences",
                                                         fasta_proteome,
                                                         output_path)

            cosmic_data_ensp_filt_var = pd.DataFrame()

            for h5_path in cosmic_data_ensp_filt_var_h5:
                h5_path_key = os.path.splitext(os.path.basename(h5_path))[0]
                cosmic_data_ensp_filt_var_temp = pd.read_hdf(h5_path,
                                                             h5_path_key)
                cosmic_data_ensp_filt_var = pd.concat([cosmic_data_ensp_filt_var, cosmic_data_ensp_filt_var_temp])
        else:
            fasta_proteome = read_fasta(config_yaml["reference_proteome"], "uniprot")

        # add disease information
        if config_yaml["add_disease_info"]:
            cosmic_data_ensp_filt_var_out = annotate_variant_info(cosmic_data_ensp_filt_var,
                                                                  "CosmicID",
                                                                  config_yaml["annotation_data"],
                                                                  config_yaml["min_spec_pep_len"],
                                                                  config_yaml["max_spec_pep_len"]).to_csv(
                os.path.join(output_path, "{}_disease_annotation_cosmic.tsv".format(timestamp_str)),
                sep="\t")

        # mapped SAAVS to FASTA
        cosmic_data_ensp_filt_var_list = convert_df_to_bio_list(cosmic_data_ensp_filt_var,
                                                                "cosmic",
                                                                config_yaml["min_spec_pep_len"],
                                                                config_yaml["max_spec_pep_len"],
                                                                config_yaml["keep_saav_dups_in_fasta"])

        # write to FASTA
        log.info("Save annotated COSMIC SAAVs as FASTA..")
        write_fasta(cosmic_data_ensp_filt_var_list, output_path, timestamp_str, "SAAV_sequences", "cosmic")

        fasta_proteome_name = os.path.basename(config_yaml["reference_proteome"]).split(".")[0]
        fasta_output = convert_df_to_bio_list(fasta_proteome, "uniprot")
        fasta_combined_output = fasta_output + cosmic_data_ensp_filt_var_list

        if config_yaml["generate_subFASTA"]:
            log.info("Save subFASTA proteome..")

            write_fasta(fasta_output, output_path, timestamp_str, "subFASTA", "cosmic")
            write_fasta(fasta_combined_output, output_path, timestamp_str, "subFASTA_SAAV", "cosmic")
        else:
            log.info("Save FASTA proteome..")
            write_fasta(fasta_combined_output, output_path, timestamp_str, "FASTA_SAAV", "cosmic")

    if config_yaml["map_tso"]:
        log.info("Load CombinedVariantOutput of TSO 500 pipeline..")
        tso_data = read_tso(config_yaml["tso_path"])

        # mutate
        log.info("Mutate NCBI sequences with TSO mutations..")
        tso_data_mut = multi_process("process_mutation",
                                     tso_data,
                                     "TSO mutations",
                                     config_yaml)

        # process saavs
        log.info("Process TSO mutations..")
        tso_data_processed = multi_process("process_saavs",
                                           tso_data_mut,
                                           "variants",
                                           config_yaml)

        # get UniProt IDs via NCBI ID
        log.info("Map UniProt IDs to ENSEMBL IDs..")
        tso_prot_ids_unique = tso_data_processed[
            ~tso_data_processed["ProteinID"].duplicated()
        ]["ProteinID"].tolist()

        uniprot_lookup = get_uniprot_id(tso_prot_ids_unique,
                                        fmt_from="RefSeq_Protein",
                                        split_str=" ").rename({"FromID": "ProteinID",
                                                              "ToID": "UniProtID"},
                                                              axis=1)
        uniprot_lookup["ProteinID"].apply(lambda x: x[:15])

        # map UniProt IDs with lookup table
        tso_data_processed["UniProtID"] = multi_process("process_uniprot_ids",
                                                        tso_data_processed["ProteinID"],
                                                        "ids",
                                                        uniprot_lookup)

        tso_data_processed["UniProtID"] = tso_data_processed["UniProtID"].replace("", "NoUniID")

        # keep proteins which are present in reference dataset if provided
        if config_yaml["reference_dataset"] != "":
            log.info("Filter UniProt IDs with reference dataset..")
            tso_data_processed = filter_id_with_reference(tso_data_processed, config_yaml)

        # filter sequences against reference proteome
        if config_yaml["filter_seq_with_reference"]:
            log.info("Filter variant sequences with reference proteome..")
            if config_yaml["generate_subFASTA"] and config_yaml["reference_dataset"] != "":
                log.info("Generate subFASTA proteome..")
                fasta_proteome = subset_fasta_db(config_yaml["reference_proteome"],
                                                 config_yaml["reference_dataset"])
            else:
                fasta_proteome = read_fasta(config_yaml["reference_proteome"], "uniprot")

            # create temp folder
            if not os.path.exists(os.path.join(output_path, "temp")):
                os.mkdir(os.path.join(output_path, "temp"))

            tso_data_processed_h5 = multi_process("filter_seq_with_reference",
                                                  tso_data_processed,
                                                  "sequences",
                                                  fasta_proteome,
                                                  output_path)

            tso_data_processed = pd.DataFrame()

            for h5_path in tso_data_processed_h5:
                h5_path_key = os.path.splitext(os.path.basename(h5_path))[0]
                tso_data_processed_var_temp = pd.read_hdf(h5_path,
                                                          h5_path_key)
                tso_data_processed = pd.concat([tso_data_processed, tso_data_processed_var_temp])
        else:
            fasta_proteome = read_fasta(config_yaml["reference_proteome"], "uniprot")

        # add disease information
        if config_yaml["add_disease_info"]:
            tso_data_processed_out = annotate_variant_info(tso_data_processed,
                                                           "Gene",
                                                           config_yaml["annotation_data"],
                                                           config_yaml["min_spec_pep_len"],
                                                           config_yaml["max_spec_pep_len"]).to_csv(
                os.path.join(output_path, "{}_disease_annotation_tso.tsv".format(timestamp_str)),
                sep="\t")

        # mapped SAAVS to FASTA
        tso_data_processed_list = convert_df_to_bio_list(tso_data_processed,
                                                         "tso",
                                                         config_yaml["min_spec_pep_len"],
                                                         config_yaml["max_spec_pep_len"],
                                                         config_yaml["keep_saav_dups_in_fasta"])

        # write to FASTA
        log.info("Save annotated TSO SAAVs as FASTA..")
        write_fasta(tso_data_processed_list, output_path, timestamp_str, "SAAV_sequences", "tso")

        fasta_proteome_name = os.path.basename(config_yaml["reference_proteome"]).split(".")[0]
        fasta_output = convert_df_to_bio_list(fasta_proteome, "uniprot")
        fasta_combined_output = fasta_output + tso_data_processed_list

        if config_yaml["generate_subFASTA"]:
            log.info("Save subFASTA proteome..")

            write_fasta(fasta_output, output_path, timestamp_str, "subFASTA", "tso")
            write_fasta(fasta_combined_output, output_path, timestamp_str, "subFASTA_SAAV", "tso")
        else:
            log.info("Save FASTA proteome..")
            write_fasta(fasta_combined_output, output_path, timestamp_str, "FASTA_SAAV", "tso")

    end_time = time.time()
    log.info("Runtime (total): {}min".format(str(round((end_time - start_time) / 60, 2))))

    # remove temp folder
    shutil.rmtree(os.path.join(output_path, "temp"), ignore_errors=True)
