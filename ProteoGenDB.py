import os
import io
import re
import shutil
import time
import gzip
import yaml
import warnings
import pandas as pd
import numpy as np
import multiprocessing
import requests
import logging
import json
import platform
import vcf
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List, cast
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError
from urllib3.util.retry import Retry
from argparse import ArgumentParser
from datetime import datetime
from colorlog import ColoredFormatter
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from time import sleep
from tqdm import tqdm

global start_time

LOG_LEVEL   = logging.DEBUG
LOGFORMAT   = "%(log_color)s%(asctime)s %(message)s%(reset)s"
LOGFORMAT_W = "%(log_color)s%(message)s%(reset)s"

stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(ColoredFormatter(LOGFORMAT_W))
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
    log.info(" v2.3.0 \n")


def multi_process(func, input_df, unit, *args):
    # fail-safe for empty input_df
    if len(input_df) == 0:
        return (pd.DataFrame(columns=input_df.columns)
                if isinstance(input_df, pd.DataFrame)
                else pd.Series(dtype=object))

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


def read_strelka_vcf(vcf_path):
    strelka_df = pd.DataFrame(columns=["ProteinID", "UniProtID", "VariantPos", "GeneID"])

    if not isinstance(vcf_path, list):
        vcf_path = [vcf_path]

    for vcf_file in vcf_path:
        vcf_reader = vcf.Reader(open(vcf_file, 'r'))

        for record in vcf_reader:
            info = record.INFO

            # check if the variant is exonic and either nonsynonymous_SNV or stopgain
            if 'ExonicFunc.refGeneWithVer' in info:
                exonic_func = info['ExonicFunc.refGeneWithVer'][0]
                if exonic_func not in ['nonsynonymous_SNV', 'stopgain', 'nonframeshift_deletion']:
                    continue
            else:
                continue

            # extract AAChange information
            if 'AAChange.refGeneWithVer' in info:
                aa_changes = info['AAChange.refGeneWithVer']
                for aa_change in aa_changes:
                    parts = aa_change.split(':')
                    if len(parts) >= 5:
                        gene_id, protein_id, _, _, aa_pos = parts[:5]
                        protein_id = protein_id.split('.')[0]  # Remove version number
                        aa_pos = aa_pos.split('.')[-1]  # Get only the amino acid change

                        if "delins" in aa_pos or "_" in aa_pos or "fs" in aa_pos:
                            continue

                        # replace X to * (stopgain)
                        aa_pos = aa_pos.replace("X", "*")
                        # replace del to - (deletion)
                        aa_pos = aa_pos.replace("del", "-")

                        strelka_df = pd.concat([strelka_df, pd.DataFrame({
                            "ProteinID": protein_id,
                            "VariantPos": [[aa_pos]],
                            "GeneID": gene_id
                        })], ignore_index=True)

    # map RefSeq IDs to UniProt IDs
    # not all RefSeq IDs can be mapped to UniProt IDs!
    unique_protein_ids = strelka_df["ProteinID"].unique().tolist()

    uniprot_mapping = get_uniprot_id(
        unique_protein_ids,
        fmt_from="RefSeq_Nucleotide",
        split_str=" "
    )

    # create lookup dictionary
    uniprot_dict = dict(zip(uniprot_mapping["FromID"], uniprot_mapping["ToID"]))

    # add UniProt IDs to the dataframe
    strelka_df["UniProtID"] = strelka_df["ProteinID"].map(uniprot_dict)

    # fill NaN values with "NoUniID"
    strelka_df["UniProtID"] = strelka_df["UniProtID"].fillna("NoUniID")

    return strelka_df


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

    # if only one entry with NA is present return empty dataframe with all columns
    if tso_df["P-Dot Notation"].isna().all():
        tso_df = pd.DataFrame(columns=["P-Dot Notation", "ProteinID", "VariantPos3", "VariantPos", "Sequence"])
        return tso_df

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
    tso_df["Sequence"] = multi_process("fetch_fasta", tso_df["ProteinID"].tolist(), "ids", "ncbi")

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


def fetch_fasta(pids, df_status, jid, p_bar_val, *args):
    try:
        # urllib3 >=1.26
        retry_kwargs = {"allowed_methods": frozenset(["GET"])}
    except Exception:
        # fallback for older urllib3
        retry_kwargs = {"method_whitelist": frozenset(["GET"])}
    import random

    def _make_session():
        sess = requests.Session()
        r = Retry(
            total=5, connect=5, read=5,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            **retry_kwargs
        )
        ad = HTTPAdapter(max_retries=r, pool_connections=16, pool_maxsize=16)
        sess.mount("https://", ad); sess.mount("http://", ad)
        sess.headers.update({"User-Agent": "ProteoGenDB/2.2 (requests)"})
        return sess

    def _sleep_jitter(base):
        time.sleep(base + random.random() * 0.5)

    seq_records = []
    mode = args[0][0]
    session = _make_session()

    if mode == "ncbi":
        base_url = "https://www.ncbi.nlm.nih.gov/search/api/download-sequence/?db=protein&id={pid}&filename={pid}"
        for i_pid, pid in enumerate(pids):
            got = False
            for attempt in range(6):
                try:
                    resp = session.get(base_url.format(pid=pid), timeout=20)
                    txt = (resp.text or "")
                    # FASTA must start with ">"
                    if resp.ok and txt.lstrip().startswith(">"):
                        try:
                            prot_seq_rec = SeqIO.read(io.StringIO(txt), "fasta").seq
                            seq_records.append(prot_seq_rec)
                            got = True
                            break
                        except Exception:
                            # sometimes NCBI returns an empty FASTA stub -> retry
                            pass
                    # retry on anything else
                except requests.exceptions.RequestException as e:
                    # network hiccup -> backoff and retry
                    log.warning(f"{pid}: NCBI fetch error ({type(e).__name__}). Retry {attempt+1}/6")
                _sleep_jitter(1.0 * (attempt + 1))
            if not got:
                log.error(f"{pid}: NCBI FASTA fetch failed after retries. Yielding empty sequence.")
                seq_records.append(Seq(""))
            p_bar_val[jid] = i_pid

    elif mode == "ncbi_NM":
        # EFetch protein translation from NM_... transcripts
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        ncbi_api_key = args[0][1]
        for i_pid, pid in enumerate(pids):
            got = False
            for attempt in range(6):
                try:
                    params = {
                        "db": "nucleotide",
                        "id": pid,
                        "rettype": "fasta_cds_aa",
                        "retmode": "xml",   # response is text containing FASTA blocks
                    }
                    if ncbi_api_key:
                        params["api_key"] = ncbi_api_key
                    resp = session.get(efetch_url, params=params, timeout=30)
                    txt = (resp.text or "").strip()
                    # very defensive: grab anything after first header line if present
                    if resp.ok and ">" in txt:
                        # concatenate all lines except headers -> amino-acid sequence(s), pick first CDS
                        lines = [ln.strip() for ln in txt.splitlines()]
                        # find first FASTA block
                        block = []
                        in_block = False
                        for ln in lines:
                            if ln.startswith(">"):
                                if block:
                                    break
                                in_block = True
                                continue
                            if in_block and ln:
                                block.append(ln)
                        translated_seq = "".join(block)
                        if translated_seq:
                            seq_records.append(Seq(translated_seq))
                            got = True
                            break
                    # retry otherwise
                except requests.exceptions.RequestException as e:
                    log.warning(f"{pid}: EFetch error ({type(e).__name__}). Retry {attempt+1}/6")
                # rate limits
                _sleep_jitter(2.0 if ncbi_api_key else 5.0)
            if not got:
                log.error(f"{pid}: EFetch failed after retries. Yielding empty sequence.")
                seq_records.append(Seq(""))
            p_bar_val[jid] = i_pid

    elif mode == "uniprot":
        # EBI Proteins API JSON chunked by 100 accessions
        pid_list = [pids[i:i + 100] for i in range(0, len(pids), 100)]
        fetched = 0
        for chunk_idx, chunk_ids in enumerate(pid_list):
            pids_str = ",".join(chunk_ids)
            base_url = "https://www.ebi.ac.uk/proteins/api/proteins?offset=0&size=100&accession={pids}"
            got = False
            for attempt in range(6):
                try:
                    resp = session.get(
                        base_url.format(pids=pids_str),
                        headers={"Accept": "application/json"},
                        timeout=30
                    )
                    if not resp.ok:
                        raise requests.exceptions.RequestException(f"HTTP {resp.status_code}")
                    # JSON can be incomplete on transient failures
                    try:
                        seq_json = resp.json()
                    except Exception:
                        # retry JSON decoding issues
                        raise requests.exceptions.RequestException("JSON decode error")
                    # sort by request order
                    try:
                        seq_json = sorted(seq_json, key=lambda k: chunk_ids.index(k['accession']))
                    except Exception:
                        pass
                    for seq in seq_json:
                        seq_header = f">sp|{seq.get('accession','')}|{seq.get('id','')}\n"
                        seq_str = (seq.get("sequence") or {}).get("sequence") or ""
                        if not seq_str:
                            seq_records.append(Seq(""))
                        else:
                            try:
                                prot_seq_rec = SeqIO.read(io.StringIO(seq_header + seq_str), "fasta").seq
                                seq_records.append(prot_seq_rec)
                            except Exception:
                                seq_records.append(Seq(seq_str))
                    got = True
                    break
                except requests.exceptions.RequestException as e:
                    log.warning(f"UniProt chunk {chunk_idx+1}/{len(pid_list)} error ({type(e).__name__}). Retry {attempt+1}/6")
                    _sleep_jitter(1.0 * (attempt + 1))
            if not got:
                log.error(f"UniProt chunk {chunk_idx+1}: failed after retries. Filling chunk with empty sequences.")
                # keep output length aligned with input
                for _ in chunk_ids:
                    seq_records.append(Seq(""))

            fetched += len(chunk_ids)
            p_bar_val[jid] = fetched

    # finalize to shared manager dict
    seq_records_df = pd.Series(seq_records)

    df_status[jid] = seq_records_df


def get_annotation_data(pids, df_status, jid, p_bar_val, *args):
    api_name = args[0][0]
    api_url  = args[0][1]
    annot_folder = args[0][2]

    os.makedirs(annot_folder, exist_ok=True)
    db_path = os.path.join(annot_folder, f"{api_name}_cache.sqlite")

    with closing(_sqlite_connect(db_path)) as con:
        _sqlite_init(con)

        # Preload cache in chunks
        cached: dict = {}
        chunk = 900
        for i in range(0, len(pids), chunk):
            part = pids[i:i+chunk]
            q = f"SELECT pid,data FROM cache WHERE pid IN ({','.join('?'*len(part))})"
            for pid, data in con.execute(q, part).fetchall():
                try:
                    cached[pid] = json.loads(data)
                except Exception:
                    pass

        now = int(time.time())
        # Single pass over ALL pids to keep progress in sync
        for i, pid in enumerate(pids):
            if pid not in cached:
                # fetch (retry on transient net issues)
                while True:
                    try:
                        api_data = requests.get(api_url.format(pid=pid)).json()
                        break
                    except ConnectionError:
                        log.error("UniProt REST API issue.. retry..")
                        sleep(1)
                # insert (tolerate brief locks)
                for _ in range(8):
                    try:
                        con.execute(
                            "INSERT OR IGNORE INTO cache(pid,data,ts) VALUES (?,?,?)",
                            (pid, json.dumps(api_data), now)
                        )
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e).lower():
                            sleep(0.25)
                            continue
                        raise
                cached[pid] = api_data

            p_bar_val[jid] = i  # <- advance for every pid

        result_data = [cached.get(pid) for pid in pids]

    df_status[jid] = pd.Series(result_data)


def get_variant_info(pids, df_status, jid, p_bar_val, *args):
    cfg = args[0][0]
    rows = []

    for i, rec in enumerate(pids):
        feats = (rec or {}).get("features") or []

        for v in feats:
            # filters
            if cfg["variant_reviewed_filter"]:
                cs = v.get("clinicalSignificances") or []
                if not cs or not any(s in (cs[0].get("sources") or []) for s in cfg["variant_reviewed_filter"]):
                    continue
            if cfg["variant_evidences_cutoff"] > 0:
                ev = v.get("evidences") or []
                if len(ev) < cfg["variant_evidences_cutoff"]:
                    continue

            begin = v.get("begin")
            end   = v.get("end")
            wild  = v.get("wildType", "?")
            mut   = v.get("mutatedType")

            varpos_str = f"{wild}{begin}{mut}"
            varpos     = [varpos_str] if begin == end else []

            # collect one row
            row = {
                "DL_UniProtID": rec.get("accession"),
                "DL_ftID":      v.get("ftId"),
                "DL_CosmicID":  next((x.get("id") for x in (v.get("xrefs") or []) if x.get("name")=="cosmic curated"), None),
                "DL_ClinVarID": next((x.get("id") for x in (v.get("xrefs") or []) if x.get("name")=="ClinVar"), None),
                "DL_VariantPosStr": varpos_str if varpos else "",
                "DL_VariantPos":    varpos,
                "DL_MAF": next((pf.get("frequency") for pf in (v.get("populationFrequencies") or []) if pf.get("populationName")=="MAF"), None),
                "DL_sig_patho": np.nan, "DL_sig_likely_patho": np.nan,
                "DL_sig_likely_benign": np.nan, "DL_sig_benign": np.nan,
                "DL_sig_uncertain": np.nan, "DL_sig_conflict": np.nan,
                "DL_disease_association": "1" if v.get("association") else "0",
            }

            for cs in (v.get("clinicalSignificances") or []):
                t, srcs = cs.get("type"), ";".join(cs.get("sources") or [])
                if   t == "Likely pathogenic": row["DL_sig_likely_patho"] = srcs
                elif t == "Pathogenic": row["DL_sig_patho"] = srcs
                elif t == "Variant of uncertain significance": row["DL_sig_uncertain"] = srcs
                elif t == "Benign": row["DL_sig_benign"] = srcs
                elif t == "Likely benign": row["DL_sig_likely_benign"] = srcs
                elif t == "Conflicting interpretations of pathogenicity": row["DL_sig_conflict"] = srcs

            rows.append(row)

        p_bar_val[jid] = i

    df_status[jid] = pd.DataFrame(rows, copy=False)


def map_variant_info(dl_df, df_status, jid, p_bar_val, *args):
    input_df, min_len, max_len = args[0]
    cols = input_df.columns.union(dl_df.columns)
    out  = pd.DataFrame(columns=cols)

    for i, (_, var) in enumerate(dl_df.iterrows()):
        prot = input_df[input_df.UniProtID.eq(var.DL_UniProtID)]
        hits = prot[prot.VariantPos.isin(var.DL_VariantPos)]
        if len(hits) and hits.VarSeqCleave.values[0] is not None:
            pep = hits.VarSeqCleave.values[0]
            if min_len <= len(pep) <= max_len:
                out = pd.concat([out,
                                 pd.concat([hits.reset_index(drop=True).drop(["VarSeq"], axis=1, errors="ignore"),
                                            pd.DataFrame(var).T.reset_index(drop=True)], axis=1)],
                                ignore_index=True)
        p_bar_val[jid] = i
    df_status[jid] = out


def _sqlite_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30, isolation_level=None)  # autocommit
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA busy_timeout=5000")
    return con

def _sqlite_init(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            pid TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            ts   INTEGER NOT NULL
        )
    """)


def annotate_variant_info(input_df, left_join, annot_data_path, min_pep_len, max_pep_len):
    log.info(f"Collect disease information for {left_join}s..")

    df = input_df.copy()

    # lighten memory
    df.drop(["Sequence"], axis=1, inplace=True, errors="ignore")

    # one row per variant
    df = df.explode('VariantPos')

    # map each VariantPos -> peptide string
    def _pep_at_pos(row):
        pos = row['VariantPos']
        m = row['VarSeqCleave'][0] if isinstance(row['VarSeqCleave'], list) else row['VarSeqCleave']
        if isinstance(m, dict):
            return m.get(pos, '')
        return ''

    df['VarSeq'] = df.apply(_pep_at_pos, axis=1)
    df['VarSeqCleave'] = df['VarSeq']
    df.drop_duplicates(subset=["UniProtID", "VariantPos", "VarSeqCleave"], keep="first", inplace=True)

    # context columns to preserve if present
    ctx_cols = [c for c in ["GeneID", "ProteinID", "RelPosMap"] if c in df.columns]

    # base set = all SAAV peptides that would go to FASTA
    mask = df['VarSeqCleave'].apply(lambda s: bool(s) and min_pep_len <= len(str(s)) <= max_pep_len)
    base_df = df.loc[mask, ["UniProtID", "VariantPos", "VarSeqCleave", "VarSeq"] + ctx_cols].reset_index(drop=True)

    annotate_data = multi_process("get_annotation_data",
                                  df.UniProtID.drop_duplicates().to_list(),
                                  left_join,
                                  "ebi",
                                  "https://www.ebi.ac.uk/proteins/api/variation/{pid}",
                                  annot_data_path)

    log.info(f"Extract variant information for {left_join}s..")
    disease_lookup_df = multi_process("get_variant_info",
                                      annotate_data,
                                      left_join,
                                      config_yaml)

    # remove duplicates with UniProtID and VarPos
    disease_lookup_df = disease_lookup_df.drop_duplicates(
        subset=["DL_UniProtID", "DL_VariantPosStr"])

    log.info("Map variant information..")
    # rows that do have UniProt-annotated variants and pass peptide-length filter
    ann_df = multi_process("map_variant_info",
                           disease_lookup_df,
                           "Variants",
                           df,
                           min_pep_len,
                           max_pep_len)

    # keep only columns for FASTA header + disease fields
    drop_cols = ["DL_UniProtID", "DL_CosmicID", "DL_VariantPos", "DL_VariantPosStr"]
    ann_df.drop([c for c in drop_cols if c in ann_df.columns], axis=1, inplace=True)
    ann_df.rename(columns=lambda x: x.replace("DL_", ""), inplace=True)

    # mark annotated
    if not ann_df.empty:
        ann_df["Annotated"] = True

    # add unannotated SAAVs with NA fields
    key_cols = ["UniProtID", "VariantPos", "VarSeqCleave"]
    have = ann_df[key_cols].drop_duplicates() if not ann_df.empty else pd.DataFrame(columns=key_cols)
    missing = base_df.merge(have, on=key_cols, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")

    # ensure same columns -> NA-fill disease fields
    if ann_df.empty:
        # define disease columns when nothing was annotated
        disease_cols = ["ftID", "ClinVarID", "MAF",
                        "sig_patho", "sig_likely_patho", "sig_likely_benign",
                        "sig_benign", "sig_uncertain", "sig_conflict",
                        "disease_association"]
        ann_df = pd.DataFrame(columns=key_cols + ["VarSeq"] + ctx_cols + disease_cols + ["Annotated"])

    for c in ann_df.columns:
        if c not in missing.columns:
            missing[c] = "NA"
    missing["Annotated"] = False

    # combine and final NA normalization
    out_df = pd.concat([ann_df, missing[ann_df.columns]], ignore_index=True)
    obj_cols = [c for c in out_df.columns if out_df[c].dtype == object and c != "Annotated"]
    out_df[obj_cols] = out_df[obj_cols].replace({np.nan: "NA"})

    return out_df


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


def process_saavs(df, df_status, jid, p_bar_val, *args):
    cfg = args[0][0]

    df["VarSeq"] = None
    df["VarSeqCleave"] = None
    df["RelPosMap"] = None
    var_seq_len = cfg["var_seq_length"]

    for pos in df.itertuples():
        var_seq_temp = {}
        var_seq_cleave_temp = {}
        relpos_temp = {}
        if pos.Sequence is not None:
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
                    cleave_out = cleave_sequence(var_seq_temp[p], cfg)
                    # cleave_out is (Seq, rel_idx) in SAAV mode (non-isoform)
                    if isinstance(cleave_out, tuple):
                        pep, rel_idx = cleave_out
                    else:
                        pep, rel_idx = cleave_out, None
                    var_seq_cleave_temp[p] = pep
                    # define RelPos:
                    # stop "*": len(pep)+1
                    # deletion "-": rel_idx
                    # substitution: rel_idx
                    if pep is not None:
                        if p[-1] == "*":
                            relpos_temp[p] = len(str(pep)) + 1
                        else:
                            relpos_temp[p] = rel_idx
            df["VarSeq"].loc[pos.Index] = [var_seq_temp]
            df["VarSeqCleave"].loc[pos.Index] = [var_seq_cleave_temp]
            df["RelPosMap"].loc[pos.Index] = [relpos_temp]
        else:
            df["VarSeq"].loc[pos.Index] = [{}]
            df["VarSeqCleave"].loc[pos.Index] = [{}]
            df["RelPosMap"].loc[pos.Index] = [{}]

        # update progress bar value process-wise
        p_bar_val[jid] = pos.Index

    # drop everything with invalid var seq
    df = df[df["VarSeq"].str[0] != {}]

    df_status[jid] = df


def process_mutation(df, df_status, jid, p_bar_val, *args):

    df["SequenceMut"] = None

    for i_seq, seq in df.iterrows():
        var_pos = int(seq.VariantPos[0][1:-1])
        var_sub = seq.VariantPos[0][-1]
        var_aa = seq.VariantPos[0][0]

        # keep only variants where consensus AA occurs is in sequence
        # --> missmatch of mapped sequence (ENSEMBL (read_fasta, "ensembl") or NCBI (fetch_fasta)
        if len(seq.Sequence) < var_pos:
            # drop sequence if variant position is out of range
            df["SequenceMut"].loc[i_seq] = Seq("")
        else:
            if seq.Sequence[var_pos - 1:var_pos][0] == var_aa:
                if not var_sub == "-" and not var_sub == "*":
                    df["SequenceMut"].loc[i_seq] = seq.Sequence[:var_pos - 1] + var_sub + seq.Sequence[var_pos:]
                elif var_sub == "*":
                    df["SequenceMut"].loc[i_seq] = seq.Sequence[:var_pos - 1]
                elif var_sub == "-":
                    df["SequenceMut"].loc[i_seq] = seq.Sequence[:var_pos - 1] + seq.Sequence[var_pos:]
                else:
                    df["SequenceMut"].loc[i_seq] = Seq("")

        # update progress bar
        p_bar_val[jid] = i_seq

    df = df.drop(columns=["Sequence"]).rename(columns={"SequenceMut": "Sequence"})

    df_status[jid] = df


def cleave_sequence(var_dict, cfg, full_protein=False):
    regex = re.compile(cfg["enzyme_specificity"], re.IGNORECASE)

    enz_spec = [(str(var_dict).index(enz_match) + 1, str(var_dict).index(enz_match) + len(enz_match), enz_match)
                for enz_match in regex.findall(str(var_dict))]

    enz_peptide = None

    cleaved_peptides = []

    for e in enz_spec:
        enz_pos_low, enz_pos_high, enz_group = e

        if full_protein:
            if cfg["min_spec_pep_len"] <= enz_pos_high - enz_pos_low + 1 <= cfg["max_spec_pep_len"]:
                cleaved_peptides.append(Seq(enz_group))
        else:
            if enz_pos_low <= cfg["var_seq_length"] + 1 <= enz_pos_high:
                # compute relative index of the variant within this peptide
                center = cfg["var_seq_length"] + 1
                leading_hashes = len(enz_group) - len(enz_group.lstrip("#"))
                if "#" in enz_group:
                    enz_group = enz_group.replace("#", "")
                rel_idx = center - enz_pos_low + 1 - leading_hashes
                enz_peptide = Seq(enz_group)
                return (enz_peptide, rel_idx)
            else:
                enz_peptide = None
        if not enz_spec:
            enz_peptide = None

    if full_protein:
        return cleaved_peptides

    return enz_peptide


def filter_id_with_reference(input_df, cfg, column_str="UniProtID"):
    input_df_noid = input_df[input_df[column_str] == "NoUniID"]

    with open(cfg["reference_dataset"]) as ref:
        ref_df_sep = pd.read_table(ref, sep=None, iterator=True, engine="python")
        ref_df_sep_det = ref_df_sep._engine.data.dialect.delimiter
        ref_df = pd.read_table(ref, sep=ref_df_sep_det)

    ref_list = ref_df.iloc[:, 0].tolist()

    output_df_temp = input_df[input_df[column_str].isin(ref_list)]
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

    # process ref_df sequences with enzyme_specificity to get a pandas series
    regex = re.compile(".(?:(?<![KR]).)*")
    ref_df = ref_df.findall(regex)

    # explode reg_df to get a list of all peptides
    ref_df = ref_df.explode()

    # deduplicate red_df
    ref_df = ref_df.drop_duplicates()

    i_var_pep = 0
    for var_pep in input_df.itertuples():
        temp_var_seq_str = str(next(iter(var_pep.VarSeqCleave[0].values()))).upper()
        ref_contains_bool = ref_df.eq(temp_var_seq_str).any()
        temp_bool_list.append(ref_contains_bool)

        p_bar_val[jid] = i_var_pep
        i_var_pep += 1

    bool_df = pd.Series(temp_bool_list, name="bools")
    output_df = input_df[~bool_df.values]  # boolean series needs to be inverted by ~

    h5_path_out = os.path.join(output_folder, "temp/filter_seq_{}.h5".format(jid))

    output_df.to_hdf(h5_path_out, key="filter_seq_{}".format(jid), mode='w')

    df_status[jid] = pd.Series(h5_path_out)


def convert_df_to_bio_list(pd_seq, seq_format, min_pep_len=None, max_pep_len=None, keep_dups=None):
    # fail-safe checks
    if pd_seq.empty:
        return []

    seq_records = []
    seq_dups = []
    var_header = None

    if seq_format in ["galaxy", "cosmic", "tso", "strelka", "isoform", "mfa", "uniprot_mut", "saav_list"]:
        for i_var_pep, var_pep in pd_seq.VarSeqCleave.items():
            var_seq_last = None
            var_pos_last = None

            for var_pos, var_seq in var_pep[0].items():
                if var_seq and min_pep_len <= len(var_seq) <= max_pep_len:
                    # Prefer precomputed relative position (independent of full protein sequence)
                    rel_pos = "NA"
                    if "RelPosMap" in pd_seq.columns:
                        try:
                            rel_map = pd_seq.RelPosMap.loc[i_var_pep][0]
                            if isinstance(rel_map, dict) and var_pos in rel_map and rel_map[var_pos] is not None:
                                rel_pos = rel_map[var_pos]
                        except Exception:
                            pass
                    # fallback for stop codons if precomputed map not present
                    if rel_pos == "NA" and var_pos[-1] == "*":
                        rel_pos = len(str(var_seq)) + 1
                    if var_seq == var_seq_last:
                        # this concatenates variants with the same peptide
                        # drop last element of list
                        seq_records.pop()
                        seq_dups.pop()
                        # append variant position to last one
                        var_pos = "_".join((var_pos_last, var_pos))
                    if seq_format == "galaxy":
                        # FASTA header: sp|UniProtID_VariantPos~RelPos|ENSEMBLID
                        var_header = "sp|{}_{}~{}|{}".format(
                            pd_seq.UniProtID.loc[i_var_pep],
                            var_pos,
                            rel_pos,
                            pd_seq.ProteinID.loc[i_var_pep].split("_")[0])
                    elif seq_format == "cosmic":
                        # FASTA header: sp|UniProtID_ProteinID_VariantPos~RelPos|COSMICID
                        var_header = "sp|{}_{}_{}~{}|{}".format(
                            pd_seq.UniProtID.loc[i_var_pep],
                            pd_seq.ProteinID.loc[i_var_pep],
                            var_pos,
                            rel_pos,
                            "{}".format(pd_seq.CosmicID.loc[i_var_pep]))
                    elif seq_format in ["tso", "strelka"]:
                        # FASTA header: sp|UniProtID_VariantPos~RelPos|RefSeq_Protein
                        var_header = "sp|{}_{}~{}|{}".format(
                            pd_seq.UniProtID.loc[i_var_pep],
                            var_pos,
                            rel_pos,
                            pd_seq.ProteinID.loc[i_var_pep])
                    elif seq_format == "isoform":
                        # FASTA header: sp|UniProtID_PepID|RefSeq_Protein
                        var_header = "sp|{}_{}{}".format(
                            pd_seq.UniProtID.loc[i_var_pep],
                            var_pos,
                            [f"|{pd_seq.Consensus.loc[i_var_pep]}" if "Consensus" in pd_seq.columns else ""][0])
                    elif seq_format == "mfa":
                        # FASTA header: sp|UniProtID_VariantPos~RelPos|RefSeq_GeneID
                        var_header = "sp|{}_{}~{}|{}".format(
                            pd_seq.UniProtID.loc[i_var_pep],
                            var_pos,
                            rel_pos,
                            pd_seq.GeneID.loc[i_var_pep])
                    elif seq_format in ["uniprot_mut", "saav_list"]:
                        # FASTA header: sp|UniProtID_VariantPos~RelPos|VariantID
                        variant_id = pd_seq.VariantID.loc[i_var_pep] if 'VariantID' in pd_seq.columns else var_pos
                        var_header = "sp|{}_{}~{}|{}".format(
                            pd_seq.UniProtID.loc[i_var_pep],
                            var_pos,
                            rel_pos,
                            variant_id)
                    record = SeqRecord(var_seq, var_header, '', '')
                    seq_records.append(record)
                    seq_dups.append(str(var_seq))
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


def read_saav_list(saav_list_path):
    saav_df = pd.read_csv(saav_list_path, sep=None, engine='python')
    saav_df = saav_df.rename(columns={saav_df.columns[0]: 'UniProtID', saav_df.columns[1]: 'AA_change'})
    saav_df['VariantPos'] = saav_df['AA_change'].apply(lambda x: [x])
    return saav_df


def read_enst_list(enst_path):
    df = pd.read_csv(enst_path, sep=None, engine="python")
    df = df.rename(columns={df.columns[0]: "TranscriptID",
                            df.columns[1]: "VariantPos"})
    df["VariantPos"] = df["VariantPos"].str.strip().apply(lambda x: [x])
    return df


def prepare_reference_proteome(cfg):
    if cfg.get("_proteome_cache") is not None:
        return cfg["_proteome_cache"]

    if cfg["generate_subFASTA"] and cfg["reference_dataset"]:
        log.info("Generate subFASTA proteome..")
        prot = subset_fasta_db(cfg["reference_proteome"],
                               cfg["reference_dataset"])
    else:
        prot = read_fasta(cfg["reference_proteome"], "uniprot")

    cfg["_proteome_cache"] = prot
    return prot


def filter_to_consensus(df, proteome, seq_col="Sequence", id_col="UniProtID"):
    # get reference sequences from proteome
    ref = (
        proteome
        .dropna(subset=["Sequence", "Identifier"])
        .drop_duplicates(subset=["Identifier"])
        .set_index("Identifier")["Sequence"]
        .to_dict()
    )

    keep_mask: List[bool] = []

    # drop duplicated sequences in df with combined id_col and VariantPos identifier
    df = df.drop_duplicates(subset=[id_col, seq_col], keep="first")

    for uni_id, nm_seq in zip(df[id_col], df[seq_col]):
        nm_str = str(nm_seq).upper() if pd.notna(nm_seq) else ""
        # Note: NoUniID will be filtered out here!
        ref_str = str(ref.get(uni_id, ""))

        # NM sequence identical to reference sequence -> keep
        if nm_str == ref_str:
            keep_mask.append(True)
            continue

        # NM sequence is part of the reference sequence -> drop (aka isoform)
        if nm_str in ref_str:
            keep_mask.append(False)
            continue

        keep_mask.append(False)

    return df.loc[keep_mask].reset_index(drop=True)


def load_galaxy(cfg: dict) -> Tuple[pd.DataFrame, str, str, None, str]:
    df = read_fasta(cfg["fasta_proteogen_path"], "galaxy")

    log.info("Annotate SAAVs from Galaxy Workflow..")
    # extract SAAVs position from ProteinID
    df_temp = df.copy()
    df_temp["VariantPos"] = df["ProteinID"].str.split(pat="_").str[-1]

    # logic if ProteinID == VariantPos --> isoform/None
    df_temp.loc[(df_temp.VariantPos == df_temp.ProteinID), 'VariantPos'] = None

    # split iso and saavs
    df_var = df_temp[df_temp.VariantPos.notnull()]
    # df_iso = df_temp[df_temp.VariantPos.isnull()]

    # multiple SAAVs to list
    df_var["VariantPos"] = df_var["VariantPos"].str.split(pat=".")

    return df_var, "GeneID", "galaxy", None, "variants"


def load_cosmic(cfg: dict) -> Tuple[pd.DataFrame, str, str, str, str]:
    log.info("Load ENSEMBL reference FASTA..")
    ensembl_seqs = read_fasta(cfg["fasta_ensembl"], "ensembl")[["ProteinID", "Sequence"]]

    log.info("Load CosmicMutantExport.tsv..")
    cosmic_cols = {
        "Accession Number":    "str",
        "LEGACY_MUTATION_ID":  "str",
        "Mutation AA":         "str",
        "HGVSP":               "str",
        "Primary site":        "str"
    }
    cosmic = pd.read_table(
        cfg["cosmic_mutant_export"],
        usecols=list(cosmic_cols),
        encoding="cp1252"
    )

    if cfg.get("cosmic_primary_site_filter", False):
        desired_site = cfg.get("cosmic_primary_site_set")
        if desired_site:
            cosmic = cosmic[cosmic["Primary site"] == desired_site]
        else:
            log.warning("cosmic_primary_site_filter is True "
                        "but 'cosmic_primary_site_set' not given – "
                        "keeping all sites.")

    cosmic = cosmic[cosmic["HGVSP"].notnull()].filter(["LEGACY_MUTATION_ID",
                                                       "Mutation AA",
                                                       "HGVSP"])

    cosmic["ProteinID"]  = cosmic["HGVSP"].str.split(":").str[0]  # ENSP
    cosmic["VariantPos"] = cosmic["Mutation AA"].str.replace(r"^p\.", "", regex=True)

    # drop all coding silent substitutions, deletions, insertions, duplications, frameshifts
    cosmic = cosmic[
        ~cosmic["VariantPos"].str.contains(r"=|del|ins|dup|fs|ext|Sec|\?", case=False, na=False)
    ]

    cosmic = cosmic.rename(columns={"LEGACY_MUTATION_ID": "CosmicID"})
    cosmic = cosmic[["CosmicID", "ProteinID", "VariantPos"]].drop_duplicates()
    cosmic["VariantPos"] = cosmic["VariantPos"].apply(lambda x: [x])

    # add ENSEMBL sequences
    cosmic = cosmic.merge(ensembl_seqs, on="ProteinID", how="left")

    # list for UniProt mapping
    prot_ids = cosmic["ProteinID"].drop_duplicates().tolist()

    # first, try the full ID incl. sub-version
    lookup_full = get_uniprot_id(prot_ids).rename(
        columns={"FromID": "ProteinID", "ToID": "UniProtID"}
    )

    # re-try the still-unmapped IDs after stripping the ".x" suffix
    unresolved      = [p for p in prot_ids if p not in lookup_full["ProteinID"].values]
    unresolved_base = [p.split(".", 1)[0] for p in unresolved]

    lookup_base = pd.DataFrame()
    if unresolved_base:
        lookup_base = get_uniprot_id(unresolved_base).rename(
            columns={"FromID": "ProteinID", "ToID": "UniProtID"}
        )

    # combined lookup table
    uniprot_lookup = pd.concat([lookup_full, lookup_base], ignore_index=True)

    # broadcast UniProtIDs to every row
    cosmic["UniProtID"] = multi_process(
        "process_uniprot_ids",
        cosmic["ProteinID"].tolist(),
        "ids",
        uniprot_lookup
    )
    cosmic["UniProtID"] = cosmic["UniProtID"].replace("", "NoUniID")

    # ready for the central pipeline
    return cosmic, "CosmicID", "cosmic", "COSMIC mutations", "COSMIC variants"


def load_tso(cfg: dict) -> Tuple[pd.DataFrame, str, str, str, str]:
    df = read_tso(cfg["tso_path"])
    if df.empty:
        log.error("No variants in TSO data..")
        return pd.DataFrame(), "", "", "", ""

    # get UniProt IDs via NCBI ID
    log.info("Map UniProt IDs to ENSEMBL IDs..")
    uniprot = get_uniprot_id(df["ProteinID"].drop_duplicates().tolist(),
                             fmt_from="RefSeq_Protein",
                             split_str=" ").rename(columns={"FromID": "ProteinID",
                                                             "ToID": "UniProtID"})

    uniprot["ProteinID"].apply(lambda x: x[:15])

    # map UniProt IDs with lookup table
    df["UniProtID"] = multi_process("process_uniprot_ids",
                                    df["ProteinID"],
                                    "ids",
                                    uniprot)

    df["UniProtID"] = df["UniProtID"].fillna("NoUniID")

    return df, "Gene", "tso", "TSO mutation", "variants"


def load_mfa(cfg: dict) -> Tuple[pd.DataFrame, str, str, str, str]:
    mfa = pd.read_table(cfg["mfa_path"])

    # columns to keep: gene_name, Variant_Classification, tx, aaChange, ExonicFunc.refGene, TumorVAF
    mfa = mfa[
        ["gene_name", "Variant_Classification", "tx", "txChange", "aaChange", "ExonicFunc.refGene", "TumorVAF"]]

    # drop rows without aaChange
    mfa = mfa[mfa["aaChange"].str.startswith("p.", na=False)]

    # keep nonsynonymous SNV
    mfa = mfa[mfa["ExonicFunc.refGene"].isin(["nonsynonymous SNV", "stopgain"])]

    # drop rows where txChange contains an "_"
    mfa = mfa[~mfa["txChange"].str.contains("_")]

    # change X to * in aaChange
    mfa["aaChange"] = mfa["aaChange"].str.replace("X", "*")

    # filter for VAF (according to config.yaml value)
    mfa = mfa[mfa["TumorVAF"] >= cfg["mfa_vaf_cutoff"]]

    # strip p. from aaChange
    mfa["VariantPos"] = mfa["aaChange"].str.replace("p.", "", regex=False)

    # put VariantPos in list
    mfa["VariantPos"] = mfa["VariantPos"].apply(lambda x: [x])

    # map gene to uniprot, fetch sequences, mutate, digest
    uniprot = get_uniprot_id(mfa["gene_name"].unique(), fmt_from="Ensembl")\
              .rename(columns={"FromID": "gene_name", "ToID": "UniProtID"})

    log.info("Fetching UniProt sequences for mfa variants..")
    uniprot["Sequence"] = multi_process("fetch_fasta",
                                        uniprot["UniProtID"].tolist(),
                                        "ids",
                                        "uniprot")

    mfa = mfa.merge(uniprot, on="gene_name", how="left").dropna(subset=["Sequence"])

    # rename gene_name to GeneID
    mfa = mfa.rename(columns={"gene_name": "GeneID"})

    # drop NAs in GeneID
    mfa = mfa.dropna(subset=["GeneID"])

    return mfa, "Gene", "mfa", "mfa mutation", "variants"


def load_strelka(cfg: dict) -> Tuple[pd.DataFrame, str, str, str, str]:
    df = read_strelka_vcf(cfg["strelka_vcf_path"])
    if df.empty:
        return df, "", "", "", ""

    # fetch protein sequences
    log.info("Fetching protein sequences for Strelka variants..")
    df_uniprot_ids = df["ProteinID"].drop_duplicates().tolist()
    df_sequences = multi_process("fetch_fasta",
                                  df_uniprot_ids,
                                  "ids",
                                  "ncbi_NM",
                                  cfg["ncbi_api_key"])

    # merge sequences with variant data
    df_sequences_df = pd.DataFrame(
        list(zip(df_uniprot_ids, df_sequences)),
        columns=["ProteinID", "Sequence"]
    )
    df = df.merge(df_sequences_df, on="ProteinID", how="left")

    if cfg["strelka_mutation_filter"]:
        # filter out WT sequences of that are not in the reference proteome
        log.info("Filter Strelka variants with reference proteome..")
        # drop sequences that could not be matched with a consensus sequence
        df = filter_to_consensus(df, prepare_reference_proteome(cfg), seq_col="Sequence", id_col="UniProtID")

    return df, "ProteinID", "strelka", "WES mut", "variants"


def load_saav_list(cfg: dict) -> Tuple[pd.DataFrame, str, str, str, str]:
    df = read_saav_list(cfg["saav_list_path"])

    log.info("Fetching protein sequences for SAAVs..")
    df["Sequence"] = multi_process("fetch_fasta",
                                   df["UniProtID"].drop_duplicates().tolist(),
                                   "ids",
                                   "uniprot")

    return df, "UniProtID", "uniprot_mut", "SAAVs", "variants"


def load_uniprot(cfg: dict) -> Tuple[pd.DataFrame, str, str, str, str]:
    ref_ids = pd.read_table(cfg["reference_dataset"]).iloc[:, 0].drop_duplicates()

    log.info("Get mutations from UniProt..")
    anns = multi_process("get_annotation_data",
                         ref_ids,
                         "UniProtID",
                         "ebi",
                         "https://www.ebi.ac.uk/proteins/api/variation/{pid}",
                         cfg["annotation_data"])

    log.info("Process UniProt mutations data..")
    anns = multi_process("get_variant_info",
                         anns,
                         "uni",
                         cfg)

    proteome = read_fasta(cfg["reference_proteome"], "uniprot")[["Identifier","Sequence"]]
    anns = anns.merge(proteome,
                      left_on="DL_UniProtID",
                      right_on="Identifier",
                      how="left").rename(columns={"DL_UniProtID":"UniProtID",
                                                  "DL_VariantPos":"VariantPos",
                                                  "DL_ftID":"VariantID"}).drop(columns="Identifier")
    anns = anns[["UniProtID","VariantPos","VariantID","Sequence"]]

    # drop empty list in VariantPos
    anns = anns[anns["VariantPos"].str.len() > 0]

    # drop entries with A123None or None123A in VariantPos (fix for None values)
    anns = anns[~anns["VariantPos"].apply(lambda x: any(
        [v.startswith("None") or v.endswith("None") for v in x]
    ))]

    # drop entries with A16AA or EP16A in VariantPos (multiple like APG188E or E188APG as well)
    anns = anns[~anns["VariantPos"].apply(lambda x: any(
        [re.match(r"^[A-Z]{2,4}\d+[A-Z]{1,4}$", v) or re.match(r"^[A-Z]{1,4}\d+[A-Z]{2,4}$", v) for v in x]
    ))]

    return anns, "GeneID", "uniprot_mut", "UniProt mutations", "variants"


def load_enst(cfg: dict) -> Tuple[pd.DataFrame, str, str, str, str]:
    df = read_enst_list(cfg["enst_path"])

    # map to UniProt
    log.info("Mapping ENST transcript IDs to UniProt IDs..")
    uniprot = get_uniprot_id(df["TranscriptID"].drop_duplicates().tolist(),
                             fmt_from="Ensembl_Transcript")\
              .rename(columns={"FromID": "TranscriptID",
                               "ToID":   "UniProtID"})
    df = df.merge(uniprot, on="TranscriptID", how="left")
    df["UniProtID"] = df["UniProtID"].fillna("NoUniID")

    # fetch sequences
    log.info("Fetching UniProt sequences..")
    ids  = df.loc[df.UniProtID != "NoUniID", "UniProtID"].drop_duplicates().tolist()
    seqs = multi_process("fetch_fasta",
                         ids,
                         "ids",
                         "uniprot")
    df   = df.merge(pd.DataFrame(list(zip(ids, seqs)),
                                 columns=["UniProtID", "Sequence"]),
                    on="UniProtID", how="left")

    return df, "UniProtID", "uniprot_mut", "ENST mutations", "variants"


def load_illumina_json(cfg: dict) -> Tuple[pd.DataFrame, str, str, str, str]:
    path = cfg.get("illumina_json_path", "")
    if not path or not os.path.exists(path):
        return pd.DataFrame(), "", "", "", ""

    # if gzipped extract in memory (check file type --> read first few bytes)
    root = {}
    file_start = open(path, "rb").read(3)
    if file_start == b'\x1f\x8b\x08':
        with gzip.open(path, "rt") as fh:
            root = json.load(fh)
    else:
        with open(path, "r") as fh:
            root = json.load(fh)

    positions = root.get("positions")
    if positions is None:
        # file may be an array already
        positions = root if isinstance(root, list) else []

    rows = []
    for pos in positions:
        if pos.get("filters") != ["PASS"]:
            continue
        for var in pos.get("variants", []):
            vid = var.get("vid", "")
            for tx in var.get("transcripts", []):
                cons = tx.get("consequence") or []
                if "missense_variant" not in cons:
                    continue
                hgvsp = tx.get("hgvsp") or ""
                hgnc = tx.get("hgnc") or ""
                # expect "...:p.(Thr521Ala)"
                m = re.search(r":p\.\(([^)]+)\)", hgvsp)
                if not m:
                    continue
                aa3 = m.group(1)  # e.g. Thr521Ala
                prot = tx.get("proteinId") or hgvsp.split(":", 1)[0]
                if not prot:
                    continue
                rows.append({"ProteinID": prot, "VariantPos3": aa3, "VariantID": vid, "GeneID": hgnc})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), "", "", "", ""

    aa_dict = {
        'Cys': 'C', 'Asp': 'D', 'Ser': 'S', 'Gln': 'Q', 'Lys': 'K',
        'Ile': 'I', 'Pro': 'P', 'Thr': 'T', 'Phe': 'F', 'Asn': 'N',
        'Gly': 'G', 'His': 'H', 'Leu': 'L', 'Arg': 'R', 'Trp': 'W',
        'Ala': 'A', 'Val': 'V', 'Glu': 'E', 'Tyr': 'Y', 'Met': 'M',
        'Ter': '*', 'del': '-'
    }
    aa_regex = "|".join(aa_dict.keys())

    df["VariantPos"] = (
        df["VariantPos3"]
        .str.replace(aa_regex, lambda x: aa_dict[x.group()], regex=True)
        .apply(lambda s: [s])
    )

    # map ProteinID -> UniProtID
    ensp_ids = df[df.ProteinID.str.startswith("ENSP", na=False)]["ProteinID"].drop_duplicates().tolist()

    maps = []
    if ensp_ids:
        maps.append(
            get_uniprot_id(ensp_ids, fmt_from="Ensembl_Protein", ens_sub=False)
            .rename(columns={"FromID":"ProteinID","ToID":"UniProtID"})
        )
    uni = pd.concat(maps, ignore_index=True) if maps else pd.DataFrame(columns=["ProteinID","UniProtID"])
    df = df.merge(uni, on="ProteinID", how="left")
    df["UniProtID"] = df["UniProtID"].fillna("NoUniID")

    # fetch sequences for UniProt IDs
    ids  = df.loc[df.UniProtID != "NoUniID", "UniProtID"].drop_duplicates().tolist()
    if ids:
        seqs = multi_process("fetch_fasta", ids, "ids", "uniprot")
        df = df.merge(pd.DataFrame(list(zip(ids, seqs)), columns=["UniProtID","Sequence"]),
                      on="UniProtID", how="left")

    # keep only rows with sequence
    df = df.dropna(subset=["Sequence"])

    df = df[["UniProtID", "ProteinID", "VariantID", "GeneID", "VariantPos", "VariantPos3", "Sequence"]]

    return df, "UniProtID", "uniprot_mut", "Illumina mutations", "variants"


def load_vep_json(cfg: dict) -> Tuple[pd.DataFrame, str, str, str, str]:
    path = cfg.get("vep_json_path", "")
    if not path or not os.path.exists(path):
        return pd.DataFrame(), "", "", "", ""

    # load JSONL (NDJSON)
    root: List[dict] = []
    try:
        with open(path, "rb") as fh:
            magic = fh.read(3)
        is_gz = (magic == b"\x1f\x8b\x08")
        opener = (lambda p: gzip.open(p, "rt")) if is_gz else (lambda p: open(p, "rt"))

        with opener(path) as fh:
            for i, ln in enumerate(fh, 1):
                s = (ln or "").strip()
                if not s:
                    continue
                if s in ("[", "]", ","):
                    log.error(f"VEP JSONL reader: invalid token on line {i} ('{s}'). Only JSONL objects are supported.")
                    return pd.DataFrame(), "", "", "", ""
                if s.endswith(","):
                    s = s[:-1].strip()
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError as e:
                    log.error(f"VEP JSONL reader: JSON error on line {i}: {e}")
                    return pd.DataFrame(), "", "", "", ""
                if not isinstance(obj, dict):
                    log.error(f"VEP JSONL reader: line {i} is not a JSON object. Only JSONL objects are supported.")
                    return pd.DataFrame(), "", "", "", ""
                root.append(obj)
    except Exception as e:
        log.error(f"Could not read VEP JSONL file: {e}")
        return pd.DataFrame(), "", "", "", ""

    if not root:
        return pd.DataFrame(), "", "", "", ""

    allowed_terms = {"missense_variant", "inframe_deletion", "inframe_insertion"}

    def _pass_filter(inp: str) -> bool:
        if not inp or "\t" not in inp:
            return True
        parts = inp.split("\t")
        # from VCF: CHROM POS ID REF ALT QUAL FILTER INFO  -> FILTER idx=6
        return len(parts) >= 7 and parts[6] == "PASS"

    def _pick_transcript(tcs: List[dict]) -> Optional[dict]:
        # prefer MANE Select with hgvsp
        # else first protein_coding with hgvsp
        # else first with hgvsp
        if not tcs:
            return None
        # pre-filter by consequence + hgvsp presence
        cand = [t for t in tcs
                if set(t.get("consequence_terms") or []).intersection(allowed_terms)
                and t.get("hgvsp")]
        if not cand:
            return None
        for t in cand:
            # MANE can come as "mane_select" (string) or "mane" (list)
            mane_sel = t.get("mane_select", "")
            mane_list = t.get("mane") or []
            if isinstance(mane_list, list) and "MANE_Select" in mane_list:
                return t
            if isinstance(mane_sel, str) and mane_sel:
                return t
        for t in cand:
            if t.get("biotype") == "protein_coding":
                return t
        return cand[0]

    def _hgvsp_to_parts(hgvsp: str) -> Tuple[Optional[str], Optional[str]]:
        #  split 'NP_001165882.1:p.Ala319_Thr320del' -> (ProteinID, 'Ala319_Thr320del')
        if not hgvsp or ":" not in hgvsp:
            return None, None
        prot, tail = hgvsp.split(":", 1)
        m = re.search(r"p\.\(?([A-Za-z]{3}(?:_[A-Za-z]{3}\d+)?\d+(?:[A-Za-z]{3}|Ter|del|dup))\)?", tail)
        return prot, m.group(1) if m else None

    aa_dict = {
        'Cys': 'C', 'Asp': 'D', 'Ser': 'S', 'Gln': 'Q', 'Lys': 'K',
        'Ile': 'I', 'Pro': 'P', 'Thr': 'T', 'Phe': 'F', 'Asn': 'N',
        'Gly': 'G', 'His': 'H', 'Leu': 'L', 'Arg': 'R', 'Trp': 'W',
        'Ala': 'A', 'Val': 'V', 'Glu': 'E', 'Tyr': 'Y', 'Met': 'M',
        'Ter': '*', 'del': '-'
    }
    aa_regex = "|".join(aa_dict.keys())

    rows: List[dict] = []
    for rec in root:
        try:
            if not _pass_filter(rec.get("input", "")):
                continue
            if rec.get("most_severe_consequence") not in allowed_terms:
                continue

            tx = _pick_transcript(rec.get("transcript_consequences") or [])
            if not tx:
                continue

            prot_id, pos3 = _hgvsp_to_parts(tx.get("hgvsp", ""))
            if not prot_id or not pos3:
                continue

            # drop complex/range HGVS
            low = pos3.lower()
            if "_" in pos3 or "delins" in low or "fs" in low:
                continue
            # insertions/duplications are currently not supported by process_mutation/process_saavs
            if "dup" in low:
                continue

            # convert three-letter AA codes -> one-letter
            pos1 = re.sub(aa_regex, lambda m: aa_dict[m.group(0)], pos3)

            # build one row
            rows.append({
                "ProteinID": prot_id,
                "VariantPos3": pos3,
                "VariantPos": [pos1],
                "GeneID": tx.get("gene_symbol") or tx.get("gene_id") or "NA"
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), "", "", "", ""

    # map to UniProt
    p_np_xp = df[df["ProteinID"].str.startswith(("NP_", "XP_"), na=False)]["ProteinID"].drop_duplicates().tolist()

    # map RefSeq NP_/XP_ -> UniProt
    uniprot_lookup = get_uniprot_id(p_np_xp,
                                    fmt_from="RefSeq_Protein",
                                    split_str=",").rename(columns={"FromID": "ProteinID",
                                                                   "ToID": "UniProtID"})

    df["UniProtID"] = multi_process("process_uniprot_ids",
                                    df["ProteinID"].tolist(),
                                    "ids",
                                    uniprot_lookup)
    df["UniProtID"] = df["UniProtID"].fillna("NoUniID")

    # fetch sequences
    np_xp_ids = df[df["ProteinID"].str.startswith(("NP_", "XP_"), na=False)]["ProteinID"].drop_duplicates().tolist()

    log.info("Fetching NCBI protein sequences for VEP variants..")
    seqs = multi_process("fetch_fasta",
                         np_xp_ids,
                         "ids",
                         "ncbi")
    df = df.merge(pd.DataFrame(list(zip(np_xp_ids, seqs)), columns=["ProteinID", "Sequence"]), on="ProteinID", how="left")

    # keep only rows that have a sequence
    df = df.dropna(subset=["Sequence"])

    # final column order
    df = df[["UniProtID", "ProteinID", "GeneID", "VariantPos", "VariantPos3", "Sequence"]]

    return df, "UniProtID", "uniprot_mut", "VEP mutation", "variants"


@dataclass
class SourceSpec:
    cfg_flag: str
    loader:   Callable[[dict], Tuple[pd.DataFrame, str, str, Optional[str], str]]
    description: str
    tag: str


def run_source_pipeline(src_df: pd.DataFrame,
                        join_key: str,
                        seq_fmt: str,
                        description: str,
                        tag: str,
                        cfg: dict,
                        out_dir: str,
                        ts: str,
                        proc_mut_unit: str,
                        proc_saavs_unit: str) -> None:

    if src_df.empty:
        log.warning(f"{description}: no data – skipped.")
        return

    # keep only proteins present in reference dataset (if requested)
    if cfg["reference_dataset"]:
        log.info("Filter UniProtIDs with reference dataset..")
        src_df = filter_id_with_reference(src_df, cfg)

    # if no reference list drop orphan IDs
    elif not cfg["filter_seq_with_reference_add_no_ids"]:
        log.warning(
            "No reference dataset provided, but "
            "'filter_seq_with_reference_add_no_ids' is False – dropping NoUniID rows.")
        if "UniProtID" in src_df.columns:
            src_df = src_df[src_df["UniProtID"] != "NoUniID"]

    # mutate sequences
    if proc_mut_unit:
        log.info("Generating mutated sequences..")
        src_df = multi_process("process_mutation",
                               src_df,
                               proc_mut_unit,
                               cfg)

    # process SAAVs
    log.info("Processing mutations..")
    src_df = multi_process("process_saavs",
                           src_df,
                           proc_saavs_unit,
                           cfg)

    # filter against the reference proteome
    proteome_available = bool(cfg.get("reference_proteome"))
    proteome = pd.DataFrame()

    # filter peptide sequences against reference proteome (if requested)
    if cfg["filter_seq_with_reference"]:
        log.info("Filter variant peptides with reference proteome..")
        proteome = prepare_reference_proteome(cfg)

        # need a temp folder for the h5 chunks
        os.makedirs(os.path.join(out_dir, "temp"), exist_ok=True)

        h5_paths = multi_process("filter_seq_with_reference",
                                 src_df, "sequences",
                                 proteome, out_dir)

        src_df = pd.concat(cast(List[pd.DataFrame], [pd.read_hdf(p) for p in h5_paths]), ignore_index=True)
    elif proteome_available:
        proteome = prepare_reference_proteome(cfg)

    # add disease / variant annotation if enabled
    if cfg["add_disease_info"]:
        src_df_out = annotate_variant_info(src_df, join_key,
                                           cfg["annotation_data"],
                                           cfg["min_spec_pep_len"],
                                           cfg["max_spec_pep_len"])
        src_df_out.to_csv(os.path.join(out_dir,
                        f"{ts}_disease_annotation_{tag}.tsv"), sep="\t")

    # make fasta records
    recs = convert_df_to_bio_list(src_df, seq_fmt,
                                  cfg["min_spec_pep_len"],
                                  cfg["max_spec_pep_len"],
                                  cfg["keep_saav_dups_in_fasta"])

    # write variant fasta
    log.info(f"Save {description} peptides as FASTA..")
    write_fasta(recs, out_dir, ts, "SAAV_sequences", tag)

    # combined (reference + variant) FASTA only if a proteome is available
    if proteome_available and not proteome.empty:
        prot_recs = convert_df_to_bio_list(proteome, "uniprot")
        combined = prot_recs + recs
        fasta_label = "subFASTA" if cfg["generate_subFASTA"] else "FASTA"
        write_fasta(combined, out_dir, ts, f"{fasta_label}_SAAV", tag)

        # optional standalone subproteome
        if cfg["generate_subFASTA"]:
            write_fasta(prot_recs, out_dir, ts, "subFASTA", tag)

SOURCE_SPECS: List[SourceSpec] = [
    SourceSpec("map_galaxy", load_galaxy, "Galaxy RNAseq mutations", "galaxy"),
    SourceSpec("map_cosmic", load_cosmic, "COSMIC mutations", "cosmic"),
    SourceSpec("map_tso", load_tso, "TSO500 mutations", "tso"),
    SourceSpec("map_mfa", load_mfa, "MFA mutations", "mfa"),
    SourceSpec("map_strelka_vcf", load_strelka, "strelka mutations", "strelka"),
    SourceSpec("map_saav_list", load_saav_list, "SAAV list", "saav_list"),
    SourceSpec("map_uniprot", load_uniprot,"UniProtID mutations",  "uniprot"),
    SourceSpec("map_enst", load_enst, "ENST mutations", "enst"),
    SourceSpec("map_illumina_json", load_illumina_json, "Illumina Connected Annotations", "illumina_json"),
    SourceSpec("map_vep_json", load_vep_json, "Ensembl VEP JSON", "vep_json"),
]


def main() -> None:
    start_time = time.time()
    ts = datetime.now().strftime("%y%m%d%H%M")

    # argument parsing
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", dest="config_yaml",
                        required=True, help="path to config YAML file")
    args = parser.parse_args()

    # welcome banner
    print_welcome()
    stream.setFormatter(ColoredFormatter(LOGFORMAT, "%H:%M:%S"))

    cfg = read_yaml(args.config_yaml)
    out_dir = os.path.dirname(args.config_yaml)
    global config_yaml
    config_yaml = cfg

    # iterate over all sources
    for spec in SOURCE_SPECS:
        if not cfg.get(spec.cfg_flag, False):
            continue

        log.info(f"Processing {spec.description} input..")
        df, join_key, seq_fmt, proc_mut_unit, proc_saavs_unit = spec.loader(cfg)
        if df.empty:
            log.warning(f"{spec.description}: no entries – nothing written.")
            continue
        run_source_pipeline(df, join_key, seq_fmt, spec.description, spec.tag, cfg, out_dir, ts, proc_mut_unit, proc_saavs_unit)

    # tidy up and show total runtime
    shutil.rmtree(os.path.join(out_dir, "temp"), ignore_errors=True)
    mins = round((time.time() - start_time) / 60, 2)
    log.info(f"runtime (total): {mins} min")


if __name__ == "__main__":
    main()
