import ftplib
import os
import gzip
import shutil
import sys

import pandas as pd
from Bio import SeqIO
from ProteoGenDB import read_fasta, convert_df_to_bio_list

# create temp dir
tmp_dir = "ensembl_merger_temp"
os.mkdir(tmp_dir)

# download everything from http://ftp.ensembl.org/pub/
ftp = ftplib.FTP("ftp.ensembl.org")
ftp.login("anonymous", "")
ftp.cwd("pub")

files = []

organism_suffix = "Mus_musculus.GRCm39_{}.pep.all.fa.gz"  # "Homo_sapiens.GRCh38_{}.pep.all.fa.gz"
organism_suffix_fa = "Mus_musculus.GRCm39_{}.pep.all.fa"  # "Homo_sapiens.GRCh38_{}.pep.all.fa"
organism_path = "{}/fasta/mus_musculus/pep/Mus_musculus.GRCm39.pep.all.fa.gz"  # "{}/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz"
organism_combined_fa = "combined_Mus_musculus.GRCm39.pep.all.fa"  # "combined_Homo_sapiens.GRCh38.pep.all.fa"

try:
    files = ftp.nlst()
except ftplib.error_perm as resp:
    if str(resp) == "550 No files found":
        print("No files in this directory")
    else:
        raise

for f in files:
    if "release-" in f:
        print("Download {}".format(f))

        # reformat version
        if len(f.split("-")[1]) == 2:
            f_zero = "0"
        else:
            f_zero = ""

        f_new = "{}-{}{}".format(f.split("-")[0], f_zero, f.split("-")[1])

        try:
            # download gz
            with open(os.path.join(tmp_dir, organism_suffix.format(f_new)), 'wb') as ftp_file:
                def callback(data):
                    ftp_file.write(data)

                ftp.retrbinary("RETR " + organism_path.format(f),
                               callback)

            # extract gz
            with gzip.open(os.path.join(tmp_dir, organism_suffix.format(f_new)), "rb") as fasta_gz:
                with open(os.path.join(tmp_dir, organism_suffix_fa.format(f_new)), 'wb') as fasta_out:
                    shutil.copyfileobj(fasta_gz, fasta_out)

            # delete gz
            os.remove(os.path.join(tmp_dir, organism_suffix.format(f_new)))
        except ftplib.error_perm as resp:
            print("Error: {}".format(resp))
            os.remove(os.path.join(tmp_dir, organism_suffix.format(f_new)))

fasta_df = None
current_fasta = True
for fasta in sorted(os.listdir(tmp_dir), reverse=True):
    if fasta.endswith(".fa"):
        print(os.path.join(tmp_dir, fasta))

        if current_fasta:
            fasta_df = read_fasta(os.path.join(tmp_dir, fasta), "ensembl_merge")
            current_fasta = False
        else:
            fasta_df_temp = read_fasta(os.path.join(tmp_dir, fasta), "ensembl_merge")
            fasta_df = pd.concat([fasta_df, fasta_df_temp]).groupby('ProteinID', as_index=False).first()
        os.remove(os.path.join(tmp_dir, fasta))

fasta_bio_list = convert_df_to_bio_list(fasta_df, "ensembl")

with open(organism_combined_fa, "w") as fasta_out:
    SeqIO.write(fasta_bio_list, fasta_out, "fasta")

os.rmdir(tmp_dir)

sys.exit()
