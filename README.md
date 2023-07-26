# ProteoGenDB

![ProteoGenDB pipeline](res/ProteoGenDB_pipeline.png?raw=true "ProteoGenDB pipeline")

ProteoGenDB is designed for generating custom peptide databases in the context of mass spectrometry-based proteomics. The primary aim of this tool is to identify and characterize protein variants, which remain a major challenge in modern proteomics. These variants often have important functional and clinical implications, providing insight into the genetic alterations driving cancer development and progression.

Leveraging proteogenomics, transcriptomics, and other annotation data, ProteoGenDB creates a peptide-centric FASTA database of single amino acid variants (SAAVs) using input data from various sources. These sources include processed RNAseq data from a custom Galaxy workflow, variant information from the COSMIC database, the output from Illumina’s TruSight Oncology 500 gene panel and isoform fasta databases from UniProt. By cross-referencing processed peptide variants with the UniProt database, the tool identifies variants previously reported to be associated with diseases or other clinically relevant conditions.

ProteoGenDB offers users the option to filter peptide variants against a list of proteins identified in the dataset of interest, generating a subsetted reference proteome that can further filter peptide variants. This approach reduces the filter penalty of peptide variants, as the targeted proteome contains only identified proteins. Configuration of the tool is facilitated through a YAML file, which allows users to specify input datasets, filtering options, disease annotation, and output formats.
## Requirements

- Python 3.9
- pandas 1.5.3
- NumPy 1.24.2
- pytables 3.8.0
- Biopython 1.81
- pyarrow 11.0.0
- Requests 2.25.1
- PyYAML 6.0
- BeautifulSoup4 4.11.2
- colorlog 6.7.0
- tqdm 4.64.1

## Installation

1. Clone the repository:

```bash
git clone https://github.com/npinter/ProteoGenDB.git
cd ProteoGenDB
```

2. Create a conda environment with the required packages:

```bash
conda create -n ProteoGenDB -c conda-forge python=3.9 pandas=1.5.3 numpy=1.24.2 pytables=3.8.0 biopython=1.81 pyarrow=11.0.0 requests=2.25.1 PyYAML=6.0 beautifulsoup4=4.11.2 colorlog=6.7.0 tqdm=4.64.1
```

3. Activate the conda environment:

```bash
conda activate ProteoGenDB
```

## Usage

1. Edit the `ProteoGenDB.config.yaml` file to specify the input files and parameters for the database generation.

2. Run the tool in the activated conda environment:

```bash
python ProteoGenDB.py -c path/to/your/ProteoGenDB.config.yaml
```

The output files, including the annotated SAAV sequences and the disease annotations, will be saved in the same directory as the `ProteoGenDB.config.yaml` file.

## Configuration

The `ProteoGenDB.config.yaml` file contains various parameters for controlling the database generation process. Users can specify input datasets, filtering options, and output settings. Detailed descriptions are provided in the config file.

## Ensembl Merger

The `ensembl_merger.py` script is designed to merge Ensembl protein sequence files from different releases into a single output file. The output file is a mandatory input for ProteoGenDB to map variants from the Galaxy Workflow output to UniProt IDs. This script is helpful when building a custom protein database containing sequences from multiple releases of the same organism.

### Usage

1. Edit the `organism_suffix`, `organism_suffix_fa`, `organism_path`, and `organism_combined_fa` variables in the script to match the organism for which you want to download and merge protein sequences.
    - For example, for Homo sapiens, use the following values:
        ```
        organism_suffix = "Homo_sapiens.GRCh38_{}.pep.all.fa.gz"
        organism_suffix_fa = "Homo_sapiens.GRCh38_{}.pep.all.fa"
        organism_path = "{}/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz"
        organism_combined_fa = "combined_Homo_sapiens.GRCh38.pep.all.fa"
        ```

2. Run the `ensembl_merger.py` script:

```bash
python ensembl_merger.py
```

The script will download protein sequences from the specified Ensembl releases, merge them, and save the combined protein sequences in a single FASTA file, as specified by the `organism_combined_fa` variable.

3. Use the output file from `ensembl_merger.py` as input for ProteoGenDB:

Edit the `ProteoGenDB.config.yaml` file and set the `fasta_ensembl` parameter to the path of the output file generated by `ensembl_merger.py`:

```
fasta_ensembl: "./input/combined_Homo_sapiens.GRCh38.pep.all.fa"
```
# Galaxy workflows
## SRA IDs (paired-end) workflow

![Galaxy workflow](res/SRA_ID_Galaxy_WF.png?raw=true "Galaxy workflow")

The SRA IDs (paired-end) workflow is designed to process and analyze paired-end RNAseq data from the Sequence Read Archive (SRA) database in NCBI. This workflow is a vital input source for ProteoGenDB, as it allows users to find a cohort in the SRA database that matches their cohort of interest, in terms of tumor entity and study design, and then use the identified cohort to generate protein variants.

The workflow consists of several steps that perform tasks such as data input, read extraction, alignment, quality control, variant calling, and database generation. The output is a custom protein database (FASTA) that can be used in ProteoGenDB to generate a peptide-centric SAAV database.

Please visit [usegalaxy.eu](https://usegalaxy.eu) for more information on how to use Galaxy. 

### How to use the workflow

1. Import the Galaxy workflow via [https://usegalaxy.eu/u/nikopinter/w/proteogenomics-db-gen-v12-sra-paired](https://usegalaxy.eu/u/nikopinter/w/proteogenomics-db-gen-v12-sra-paired).
2. Find a cohort on the [SRA database](https://www.ncbi.nlm.nih.gov/sra) that matches your cohort in terms of tumor entity and study design.
3. Import the SRA accession IDs into the workflow (upload with "Paste/Fetch data").
5. Run the workflow.
6. Once the workflow has finished, you will have a proteoform FASTA that can be used as input for ProteoGenDB.

### Other workflows
- for single-end RNAseq data [SRA ID (single-end)](https://usegalaxy.eu/u/nikopinter/w/proteogenomics-db-gen-v12-sra-ids-single)

## FASTQ workflow

For patient-matched RNAseq data use this workflow. It supports paired-end RNAseq data in `*.fq.gz` and `*.fastq.gz` file formats (single-end data is WIP).

### Paired-end RNAseq data

1. Make sure your paired-end RNAseq data files use the supported naming scheme: the file names should include either "1" or "R1" for forward reads and "2" or "R2" for reverse reads, separated by an underscore from the sample ID. For example: `sampleID_1.fq.gz` and `sampleID_2.fq.gz`.

2. Upload your RNAseq reads to the Galaxy FTP server by following this guide: [Galaxy FTP server connection](https://galaxyproject.org/ftp-upload/).

3. In the Galaxy interface, create a new History.

4. Upload the files as a Collection (choose the 'Collection' option, not the 'Regular' mode in the Galaxy FTP guide). Select `fastqsanger.gz` as the file type and click 'Start' followed by 'Build'.

5. Name your collection and click 'Create collection'.

6. After the import job is complete, go to the [Proteogenomics-DB-Gen-v1.3-FASTQ-PE workflow](https://usegalaxy.eu/u/nikopinter/w/proteogenomics-db-gen-v13-fastq-pe) and click the play button in the top right corner.

7. Make sure the collection is selected as the input dataset (select the collection at "1: Input dataset collection") and click 'Run Workflow' to start processing.

8. Once the workflow has finished, you will have a proteoform FASTA that can be used as input for ProteoGenDB.

### Single-end RNAseq data

**Note:** The single-end RNAseq data workflow is currently a work in progress.