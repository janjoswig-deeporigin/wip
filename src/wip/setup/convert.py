import pathlib
import shutil

import numpy as np
import parmed
from openbabel import pybel
from rdkit import Chem

from loguru import logger

from wip.setup import tleap


def total_charge(mol, backend="rdkit"):
    """Calculate total charge of a molecule using the specified backend"""
    if backend == "rdkit":
        return total_charge_rdkit(mol)
    elif backend == "openbabel":
        return total_charge_openbabel(mol)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def total_charge_openbabel(mol):
    """Calculate total charge of a molecule using Open Babel."""

    obmol = next(pybel.readfile("mol2", mol))
    return obmol.charge


def total_charge_rdkit(mol):
    raise NotImplementedError("RDKit backend for total_charge is not implemented yet.")


def copy_residue_names_from_prmtop_to_pdb(
    reference_prmtop: str | pathlib.Path,
    target_pdb: str | pathlib.Path,
    output_pdb: str | pathlib.Path,
    remove_hydrogens: bool = True,
    check: bool = True,
    overwrite: bool = True,
    ) -> None:
    """Copy residue names from a prmtop file to a PDB file and save the result

    Note:
        This will only keep ATOM and HETATM lines in the PDB file

    Args:
        reference_prmtop: Path to the reference prmtop file
        target_pdb: Path to the target PDB file
        output_pdb: Path to the output PDB file

    Keyword args:
        remove_hydrogens: Whether to remove hydrogen atoms from the PDB file
        check: Perform basic consistency checks between the reference and target structures
        overwrite: Whether to overwrite the output file if it already exists
    """

    output_pdb = pathlib.Path(output_pdb)
    if output_pdb.exists() and not overwrite:
        raise FileExistsError(f"Output file {output_pdb} already exists (`overwrite=False`)")

    reference_parmed_structure = parmed.load_file(str(reference_prmtop))
    reference_residue_names = np.array([residue.name for residue in reference_parmed_structure.residues])
    if check:
        target_parmed_structure = parmed.load_file(str(target_pdb))
        target_residue_names = np.array([residue.name for residue in target_parmed_structure.residues])
        if len(reference_residue_names) != len(target_residue_names):
            raise ValueError("Reference and target structures have different number of residues")

    tmp_output_pdb = output_pdb.with_suffix(".tmp.pdb")
    with open(target_pdb) as fp, open(tmp_output_pdb, "w") as out_fp:
        last_chainid = None
        last_resid = None
        reference_index = -1
        new_residue = False

        for line in fp:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            if remove_hydrogens and (line[76:78].strip() == "H"):
                continue

            chainid = line[21].strip()
            resid = int(line[22:26].strip())

            if last_chainid != chainid or last_resid != resid:
                new_residue = True
                reference_index += 1
                last_chainid = chainid
                last_resid = resid

            resname = line[17:20].strip()
            new_resname = reference_residue_names[reference_index]

            if resname == new_resname:
                out_fp.write(line)
                continue

            if new_residue:
                logger.info(f"Changing residue {resname} (resid {resid:>3}, chain {chainid}) to {new_resname} at reference index {reference_index:>3}")
                new_residue = False

            out_fp.write(line[:17] + f"{new_resname:>3}" + line[20:])

    shutil.move(tmp_output_pdb, output_pdb)


def reprotonate_pdb_with_reference(
    reference_prmtop: str | pathlib.Path,
    target_pdb: str | pathlib.Path,
    output_pdb: str | pathlib.Path,
    working_directory: str | pathlib.Path | None = None,
) -> None:
    """Protonate a PDB file using a reference prmtop file

    Args:
        reference_prmtop: Path to the reference prmtop file
        target_pdb: Path to the target PDB file
        output_pdb: Path to the output PDB file
        working_directory: Path to a working directory (will be passed down to
            tleap execution).
    """

    copy_residue_names_from_prmtop_to_pdb(
        reference_prmtop=reference_prmtop,
        target_pdb=target_pdb,
        output_pdb=output_pdb,
        remove_hydrogens=True,
        check=True,
        overwrite=True,
    )

    tleap.run_tleap_template(
        template="setup_protein",
        input_pdb=output_pdb,
        output_pdb=output_pdb,
        working_directory=working_directory,
    )