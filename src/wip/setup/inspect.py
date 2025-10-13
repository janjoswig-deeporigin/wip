import pathlib
import shutil
from typing import Callable, Iterable

import numpy as np
import parmed
from openbabel import pybel
from rdkit import Chem

from loguru import logger


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


def get_property_rdkit(
    supplier,
    property_names: str | list[str],
    convert_to: Callable | list[Callable] | None = None,
    raise_errors: bool = True
):
    """Get a specific property from molecules in an SDF file using RDKit"""

    if isinstance(supplier, (str, pathlib.Path)):
        sdf_file = pathlib.Path(supplier)
        if not sdf_file.is_file():
            raise FileNotFoundError(f"SDF file {sdf_file} not found")
        supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False)

    if isinstance(property_names, str) or not isinstance(property_names, Iterable):
        property_names = [property_names]

    if not isinstance(convert_to, Iterable):
        convert_to = [convert_to]
    if len(convert_to) == 1:
        convert_to = convert_to * len(property_names)
    if len(convert_to) != len(property_names):
        raise ValueError("Length of `convert_to` must be 1 or equal to length of `property_names`")

    for i, mol in enumerate(supplier):
        if mol is None:
            if raise_errors:
                raise ValueError(f"{sdf_file.name}: could not read molecule with index {i}")
            else:
                logger.warning(f"{sdf_file.name}: could not read molecule with index {i}")
            yield [None] * len(property_names)
            continue

        values = []
        for j, pname in enumerate(property_names):
            if not mol.HasProp(pname):
                if raise_errors:
                    raise LookupError(f"{sdf_file.name}: property '{pname}' not found for molecule with index {i}. Available properties: {list(mol.GetPropNames())}")
                else:
                    logger.warning(f"{sdf_file.name}: property '{pname}' not found for molecule with index {i}. Available properties: {list(mol.GetPropNames())}")
                values.append(None)
                continue

            value = mol.GetProp(pname)
            if convert_to[j] is None:
                values.append(value)
                continue

            try:
                value = convert_to[j](value)
            except Exception as exc:
                if raise_errors:
                    raise ValueError(f"{sdf_file.name}: could not convert property '{pname}' to desired type") from exc
                else:
                    logger.warning(f"{sdf_file.name}: could not convert property '{pname}' to desired type: {exc}")
            values.append(value)

        yield values
