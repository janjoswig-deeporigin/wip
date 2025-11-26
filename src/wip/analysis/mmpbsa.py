import logging
import pathlib
import shlex
import subprocess
from collections import defaultdict
from collections.abc import Container, Iterable
from io import StringIO

import pandas as pd
from pydantic import BaseModel

from awsem.analysis.interfaces import (
    Analysis,
    AnalysisBackend,
    AnalysisInput,
    BackendParameters,
)
from awsem.infrastructure import check_returncode
from awsem.typing import StrPath


# AMBER mask to obtain a complex stripped of solvent and ions
DEFAULT_SOLVENT_STRIP_MASK = (
    ":WAT,:HOH,:TIP3,:TIP3P,Cl*,CIO,Cs+,IB,K*,Li+,MG*,Na+,Rb+,CS,RB,NA,F,CL"
)

# AMBER masks to obtain receptor and ligand from complex
DEFAULT_RECEPTOR_STRIP_MASK = "!:LIG"
DEFAULT_LIGAND_STRIP_MASK = ":LIG"


# Pydantic models for AmberTools MMPBSA input parameters
# NOTE: Option names correspond 1:1 to those used in MMPBSA.py input files (mmpbsa.in) for AmberTools 25.
#       A complete list of options can be printed with `MMPBSA.py --input-file-help`.
#       They might actually be slightly different between versions of AmberTools.
#       `None` values mean that the option is not written to the input file and hence the internal
#       default of MMPBSA.py is used.
class AmbertoolsParametersGeneral(BackendParameters):
    debug_printlevel: int | None = None
    endframe: int | None = None
    entropy: int | None = None
    full_traj: int | None = None
    interval: int | None = None
    keep_files: int | None = None
    ligand_mask: str | None = None
    netcdf: int | None = None
    receptor_mask: str | None = None
    search_path: int | None = None
    startframe: int | None = None
    strip_mask: str | None = None
    use_sander: int | None = None
    verbose: int | None = None


class AmbertoolsParametersGB(BackendParameters):
    ifqnt: int | None = None
    igb: int | None = None
    qm_theory: str | None = None
    qm_residues: str | None = None
    qmcharge_com: int | None = None
    qmcharge_lig: int | None = None
    qmcharge_rec: int | None = None
    qmcut: float | None = None
    saltcon: float | None = None
    surfoff: float | None = None
    surften: float | None = None
    molsurf: int | None = None
    msoffset: float | None = None
    probe: float | None = None
    b: float | None = None
    alpb: int | None = None
    epsin: float | None = None
    epsout: float | None = None
    istrng: float | None = None
    Rs: float | None = None
    space: float | None = None
    arcres: float | None = None
    rbornstat: int | None = None
    dgij: int | None = None
    radiopt: int | None = None
    chagb: int | None = None
    ROH: float | None = None
    tau: float | None = None


class AmbertoolsParametersPB(BackendParameters):
    cavity_offset: float | None = None
    cavity_surften: float | None = None
    exdi: float | None = None
    emem: float | None = None
    memopt: int | None = None
    memoptzero: int | None = None
    sasopt: int | None = None
    solvopt: int | None = None
    mthick: float | None = None
    mctrdz: float | None = None
    maxarcdot: int | None = None
    poretype: int | None = None
    npbverb: int | None = None
    nfocus: int | None = None
    bcopt: int | None = None
    eneopt: int | None = None
    frcopt: int | None = None
    cutfd: float | None = None
    cutnb: float | None = None
    ipb: int | None = None
    fillratio: float | None = None
    indi: float | None = None
    inp: int | None = None
    istrng: float | None = None
    linit: int | None = None
    prbrad: float | None = None
    radiopt: int | None = None
    sander_apbs: int | None = None
    scale: float | None = None


class AmbertoolsParametersAlanineScanning(BackendParameters):
    mutant_only: int | None = None


class AmbertoolsParametersNormalmode(BackendParameters):
    dielc: float | None = None
    drms: float | None = None
    maxcyc: int | None = None
    nminterval: int | None = None
    nmendframe: int | None = None
    nmode_igb: int | None = None
    nmode_istrng: float | None = None
    nmstartframe: int | None = None


class AmbertoolsParametersDecomposition(BackendParameters):
    csv_format: int | None = None
    dec_verbose: int | None = None
    idecomp: int = 1
    print_res: str | None = None


class AmbertoolsParameters3DRISM(BackendParameters):
    buffer: float | None = None
    closure: str | None = None
    closureorder: int | None = None
    grdspc: float | None = None
    ng: str | None = None
    polardecomp: int | None = None
    rism_verbose: int | None = None
    solvbox: str | None = None
    solvcut: float | None = None
    thermo: str | None = None
    tolerance: float | None = None


class AmbertoolsParameters(BackendParameters):
    general: AmbertoolsParametersGeneral | None = None
    gb: AmbertoolsParametersGB | None = None
    pb: AmbertoolsParametersPB | None = None
    alanine_scanning: AmbertoolsParametersAlanineScanning | None = None
    nmode: AmbertoolsParametersNormalmode | None = None
    decomposition: AmbertoolsParametersDecomposition | None = None
    rism: AmbertoolsParameters3DRISM | None = None


# Pydantic models for gmx_MMPBSA input parameters
# NOTE: Option names correspond 1:1 to those used in gmx_MMPBSA input files (mmpbsa.in) version x.y.z.
class GMXMMPBSAParameters(BackendParameters):
    pass


class AmbertoolsBackend(AnalysisBackend):
    """Backend for MMPBSA analysis leveraging AmberTools' `MMPBSA.py`.

    Use cases:
       * Complex only -> stability analysis
       * (Solvated complex), complex, receptor, ligand -> binding energy analysis
    """

    class Parameters(BackendParameters):
        exe: StrPath = "MMPBSA.py"
        settings: AmbertoolsParameters = AmbertoolsParameters()
        parse_categories: set[str] | None = None

    def prepare_settings(
        self, main: Analysis, working_directory: pathlib.Path
    ) -> pathlib.Path:
        """Prepare MMPBSA input parameter file

        Based on the parameters in :attr:`main.parameters.backend_parameters`.

        Args:
            main: The main MMPBSA analysis object.
            working_directory: Path to the working directory.

        Returns:
            Path to the generated MMPBSA input parameter file
            ("mmpbsa.in" in the working directory).
        """

        file_path = working_directory / "mmpbsa.in"
        settings = main.parameters.backend_parameters.settings
        write_mmpbsa_input_file(
            file_path,
            settings,
            comment="# MMPBSA.py input file generated by AWSEM wrapper",
        )

        return file_path

    def run(self, main: Analysis) -> None:
        """Run the analysis.

        Prepares input parameters, collects input files, calls AmberTools `MMPBSA.py`,
        parses the output, and performs cleanup if needed.

        Args:
            main: The main MMPBSA analysis object.
        """

        working_dir_info = main.get_working_directory()
        if (
            working_dir_info.is_temporary
            and main.parameters.cleanup
            and not main.parameters.parse_output
        ):
            logging.warning(
                "Working directory is temporary and will be cleaned up, "
                "but `parse_output` is `False`. "
                "Aborting because no results would be available after analysis."
            )
            main.cleanup()
            return

        working_dir = working_dir_info.path
        settings_file = self.prepare_settings(main, working_dir)

        check_returncode(verbose=True)(run_mmpbsa_ambertools)(
            exe=main.parameters.backend_parameters.exe,
            settings=settings_file,
            **dict(main.input),
            overwrite=True,
            cwd=working_dir,
            env=main.parameters.env,
        )

        if main.parameters.parse_output:
            # NOTE: Averaged statistics are not actually parsed yet but just stored as raw text
            results_file = working_dir / "results.out"
            if results_file.is_file():
                main.results["energies_stats"] = results_file.read_text()

            framewise_file = working_dir / "results_frames.out"
            if framewise_file.is_file():
                main.results["energies_framewise"] = self.parse_framewise_results(
                    framewise_file,
                    sub_categories=main.parameters.backend_parameters.parse_categories,
                    index_col=0,
                )

            avg_decomp_file = working_dir / "results_decomposition.out"
            if avg_decomp_file.is_file():
                main.results["decomposition_stats"] = (
                    self.parse_average_decomposition_results(avg_decomp_file)
                )

            # NOTE: Frame-wise decomposition results are not parsed yet

        if main.parameters.cleanup:
            main.cleanup()

    @staticmethod
    def parse_framewise_results(
        file: StrPath,
        sub_categories: Container[str] | None = None,
        **kwargs,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Parse the frame-wise energy results file from MMPBSA.py

        Create a data frame for each sub-table found. Super-category
        names (e.g. `"generalized_born"`, `"poisson_boltzmann"`) and
        sub-category names (e.g. `"ligand_energy_terms"`,
        `"delta_energy_terms"`) are converted to lowercase
        and spaces are replaced with underscores.

        Args:
            file: Path to the frame-wise results output file.
            sub_categories: If given, only these sub-categories
                will be read.
            kwargs: Additional keyword arguments passed to :func:`pandas.read_csv`
                to customise how sub-tables are read. Note, though that
                these tables differ over super-categories and sub-categories
                in terms of which columns are present.

        Returns:
            Nested dictionary of super-category names mapping to
                sub-category names mapping to Pandas data frames.
        """

        with open(file) as fp:
            # Relies on categories being separated by two empty lines
            super_tables = fp.read().split("\n\n\n")

        results = defaultdict(dict)
        for table in super_tables[:-1]:
            super_category, rest = table.split("\n", 1)
            super_category = super_category.rstrip(":").lower().replace(" ", "_")
            subtables = rest.split("\n\n")
            for stable in subtables:
                sub_category, rest = stable.split("\n", 1)
                sub_category = sub_category.rstrip(":").lower().replace(" ", "_")
                if (sub_categories is None) or (sub_category in sub_categories):
                    results[super_category][sub_category] = pd.read_csv(
                        StringIO(rest), **kwargs
                    )

        return dict(results)

    @staticmethod
    def parse_average_decomposition_results(
        file: StrPath,
    ) -> dict[str, pd.DataFrame]:
        """Parse the average decomposition output file from MMPBSA.py

        Create a data frame for each sub-table found. Super-category
        names (e.g. `"generalized_born"`, `"poisson_boltzmann"`) are
        are converted to lowercase. Sub-categories like for
        total energy terms do not exist here

        Note that the format of these files is a bit unwieldy and
        a few hard-coded assumptions are made for the parsing so there
        are no additional customisation options provided.

        Args:
            file: Path to the average decomposition results output file.

        Returns:
            Dictionary of super-category names mapping to
            Pandas data frames.
        """

        with open(file) as fp:
            # The idecomp setting is always required so we take it as a sign that a new table starts here
            super_tables = fp.read().split("idecomp")

        results = {}
        for table in super_tables[1:]:
            _, super_category, *_, header1, header2, rest = table.split("\n", 6)
            super_category = (
                super_category.split(":")[1]
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("_solvent", "")
            )

            # Headers are spread over two lines and are incomplete
            # We also replace bulky column names
            header2 += ",Avg.,Std. Dev.,Std. Err. of Mean"
            header1 = (
                header1.replace("Avg.", "avg")
                .replace("Std. Dev.", "std")
                .replace("Std. Err. of Mean", "sem")
            )
            header2 = (
                header2.replace("Avg.", "avg")
                .replace("Std. Dev.", "std")
                .replace("Std. Err. of Mean", "sem")
            )

            column_names = []
            last_a = ""
            for a, b in zip(header1.split(","), header2.split(",")):
                if not b:
                    column_names.append(a)
                else:
                    if not a:
                        a = last_a
                    column_names.append(f"{a} ({b.strip()})")
                last_a = a

            results[super_category] = pd.read_csv(
                StringIO(rest), header=None, names=column_names
            )

        return results


class MMPBSA(Analysis):
    """Perform MMPBSA analysis on a given system"""

    class Input(AnalysisInput):
        solvated_topology_file: StrPath | None = None
        complex_topology_file: StrPath
        receptor_topology_file: StrPath | None = None
        ligand_topology_file: StrPath | None = None
        structure_files: StrPath | list[StrPath]
        receptor_structure_files: StrPath | list[StrPath] | None = None
        ligand_structure_files: StrPath | list[StrPath] | None = None

    _registered_backends = {
        "default": AmbertoolsBackend,
        "ambertools": AmbertoolsBackend,
        "mmpbsa.py": AmbertoolsBackend,
    }

    def __init__(self, *args, backend="MMPBSA.py", **kwargs) -> None:
        super().__init__(*args, backend=backend, **kwargs)


def write_mmpbsa_input_file(
    file_path: StrPath, settings: BaseModel, comment: str = None
) -> None:
    """Write an MMPBSA input parameter file based on a Pydantic model.

    Args:
        file_path: Path to the output MMPBSA input parameter file. Will be overwritten
            if it already exists.
        settings: Pydantic model containing MMPBSA input parameters.
        comment: Optional comment to add at the top of the file.
    """

    with open(file_path, "w") as fp:
        if comment is not None:
            fp.write(comment + "\n")

        for field, value in settings:
            if value is not None:
                fp.write(f"&{field}\n")
                for option_name, option_value in value:
                    if option_value is not None:
                        fp.write(f"  {option_name}={option_value},\n")
                fp.write("/\n")


def run_ante_mmpbsa_ambertools(
    *,
    exe: StrPath = "ante-MMPBSA.py",
    topology_file: StrPath,
    output_complex_topology: StrPath | None = None,
    output_receptor_topology: StrPath | None = None,
    output_ligand_topology: StrPath | None = None,
    complex_strip_mask: StrPath | None = None,
    receptor_strip_mask: StrPath | None = None,
    ligand_strip_mask: StrPath | None = None,
    radii: str | None = None,
    overwrite: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Make a CLI call to the ante-MMPBSA preparation script in AmberTools

    Exposes most of the command-line arguments of `ante-MMPBSA.py`.

    Keyword args:
        exe: Path to ante-MMPBSA.py executable
        topology_file: Input "dry" complex topology or solvated complex topology
        output_complex_topology: Complex topology file created by
            removing `solvent_strip_mask` (solvent and ions) from `topology_file`
        output_receptor_topology: Receptor topology file created by taking
            `receptor_strip_mask` out of complex
        output_ligand_topology: Ligand topology file created by taking
            `ligand_strip_mask` out of complex
        solvent_strip_mask: Amber mask of atoms needed to be removed from
            `topology_file` to make the `output_complex_topology`. If not specified,
            but `output_complex_topology` is given, a :obj:`DEFAULT_SOLVENT_STRIP_MASK`
            is used.
        receptor_strip_mask: Amber mask of atoms kept in complex to make
            the `output_receptor_topology`. Cannot specify with `ligand_strip_mask`.
            If neither `receptor_strip_mask` nor `ligand_strip_mask` is given, but
            `output_receptor_topology` is given, a :obj:`DEFAULT_RECEPTOR_STRIP_MASK`
            is used.
        ligand_strip_mask: Amber mask of atoms kept in complex to make
            the `output_ligand_topology`. Cannot specify with `receptor_strip_mask`.
            If neither `receptor_strip_mask` nor `ligand_strip_mask` is given, but
            `output_ligand_topology` is given, a :obj:`DEFAULT_LIGAND_STRIP_MASK`
            is used.
        radii: PB/GB Radius set to set in the generated topology files.
            This is equivalent to "set PBRadii <radius>" in LEaP.
            Options are bondi, mbondi2, mbondi3, amber6, and mbondi
            and the default is to use the existing radii.
        overwrite: Whether to overwrite existing output files
        **kwargs: Additional keyword arguments passed to :meth:`subprocess.run`
    """

    command = f"{exe} -p {topology_file} "

    if output_complex_topology is not None:
        output_complex_topology = pathlib.Path(output_complex_topology)
        command += f"-c {output_complex_topology} "

        if output_complex_topology.is_file():
            if overwrite:
                output_complex_topology.unlink()
            else:
                raise FileExistsError(
                    f"Output complex topology file {output_complex_topology} "
                    "already exists. Use `overwrite=True` to overwrite."
                )

        if complex_strip_mask is None:
            complex_strip_mask = DEFAULT_SOLVENT_STRIP_MASK

    if complex_strip_mask is not None:
        command += f"-s {complex_strip_mask} "

    if output_receptor_topology is not None:
        output_receptor_topology = pathlib.Path(output_receptor_topology)
        command += f"-r {output_receptor_topology} "

        if output_receptor_topology.is_file():
            if overwrite:
                output_receptor_topology.unlink()
            else:
                raise FileExistsError(
                    f"Output receptor topology file {output_receptor_topology} "
                    "already exists. Use `overwrite=True` to overwrite."
                )

        if receptor_strip_mask is None and ligand_strip_mask is None:
            receptor_strip_mask = DEFAULT_RECEPTOR_STRIP_MASK

    if output_ligand_topology is not None:
        output_ligand_topology = pathlib.Path(output_ligand_topology)
        command += f"-l {output_ligand_topology} "

        if output_ligand_topology.is_file():
            if overwrite:
                output_ligand_topology.unlink()
            else:
                raise FileExistsError(
                    f"Output ligand topology file {output_ligand_topology} "
                    "already exists. Use `overwrite=True` to overwrite."
                )

        if receptor_strip_mask is None and ligand_strip_mask is None:
            ligand_strip_mask = DEFAULT_LIGAND_STRIP_MASK

    if receptor_strip_mask is not None:
        command += f"-m {receptor_strip_mask} "

    if ligand_strip_mask is not None:
        command += f"-n {ligand_strip_mask} "

    if radii is not None:
        command += f"-radii {radii} "

    process = subprocess.run(
        shlex.split(command), capture_output=True, text=True, **kwargs
    )

    return process


def run_mmpbsa_ambertools(
    *,
    exe: StrPath = "MMPBSA.py",
    settings: StrPath = "mmpbsa.in",
    complex_topology_file: StrPath = "complex.prmtop",
    solvated_topology_file: StrPath | None = None,
    receptor_topology_file: StrPath | None = None,
    ligand_topology_file: StrPath | None = None,
    solvated_receptor_topology_file: StrPath | None = None,
    solvated_ligand_topology_file: StrPath | None = None,
    structure_files: StrPath | Iterable[StrPath],
    receptor_structure_files: StrPath | Iterable[StrPath] | None = None,
    ligand_structure_files: StrPath | Iterable[StrPath] | None = None,
    output_file: StrPath = "results.out",
    output_file_decomposition: StrPath = "results_decomposition.out",
    output_file_frames: StrPath = "results_frames.out",
    output_file_decomposition_frames: StrPath = "results_decomposition_frames.out",
    overwrite: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Make a CLI call to the MMPBSA analysis in AmberTools

    Exposes most of the command-line arguments of `MMPBSA.py`.

    Keyword args:
        exe: Path to MMPBSA.py executable
        settings: Path to MMPBSA input parameter file
        complex_topology_file: Path to complex prmtop file
        solvated_topology_file: Optional path to solvated prmtop file needed
            when `structure_files` are solvated structures. The value of
            the `"strip_mask"` input parameter should yield the complex.
        receptor_topology_file: Optional path to receptor prmtop file
        ligand_topology_file: Optional path to ligand prmtop file
        solvated_receptor_topology_file: Optional path to solvated receptor prmtop file
            needed when `receptor_structure_files` are solvated structures.
            The value of the `"strip_mask"` input parameter should yield the
            receptor.
        ligand_solvated_topology_file: Optional path to solvated ligand prmtop file
            needed when `ligand_structure_files` are solvated structures.
            The value of the `"strip_mask"` input parameter should yield the
            ligand.
        structure_files: Path(s) to structure files (e.g. trajectory or
            coordinate files) for the complex or solvated system.
        receptor_structure_files: Optional path(s) to structure files for the
            receptor only.
        ligand_structure_files: Optional path(s) to structure files for the
            ligand only.
        overwrite: Whether to overwrite existing output files
        **kwargs: Additional keyword arguments passed to :meth:`subprocess.run`
    """

    command = f"{exe} "
    if overwrite:
        command += "-O "

    command += (
        f"-i {settings} "
        f"-o {output_file} "
        f"-eo {output_file_frames} "
        f"-do {output_file_decomposition} "
        f"-deo {output_file_decomposition_frames} "
        f"-cp {complex_topology_file} "
    )

    if receptor_topology_file is not None:
        command += f"-rp {receptor_topology_file} "
    if ligand_topology_file is not None:
        command += f"-lp {ligand_topology_file} "
    if solvated_topology_file is not None:
        command += f"-sp {solvated_topology_file} "
    if solvated_receptor_topology_file is not None:
        command += f"-srp {solvated_receptor_topology_file} "
    if solvated_ligand_topology_file is not None:
        command += f"-slp {solvated_ligand_topology_file} "

    if isinstance(structure_files, str) or not isinstance(structure_files, Iterable):
        structure_files = [structure_files]

    for f in structure_files:
        command += f"-y {f} "

    if receptor_structure_files is not None:
        if isinstance(receptor_structure_files, str) or not isinstance(
            receptor_structure_files, Iterable
        ):
            receptor_structure_files = [receptor_structure_files]

        for f in receptor_structure_files:
            command += f"-yr {f} "

    if ligand_structure_files is not None:
        if isinstance(ligand_structure_files, str) or not isinstance(
            ligand_structure_files, Iterable
        ):
            ligand_structure_files = [ligand_structure_files]

        for f in ligand_structure_files:
            command += f"-yl {f} "

    process = subprocess.run(
        shlex.split(command), capture_output=True, text=True, **kwargs
    )

    return process
