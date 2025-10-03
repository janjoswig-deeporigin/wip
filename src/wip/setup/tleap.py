import pathlib
import shlex
import subprocess
from collections.abc import Iterable
from typing import Callable


def get_template_setup_protein_pdb(
    input_pdb: str | pathlib.Path,
    output_pdb: str | pathlib.Path | None = None,
    output_rst: str | pathlib.Path | None = None,
    output_prmtop: str | pathlib.Path | None = None,
    sources: str | Iterable[str] = "leaprc.protein.ff14SB"
) -> str:
    """Generate a tleap template for preparing a protein PDB file

    Note:
        At least one output file should be specified otherwise the script will
        run but produce no output

    Args:
        input_pdb: Path to the input PDB file
    
    Keyword args:
        output_pdb: Path to the output PDB file. If `None`, will not save a PDB file.
        output_rst: Path to the output restart file.
            If `None`, will not save a restart file unless `output_prmtop` is specified.
        output_prmtop: Path to the output prmtop file.
            If `None`, will not save a prmtop file unless `output_rst` is specified.
        sources: Source files to load in tleap like force fields

    Returns:
        A tleap script as a string
    """

    if isinstance(sources, str) or not isinstance(sources, Iterable):
        sources = [sources]

    template = ""
    for s in sources:
        template += f"source {s}\n"
    template += f"mol = loadpdb {input_pdb}\n"

    if output_pdb is not None:
        template += f"savepdb mol {output_pdb}\n"

    if (output_rst is not None) or (output_prmtop is not None):
        if output_prmtop is None:
            output_prmtop = pathlib.Path(output_rst).with_suffix(".prmtop")
        elif output_rst is None:
            output_rst = pathlib.Path(output_prmtop).with_suffix(".rst7")
        template += f"saveamberparm mol {output_prmtop} {output_rst}\n"

    template += "quit\n"

    return template


TEMPLATE_MAP = {
    "setup_protein": get_template_setup_protein_pdb,
}


def run_tleap_script(
    script: str | pathlib.Path,
    working_directory: str | pathlib.Path | None = None,
    raise_on_error: bool = True,
    executable: str = "tleap"
    ) -> subprocess.CompletedProcess:
    """Run a tleap script using subprocess"""

    process = subprocess.run(
        shlex.split(f"{executable} -f {script}"),
        capture_output=True,
        encoding="utf-8",
        cwd=working_directory
    )

    if process.returncode != 0 and raise_on_error:
        raise RuntimeError(f"tleap failed with error:\n{process.stderr}")

    return process


def run_tleap_template(
    template: str | Callable[[...], str],
    working_directory: str | pathlib.Path | None = None,
    overwrite: bool = True,
    script_name: str = "tleap.in",
    **kwargs
) -> subprocess.CompletedProcess:
    """Run a tleap script from a template string

    Args:
        template: Name of the template or a custom tleap script as a string.
            Can also be a callable with optional keyword arguments that returns a tleap script as a string.

    Keyword args:
        working_directory: Working directory to run tleap in.
            If `None`, will use the current working directory.
        overwrite: Whether to overwrite existing tleap script file in the working directory.
        script_name: Name of the tleap script file to write in the working directory.
        kwargs: Additional keyword arguments to pass to the template function
    """

    try:
        template = TEMPLATE_MAP[template]
    except KeyError:
        pass

    if callable(template):
        template = template(**kwargs)

    if working_directory is None:
        working_directory = pathlib.Path.cwd()
    
    working_directory = pathlib.Path(working_directory).resolve()
    if not working_directory.exists():
        working_directory.mkdir(parents=True, exist_ok=True)

    tleap_script_path = working_directory / script_name
    if tleap_script_path.exists() and not overwrite:
        raise FileExistsError(f"tleap script {tleap_script_path} already exists (`overwrite=False`)")

    with open(tleap_script_path, "w") as fp:
        fp.write(template)

    return run_tleap_script(tleap_script_path, working_directory=working_directory)
