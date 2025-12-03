import argparse
import time
from pathlib import Path

from loguru import logger
from openmm import LangevinMiddleIntegrator, Platform
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, StateDataReporter, Simulation
from openmm.unit import kelvin, picosecond
from parmed.openmm import RestartReporter


def minimize_structure(prmtop_file: str, inpcrd_file: str, output_file: str | None = None, tolerance: float = 10.0, max_iterations: int = 0, overwrite: bool = False):
    """
    Minimize a molecular structure using OpenMM.

    Args:
        prmtop_file: Path to Amber topology file (.prmtop)
        inpcrd_file: Path to Amber coordinate file (.rst7 or .inpcrd)
        output_file: Path to output minimized coordinates (.rst7)
        tolerance: Energy tolerance for minimization (kJ/mol)
        max_iterations: Maximum number of minimization steps (0 = unlimited)
    """
    
    if output_file is None:
        output_file = pathlib.Path(inpcrd_file).with_name(f'{pathlib.Path(inpcrd_file).stem}_emin.rst7').as_posix()
    
    if Path(output_file).exists() and not overwrite:
        logger.error(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        return

    prmtop = AmberPrmtopFile(prmtop_file)
    inpcrd = AmberInpcrdFile(inpcrd_file)
    system = prmtop.createSystem()
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picosecond)

    platform = Platform.getPlatformByName('CPU')
    simulation = Simulation(prmtop.topology, system, integrator, platform=platform)
    simulation.context.setPositions(inpcrd.positions)

    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    state = simulation.context.getState(getEnergy=True)
    logger.info(f"Initial potential energy: {state.getPotentialEnergy()}")

    logger.info(f"Minimizing (tolerance={tolerance} kJ/mol, max_iterations={max_iterations})...")
    start_time = time.time()
    simulation.minimizeEnergy(tolerance=tolerance, maxIterations=max_iterations)
    end_time = time.time()
    logger.info(f"Minimization took {end_time - start_time:.2f} seconds.")

    state = simulation.context.getState(getEnergy=True, getPositions=True)
    logger.info(f"Final potential energy: {state.getPotentialEnergy()}")

    logger.info(f"Writing minimized structure to: {output_file}")
    positions = state.getPositions()

    reporter = RestartReporter(output_file, 1, write_velocities=False)
    reporter.report(simulation, state)

    logger.success("Minimization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Minimize a molecular structure using OpenMM with Amber files"
    )
    parser.add_argument(
        "--prmtop",
        required=True,
        help="Input Amber topology file (.prmtop)"
    )
    parser.add_argument(
        "--inpcrd",
        required=True,
        help="Input Amber coordinate file (.rst7 or .inpcrd)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output minimized coordinate file (.rst7)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Energy tolerance for minimization in kJ/mol (default: 10.0)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Maximum number of minimization steps, 0 for unlimited (default: 0)"
    )
    
    args = parser.parse_args()
    
    minimize_structure(
        args.prmtop,
        args.inpcrd,
        output_file=args.output,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()