using ArgParse
using BiosimMDSuite

function minimise_structure(
    bsm_xml_file::String;
    working_directory::String,
    step_size::Float64=0.001,
    energy_tolerance::Float64=1000.0
    )
    mkpath(working_directory)

    mdoptions = MDOptions(
        integrator=SteepestDescentMinimizer,
        step_size=step_size,
        energy_tolerance=energy_tolerance
        )
    driver = driver_from_xml(bsm_xml_file, mdoptions, working_directory)
    md_run!(driver, 1000)
    write_xml(driver, "final.xml")

    system_em = create_system_from_xml(working_directory * "/final.xml")
    write_mmcif(working_directory * "/final.cif", system_em.S)
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--bsm-xml-file"
            help = "Path to the input files containing directory"
            arg_type = String
            required = true
        "--working-directory"
            help = "Path to the output working directory"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    minimise_structure(args["bsm-xml-file"]; working_directory=args["working-directory"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end