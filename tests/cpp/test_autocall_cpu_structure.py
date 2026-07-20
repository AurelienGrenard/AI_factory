from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_autocall_cpu_batches_keep_openmp_and_compact_observations() -> None:
    for model in ("black_scholes", "heston", "rough_bergomi", "rough_heston"):
        source = (
            PROJECT_ROOT
            / "src_cpp"
            / "ai_factory"
            / "cpu"
            / model
            / "autocalls.cpp"
        ).read_text()

        if model in {"rough_bergomi", "rough_heston"}:
            simulation_source = (
                PROJECT_ROOT
                / "src_cpp"
                / "ai_factory"
                / "cpu"
                / model
                / "common.cpp"
            ).read_text()
            pragma = (
                "#pragma omp for schedule(static)"
                if model == "rough_bergomi"
                else "#pragma omp parallel for schedule(static)"
            )
            assert pragma in simulation_source
        else:
            assert "#pragma omp parallel for schedule(static)" in source
        assert f"{model}_observation_spots" in source
        assert f"generate_{model}_spot_paths" not in source
