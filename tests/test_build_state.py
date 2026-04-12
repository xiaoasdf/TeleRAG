from src.standards.build_state import init_build_state, load_build_state, save_build_state


def test_build_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    state = init_build_state(total_files=12, batch_size=4, vector_store_dir="data/vector_store/default", resume_enabled=True)
    state["status"] = "running"
    state["processed_files"] = 4

    save_build_state(path, state)
    loaded = load_build_state(path)

    assert loaded is not None
    assert loaded["status"] == "running"
    assert loaded["processed_files"] == 4
