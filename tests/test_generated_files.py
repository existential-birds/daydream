from daydream.generated_files import is_generated_file


def test_non_generated_source_allowed() -> None:
    assert not is_generated_file("src/app.py")
