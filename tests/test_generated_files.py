from daydream.generated_files import is_generated_file


def test_generated_sql_migration():
    assert is_generated_file("migrations/0001_init.sql")
    assert is_generated_file("backend/migrations/20240101_add_x.sql")
    assert is_generated_file("alembic/versions/abc123_add_col.py")


def test_non_generated_source_allowed():
    assert not is_generated_file("src/app.py")
    assert not is_generated_file("daydream/phases.py")


def test_codegen_and_lockfiles():
    assert is_generated_file("internal/model_generated.go")
    assert is_generated_file("Cargo.lock")
    assert is_generated_file("package-lock.json")
    assert is_generated_file("uv.lock")
    assert not is_generated_file("Cargo.toml")
    assert not is_generated_file("requirements.txt")
