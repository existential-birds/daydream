from daydream.quote_scrub import normalize_smart_quotes, scrub_smart_quotes_changed_files


def test_normalize_smart_quotes_maps_all_four_code_points_to_ascii():
    assert normalize_smart_quotes("\u201Cleft\u201D \u2018single\u2019") == '"left" \'single\''
    assert normalize_smart_quotes("not \u201D") == 'not "'


def test_normalize_smart_quotes_identity_on_ascii():
    # Legitimate ASCII code/strings are byte-identical.
    assert normalize_smart_quotes('fmt.Println("x")') == 'fmt.Println("x")'
    # The empty-string literal in a Go comment is already ASCII — untouched.
    assert normalize_smart_quotes("// not ''") == "// not ''"


def test_scrub_driver_rewrites_and_reports_changed_files(tmp_path):
    (tmp_path / "main.go").write_text("package main\n\n// not \u201D\n")
    (tmp_path / "keep.py").write_text("x = 1\n")
    scrubbed = scrub_smart_quotes_changed_files(tmp_path, ["main.go", "keep.py"])
    assert scrubbed == ["main.go"]
    assert (tmp_path / "main.go").read_text() == 'package main\n\n// not "\n'
    assert (tmp_path / "keep.py").read_text() == "x = 1\n"


def test_scrub_driver_skips_generated_binary_and_out_of_scope(tmp_path):
    gen = tmp_path / "models_generated.go"
    gen.write_text("// generated \u201D\n")
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\x00\xff\xfe smart \x80")  # invalid UTF-8
    outside = tmp_path / "outside.txt"
    outside.write_text("\u201Doutside\u201D")
    assert scrub_smart_quotes_changed_files(tmp_path, ["models_generated.go", "blob.bin"]) == []
    assert gen.read_text() == "// generated \u201D\n"
    assert binary.read_bytes() == b"\x00\xff\xfe smart \x80"
    assert outside.read_text() == "\u201Doutside\u201D"  # not in changed set → untouched


def test_scrub_driver_skips_missing_file(tmp_path):
    assert scrub_smart_quotes_changed_files(tmp_path, ["gone.go"]) == []
