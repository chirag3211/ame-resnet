#!/usr/bin/env python3
"""Unit tests for tools/record.py.
Run from the repo root: python3 -m pytest tests/test_record.py  (or python3 tests/test_record.py)

Covers:
  - parse(): well-formed AME log, scalar log, FAIL log, missing STATS
  - everything_else derived field
  - record.py main() via subprocess:
      - happy path (AME log + --backend ame): appends one correct JSON row
      - --backfill: git_commit / git_dirty / ts written as null, note contains 'backfilled'
      - backend mismatch (log says AME, --backend scalar): exits non-zero, no row written
      - FAIL log: exits 1, but still writes a row (PASS=false)
"""
import json, os, subprocess, sys, tempfile, textwrap, unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD_PY = os.path.join(REPO, "tools", "record.py")
sys.path.insert(0, os.path.join(REPO, "tools"))
from record import parse  # noqa: E402  (import after sys.path tweak)

# ---------------------------------------------------------------------------
# Fixture log strings
# ---------------------------------------------------------------------------
AME_LOG = textwrap.dedent("""\
    RUN0 INSTRET forward=200000 in_matmul=160000 in_copy=10000 copy_calls=2 ksteps=200 otiles=40
    RUN1 INSTRET forward=180000 in_matmul=150000 in_copy=8000 copy_calls=2 ksteps=190 otiles=38
    STATS backend=AME matmul_calls=7 macs=200000
    TOP1 381
    max abs err = 3.8e-06  top1 spike=381 pytorch=381
    PASS
""")

SCALAR_LOG = textwrap.dedent("""\
    RUN0 INSTRET forward=9000000 in_matmul=8500000 in_copy=10000 copy_calls=2 ksteps=0 otiles=0
    RUN1 INSTRET forward=8800000 in_matmul=8300000 in_copy=8000 copy_calls=2 ksteps=0 otiles=0
    STATS backend=reference-scalar matmul_calls=7 macs=200000
    TOP1 381
    max abs err = 3.8e-06  top1 spike=381 pytorch=381
    PASS
""")

FAIL_LOG = textwrap.dedent("""\
    RUN0 INSTRET forward=180000 in_matmul=150000 in_copy=8000 copy_calls=2 ksteps=0 otiles=0
    STATS backend=AME matmul_calls=7 macs=200000
    TOP1 999
    max abs err = 0.5   top1 spike=999 pytorch=381
    FAIL
""")

# Log where STATS backend=AME but we claim --backend scalar (mismatch)
AME_LOG_NO_RUN1 = textwrap.dedent("""\
    RUN0 INSTRET forward=200000 in_matmul=160000 in_copy=10000 copy_calls=2 ksteps=0 otiles=0
    STATS backend=AME matmul_calls=7 macs=200000
    TOP1 381
    max abs err = 3.8e-06  top1 spike=381 pytorch=381
    PASS
""")


# ---------------------------------------------------------------------------
# parse() unit tests
# ---------------------------------------------------------------------------
class TestParse(unittest.TestCase):
    def test_ame_runs(self):
        r = parse(AME_LOG)
        self.assertIn("0", r["runs"])
        self.assertIn("1", r["runs"])

    def test_run1_fields(self):
        r = parse(AME_LOG)["runs"]["1"]
        self.assertEqual(r["forward"], 180000)
        self.assertEqual(r["in_matmul"], 150000)
        self.assertEqual(r["in_copy"], 8000)
        self.assertEqual(r["copy_calls"], 2)
        self.assertEqual(r["ksteps"], 190)
        self.assertEqual(r["otiles"], 38)

    def test_everything_else_derived(self):
        for run in parse(AME_LOG)["runs"].values():
            self.assertEqual(
                run["everything_else"],
                run["forward"] - run["in_matmul"] - run["in_copy"],
            )

    def test_stats_ame(self):
        r = parse(AME_LOG)
        self.assertEqual(r["stats"]["backend"], "AME")
        self.assertEqual(r["stats"]["matmul_calls"], 7)
        self.assertEqual(r["stats"]["macs"], 200000)

    def test_stats_scalar(self):
        r = parse(SCALAR_LOG)
        self.assertEqual(r["stats"]["backend"], "reference-scalar")

    def test_top1_and_err(self):
        r = parse(AME_LOG)
        self.assertEqual(r["top1"], 381)
        self.assertAlmostEqual(r["max_abs_err"], 3.8e-6)

    def test_pass(self):
        self.assertTrue(parse(AME_LOG)["pass"])
        self.assertFalse(parse(FAIL_LOG)["pass"])

    def test_verdict(self):
        self.assertEqual(parse(AME_LOG)["verdict"], "PASS")
        self.assertEqual(parse(FAIL_LOG)["verdict"], "FAIL")

    def test_missing_stats_no_crash(self):
        r = parse("RUN1 INSTRET forward=1 in_matmul=0 in_copy=0 copy_calls=0 ksteps=0 otiles=0\nPASS\n")
        self.assertNotIn("stats", r)
        self.assertTrue(r["pass"])

    def test_only_run0_parsed(self):
        r = parse(AME_LOG_NO_RUN1)
        self.assertIn("0", r["runs"])
        self.assertNotIn("1", r["runs"])

    def test_errors_collected(self):
        log = AME_LOG + "Error: something bad happened\n"
        r = parse(log)
        self.assertTrue(any("Error" in e for e in r["errors"]))


# ---------------------------------------------------------------------------
# Subprocess integration tests for record.py main()
# ---------------------------------------------------------------------------
def _run_record(log_text, backend, extra=None, out_path=None, cwd=None):
    """Write log_text to a temp file, invoke record.py, return (returncode, out_path)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(log_text)
        log_path = f.name
    if out_path is None:
        out_fd, out_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(out_fd)
        os.unlink(out_path)  # record.py appends; start from absent
    cmd = [
        sys.executable, RECORD_PY,
        log_path,
        "--dir", "/tmp/fake_dir",
        "--backend", backend,
        "--out", out_path,
        "--set", "llvm_opt=O3",
    ]
    if extra:
        cmd.extend(extra)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or REPO)
    os.unlink(log_path)
    return result, out_path


class TestRecordMain(unittest.TestCase):
    def test_happy_path_ame(self):
        """Happy path: AME log + --backend ame → PASS row appended."""
        res, out = _run_record(AME_LOG, "ame")
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        with open(out) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["backend"], "ame")
        self.assertTrue(row["pass"])
        r1 = row["runs"]["1"]
        self.assertEqual(r1["forward"], 180000)
        self.assertEqual(r1["everything_else"], 180000 - 150000 - 8000)
        self.assertEqual(row["flags"]["llvm_opt"], "O3")
        os.unlink(out)

    def test_happy_path_scalar(self):
        """Happy path: scalar log + --backend scalar → PASS row appended."""
        res, out = _run_record(SCALAR_LOG, "scalar")
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        with open(out) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(rows[0]["backend"], "scalar")
        self.assertTrue(rows[0]["pass"])
        os.unlink(out)

    def test_backfill_nulls(self):
        """--backfill: git_commit, git_dirty, ts must be null; note contains 'backfilled'."""
        res, out = _run_record(AME_LOG, "ame", extra=["--backfill"])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        with open(out) as f:
            row = json.loads(f.read().strip())
        self.assertIsNone(row["git_commit"], "git_commit should be null with --backfill")
        self.assertIsNone(row["git_dirty"],  "git_dirty should be null with --backfill")
        self.assertIsNone(row["ts"],          "ts should be null with --backfill")
        self.assertIsNotNone(row["ingested_ts"], "ingested_ts should be set")
        self.assertIn("backfilled", row.get("note", ""))
        os.unlink(out)

    def test_backfill_known_commit(self):
        """--backfill --git-commit SHA: git_commit is the given SHA, not HEAD."""
        res, out = _run_record(AME_LOG, "ame", extra=["--backfill", "--git-commit", "abc1234"])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        with open(out) as f:
            row = json.loads(f.read().strip())
        self.assertEqual(row["git_commit"], "abc1234")
        self.assertIsNone(row["ts"])
        os.unlink(out)

    def test_backend_mismatch_ame_log_scalar_flag(self):
        """Log says STATS backend=AME but --backend scalar → non-zero exit, no row."""
        _, out = _run_record(AME_LOG, "scalar")  # should fail
        # We expect a non-zero return; the file should not exist or be empty
        res, out2 = _run_record(AME_LOG, "scalar", out_path=out)
        self.assertNotEqual(res.returncode, 0, "should refuse backend mismatch")
        # No row should have been appended
        if os.path.exists(out):
            rows = [l for l in open(out) if l.strip()]
            self.assertEqual(len(rows), 0)
        # Cleanup
        for p in (out, out2):
            if os.path.exists(p): os.unlink(p)

    def test_backend_mismatch_scalar_log_ame_flag(self):
        """Log says STATS backend=reference-scalar but --backend ame → non-zero exit."""
        res, out = _run_record(SCALAR_LOG, "ame")
        self.assertNotEqual(res.returncode, 0, "should refuse backend mismatch")
        if os.path.exists(out): os.unlink(out)

    def test_fail_log_exits_nonzero(self):
        """FAIL verdict → exit code 1, but the row IS written (pass=false)."""
        res, out = _run_record(FAIL_LOG, "ame")
        self.assertEqual(res.returncode, 1)
        with open(out) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["pass"])
        os.unlink(out)

    def test_multiple_appends(self):
        """Running record.py twice appends two rows to the same file."""
        _, out = _run_record(AME_LOG, "ame")
        _run_record(SCALAR_LOG, "scalar", out_path=out)
        with open(out) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 2)
        backends = {r["backend"] for r in rows}
        self.assertEqual(backends, {"ame", "scalar"})
        os.unlink(out)

    def test_wall_recorded(self):
        """--wall N is stored in the row."""
        res, out = _run_record(AME_LOG, "ame", extra=["--wall", "42.5"])
        self.assertEqual(res.returncode, 0)
        with open(out) as f:
            row = json.loads(f.read().strip())
        self.assertAlmostEqual(row["wall_s"], 42.5)
        os.unlink(out)


if __name__ == "__main__":
    unittest.main()
