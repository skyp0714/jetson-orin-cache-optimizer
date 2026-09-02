import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from cache_optimizer.nscache import (
    CachePPA,
    JsonPPACache,
    NSCacheConfigError,
    NSCacheParseError,
    NSCacheRunError,
    build_ppa_cache_key,
    derive_force_bank_count,
    memory_cell_input,
    parse_config_text,
    parse_summary,
    patch_cell_text,
    patch_config_text,
    patch_memory_cell_input,
    read_config,
    render_config,
    run_nscache,
    run_nscache_output,
)


SRAM_SUMMARY = textwrap.dedent(
    """
    NS-Cache banner
    =======================
    CACHE DESIGN -- SUMMARY
    =======================
    Access Mode: Normal
    Area:
     - Total Area = 0.712mm^2
    Timing:
     - Cache Hit Latency = 0.419ns
     - Cache Miss Latency = 0.307ns
     - Cache Write Latency = 0.532ns
    Power:
     - Cache Hit Dynamic Energy = 0.014nJ per access
     - Cache Miss Dynamic Energy = 0.009nJ per access
     - Cache Write Dynamic Energy = 0.019nJ per access
     - Cache Total Leakage Power = 0.234mW
    Finished!
    """
)


class SummaryParserTests(unittest.TestCase):
    def test_parses_required_sram_metrics(self):
        ppa = parse_summary(SRAM_SUMMARY)

        self.assertEqual(ppa.area_mm2, 0.712)
        self.assertEqual(ppa.hit_latency_ns, 0.419)
        self.assertEqual(ppa.miss_energy_nj, 0.009)
        self.assertEqual(ppa.leakage_power_mw, 0.234)
        self.assertIsNone(ppa.refresh_power_mw)
        self.assertIsNone(ppa.availability_percent)

    def test_converts_units_and_parses_refresh_metrics(self):
        output = textwrap.dedent(
            """
            =======================
            CACHE DESIGN -- SUMMARY
            =======================
             - Total Area = 712000um^2
             - Cache Hit Latency = 419ps
             - Cache Miss Latency = 0.0012us
             - Cache Write Latency = 750ps
             - Cache Refresh Latency = 2ns per bank
             - Cache Availability = 99.5%
             - Cache Hit Dynamic Energy = 15pJ per access
             - Cache Miss Dynamic Energy = 0.00002uJ per access
             - Cache Write Dynamic Energy = 3e-11J per access
             - Cache Refresh Dynamic Energy = 400pJ per bank
             - Cache Total Leakage Power = 234uW
             - Cache Refresh Power = 274.87nW per bank
            """
        )

        ppa = parse_summary(output)

        self.assertAlmostEqual(ppa.area_mm2, 0.712)
        self.assertAlmostEqual(ppa.hit_latency_ns, 0.419)
        self.assertAlmostEqual(ppa.miss_latency_ns, 1.2)
        self.assertAlmostEqual(ppa.write_latency_ns, 0.75)
        self.assertAlmostEqual(ppa.hit_energy_nj, 0.015)
        self.assertAlmostEqual(ppa.miss_energy_nj, 0.02)
        self.assertAlmostEqual(ppa.write_energy_nj, 0.03)
        self.assertAlmostEqual(ppa.leakage_power_mw, 0.234)
        self.assertAlmostEqual(ppa.refresh_latency_us, 0.002)
        self.assertAlmostEqual(ppa.refresh_energy_nj, 0.4)
        self.assertAlmostEqual(ppa.refresh_power_mw, 0.00027487)
        self.assertAlmostEqual(ppa.availability_percent, 99.5)
        self.assertEqual(ppa.refresh_energy_nj_per_bank, ppa.refresh_energy_nj)

    def test_uses_last_summary_and_rejects_missing_required_metric(self):
        invalid = SRAM_SUMMARY.replace(" - Cache Write Dynamic Energy = 0.019nJ per access\n", "")
        with self.assertRaisesRegex(NSCacheParseError, "write dynamic energy"):
            parse_summary(SRAM_SUMMARY + "\n" + invalid)


class ConfigTests(unittest.TestCase):
    TEMPLATE = textwrap.dedent(
        """
        -DesignTarget: cache
        -Associativity (for cache only): 16 // keep
        -Capacity (MB): 4 # baseline
        //-ForceBankA (Total AxB): 64x8
        -ForceBankA (Total AxB): 2 x 4
        -MemoryCellInputFile: config_uiuc/cell.cell
        """
    )

    def test_reads_capacity_associativity_and_bank_count(self):
        config = parse_config_text(self.TEMPLATE)
        self.assertEqual(config.capacity_kib, 4096)
        self.assertEqual(config.associativity, 16)
        self.assertEqual(config.force_bank_count, 8)
        self.assertEqual(derive_force_bank_count(self.TEMPLATE), 8)

    def test_patch_returns_variant_without_mutating_template(self):
        original = self.TEMPLATE
        variant = patch_config_text(original, capacity_kib=768, associativity=8)

        self.assertEqual(original, self.TEMPLATE)
        self.assertIn("-Capacity (KB): 768 # baseline", variant)
        self.assertIn("-Associativity (for cache only): 8 // keep", variant)
        self.assertEqual(parse_config_text(variant).force_bank_count, 8)

    def test_render_does_not_modify_template_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.cfg"
            path.write_text(self.TEMPLATE, encoding="utf-8")

            rendered = render_config(path, capacity_kib=512, associativity=4)

            self.assertEqual(path.read_text(encoding="utf-8"), self.TEMPLATE)
            self.assertEqual(read_config(path).capacity_kib, 4096)
            self.assertEqual(parse_config_text(rendered).capacity_kib, 512)

    def test_malformed_config_is_rejected(self):
        with self.assertRaisesRegex(NSCacheConfigError, "Associativity"):
            parse_config_text("-Capacity (KB): 256\n")

    def test_generated_cell_override_does_not_mutate_template(self):
        cell = "-MemCellType: gcDRAM\n"
        generated = patch_cell_text(cell, retention_time_us=315000, temperature_k=300)
        self.assertEqual(cell, "-MemCellType: gcDRAM\n")
        self.assertIn("-RetentionTime (us): 315000", generated)
        self.assertIn("-Temperature (K): 300", generated)
        config = patch_memory_cell_input(self.TEMPLATE, "/tmp/generated.cell")
        self.assertIn('-MemoryCellInputFile: "/tmp/generated.cell"', config)
        self.assertEqual(memory_cell_input(config), "/tmp/generated.cell")


class RunnerTests(unittest.TestCase):
    def _script(self, directory, body):
        path = Path(directory) / "fake_nsc.py"
        path.write_text(body, encoding="utf-8")
        return path

    def test_runner_parses_successful_process(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self._script(
                directory,
                "import sys\nprint(%r)\n" % SRAM_SUMMARY,
            )

            ppa = run_nscache(sys.executable, script, cwd=directory, timeout_s=2)

            self.assertEqual(ppa.area_mm2, 0.712)

    def test_runner_reports_exit_status_and_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self._script(
                directory,
                "import sys\nprint('bad config', file=sys.stderr)\nsys.exit(3)\n",
            )

            with self.assertRaises(NSCacheRunError) as context:
                run_nscache_output(sys.executable, script, cwd=directory, timeout_s=2)

            message = str(context.exception)
            self.assertIn("status 3", message)
            self.assertIn("bad config", message)
            self.assertIn("Command:", message)

    def test_runner_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self._script(directory, "import time\ntime.sleep(1)\n")
            with self.assertRaisesRegex(NSCacheRunError, "timed out"):
                run_nscache_output(sys.executable, script, cwd=directory, timeout_s=0.05)

    def test_run_rejects_exit_zero_without_completed_design(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self._script(
                directory,
                "print('numSolutions = 0 / numDesigns = 12')\nprint('No valid solutions.')\n",
            )
            with self.assertRaisesRegex(NSCacheRunError, "invalid design"):
                run_nscache(sys.executable, script, cwd=directory, timeout_s=2)


class PPACacheTests(unittest.TestCase):
    def test_key_includes_binary_config_and_memory_cell_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "cell.cell"
            cell.write_text("-MemCellType: SRAM\n", encoding="utf-8")
            config = root / "cache.cfg"
            config.write_text(
                "-Capacity (KB): 256\n"
                "-Associativity (for cache only): 4\n"
                "-MemoryCellInputFile: cell.cell\n",
                encoding="utf-8",
            )

            first = build_ppa_cache_key(sys.executable, config, cwd=root)
            cell.write_text("-MemCellType: MRAM\n", encoding="utf-8")
            second = build_ppa_cache_key(sys.executable, config, cwd=root)

            self.assertRegex(first, r"^[0-9a-f]{64}$")
            self.assertNotEqual(first, second)

    def test_json_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppa.json"
            cache = JsonPPACache(path)
            key = "a" * 64
            ppa = CachePPA(
                area_mm2=1,
                hit_latency_ns=2,
                miss_latency_ns=3,
                write_latency_ns=4,
                hit_energy_nj=5,
                miss_energy_nj=6,
                write_energy_nj=7,
                leakage_power_mw=8,
            )

            self.assertIsNone(cache.get(key))
            cache.put(key, ppa)

            self.assertEqual(cache.get(key), ppa)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertIn(key, payload["entries"])


if __name__ == "__main__":
    unittest.main()
