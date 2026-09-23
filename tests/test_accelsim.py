import tempfile
import textwrap
import unittest
from pathlib import Path

from cache_optimizer.accelsim import (
    AccelSimConfigError,
    AccelSimRunError,
    build_accelsim_environment,
    generate_candidate_config,
    map_relative_delta_latency,
    parse_cache_string,
    parse_config_text,
    parse_output,
    run_accelsim,
)


ORIN_CONFIG = textwrap.dedent(
    """
    # inactive duplicate must be ignored
    #-gpgpu_clock_domains 1:1:1:1
    -gpgpu_n_clusters 16
    -gpgpu_n_cores_per_cluster 1
    -gpgpu_n_mem 16
    -gpgpu_n_sub_partition_per_mchannel 1
    -gpgpu_clock_domains 1300:1300:1300:1600
    -gpgpu_adaptive_cache_config 1
    -gpgpu_shmem_option 0,8,16,32,64,100,132,164
    -gpgpu_unified_l1d_size 192
    -gpgpu_shmem_size 167936
    -gpgpu_shmem_sizeDefault 167936
    -gpgpu_cache:dl1 S:4:128:64,L:T:m:L:L,A:384:48,16:0,32 # keep L1
    -gpgpu_l1_latency 38
    -gpgpu_cache:dl2 S:128:128:16,L:B:m:L:P,A:192:32,32:0,32 # keep L2
    -gpgpu_l2_rop_latency 146
    -enable_ptx_file_line_stats 1
    """
)


def cache_block(level, entries):
    prefix = "Total_core_cache_stats_breakdown" if level == "l1" else "L2_cache_stats_breakdown"
    return "\n".join(
        f"{prefix}[{access}][{status}] = {value}"
        for access, status, value in entries
    )


SUCCESS_LOG = textwrap.dedent(
    """
    kernel_stream_id = 0
    gpu_tot_sim_cycle = 100
    gpu_tot_sim_insn = 50
    gpu_tot_ipc = 0.5
    """
) + cache_block(
    "l1",
    [
        ("GLOBAL_ACC_R", "HIT", 1),
        ("GLOBAL_ACC_R", "MISS", 1),
        ("GLOBAL_ACC_R", "MSHR_HIT", 0),
        ("GLOBAL_ACC_W", "MISS", 1),
    ],
) + "\n" + cache_block(
    "l2",
    [
        ("GLOBAL_ACC_R", "HIT", 1),
        ("GLOBAL_ACC_R", "MISS", 1),
        ("GLOBAL_ACC_W", "MISS", 1),
    ],
) + "\nGPGPU-Sim: *** simulation thread exiting ***\nGPGPU-Sim: *** exit detected ***\n"


class OutputParserTests(unittest.TestCase):
    def test_uses_final_snapshot_per_stream_and_sums_streams(self):
        first_stream_zero = "\n".join(
            [
                "kernel_stream_id = 0",
                "gpu_tot_sim_cycle = 10",
                "gpu_tot_sim_insn = 5",
                "gpu_tot_ipc = 0.5",
                cache_block(
                    "l1",
                    [
                        ("GLOBAL_ACC_R", "HIT", 1),
                        ("GLOBAL_ACC_R", "MISS", 1),
                        ("GLOBAL_ACC_R", "MSHR_HIT", 0),
                        ("GLOBAL_ACC_W", "MISS", 1),
                    ],
                ),
                cache_block(
                    "l2",
                    [
                        ("GLOBAL_ACC_R", "HIT", 1),
                        ("GLOBAL_ACC_R", "MISS", 1),
                        ("GLOBAL_ACC_W", "MISS", 1),
                    ],
                ),
            ]
        )
        stream_one = "\n".join(
            [
                "kernel_stream_id = 1",
                cache_block(
                    "l1",
                    [
                        ("GLOBAL_ACC_R", "HIT", 1),
                        ("GLOBAL_ACC_R", "MISS", 2),
                        ("GLOBAL_ACC_R", "MSHR_HIT", 0),
                        ("LOCAL_ACC_W", "HIT", 2),
                    ],
                ),
                cache_block(
                    "l2",
                    [
                        ("LOCAL_ACC_R", "HIT", 2),
                        ("LOCAL_ACC_R", "MISS", 4),
                        ("LOCAL_ACC_W", "SECTOR_MISS", 1),
                    ],
                ),
            ]
        )
        final_stream_zero = "\n".join(
            [
                "kernel_stream_id = 0",
                "gpu_tot_sim_cycle = 300",
                "gpu_tot_sim_insn = 180",
                "gpu_tot_ipc = 0.6",
                cache_block(
                    "l1",
                    [
                        ("GLOBAL_ACC_R", "HIT", 2),
                        ("GLOBAL_ACC_R", "MISS", 3),
                        ("GLOBAL_ACC_R", "SECTOR_MISS", 1),
                        ("GLOBAL_ACC_R", "MSHR_HIT", 1),
                        ("GLOBAL_ACC_W", "MISS", 4),
                    ],
                ),
                cache_block(
                    "l2",
                    [
                        ("GLOBAL_ACC_R", "HIT", 5),
                        ("GLOBAL_ACC_R", "HIT_RESERVED", 1),
                        ("GLOBAL_ACC_R", "MISS", 2),
                        ("GLOBAL_ACC_R", "SECTOR_MISS", 1),
                        ("GLOBAL_ACC_W", "HIT_RESERVED", 1),
                        ("GLOBAL_ACC_W", "MISS", 2),
                        ("L1_WRBK_ACC", "SECTOR_MISS", 3),
                    ],
                ),
                "GPGPU-Sim: *** simulation thread exiting ***",
                "GPGPU-Sim: *** exit detected ***",
            ]
        )

        parsed = parse_output("\n".join((first_stream_zero, stream_one, final_stream_zero)))

        self.assertTrue(parsed.success)
        self.assertEqual((parsed.cycles, parsed.instructions, parsed.ipc), (300, 180, 0.6))
        self.assertEqual(parsed.accesses.l1_read_hit, 4)
        self.assertEqual(parsed.accesses.l1_read_miss, 5)
        self.assertEqual(parsed.accesses.l1_write, 6)
        self.assertEqual(parsed.accesses.l2_read_hit, 8)
        self.assertEqual(parsed.accesses.l2_read_miss, 7)
        self.assertEqual(parsed.accesses.l2_write, 7)
        self.assertAlmostEqual(parsed.runtime_s(1300), 300 / 1.3e9)

    def test_limit_and_error_markers_are_not_success(self):
        limited = parse_output(
            SUCCESS_LOG.replace(
                "GPGPU-Sim: *** exit detected ***",
                "GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **",
            )
        )
        self.assertTrue(limited.reached_limit)
        self.assertFalse(limited.success)

        errored = parse_output(
            SUCCESS_LOG.replace(
                "GPGPU-Sim: *** exit detected ***",
                "GPGPU-Sim uArch: ERROR ** deadlock detected\n"
                "GPGPU-Sim: *** exit detected ***",
            )
        )
        self.assertFalse(errored.success)
        self.assertTrue(any("deadlock" in line for line in errored.error_messages))

        trace_error = parse_output(
            SUCCESS_LOG.replace(
                "GPGPU-Sim: *** exit detected ***",
                "ERROR:  undefined instruction : OP\nGPGPU-Sim: *** exit detected ***",
            )
        )
        self.assertFalse(trace_error.success)
        self.assertTrue(any("undefined instruction" in line for line in trace_error.error_messages))

    def test_missing_final_metrics_are_reported(self):
        parsed = parse_output("GPGPU-Sim: *** exit detected ***\n")
        self.assertFalse(parsed.success)
        self.assertIn("gpu_tot_sim_cycle", parsed.missing_metrics)
        with self.assertRaisesRegex(Exception, "missing metrics"):
            parsed.require_success()

class ConfigTests(unittest.TestCase):
    def test_parses_cache_strings_clocks_and_instance_counts(self):
        config = parse_config_text(ORIN_CONFIG)

        self.assertEqual(config.l1.sets, 4)
        self.assertEqual(config.l1.associativity, 64)
        self.assertEqual(config.l2.capacity_kib, 256)
        self.assertEqual(config.l2_total_capacity_kib, 4096)
        self.assertEqual(config.l1_instances, 16)
        self.assertEqual(config.clocks.core_mhz, 1300)
        self.assertEqual(config.clocks.l2_mhz, 1300)
        self.assertEqual(config.clocks.render(), "1300:1300:1300:1600")
        self.assertEqual(config.l1_latency_cycles, 38)
        self.assertEqual(config.l2_latency_cycles, 146)

    def test_candidate_mapping_is_relative_and_preserves_policies(self):
        rendered = generate_candidate_config(
            ORIN_CONFIG,
            baseline_l1_capacity_kib=256,
            baseline_l1_associativity=4,
            baseline_l2_capacity_kib=4096,
            baseline_l2_associativity=16,
            l1_capacity_kib=512,
            l1_associativity=8,
            l2_capacity_kib=8192,
            l2_associativity=16,
            baseline_l1_latency_ns=0.5,
            l1_latency_ns=1.4,
            baseline_l2_latency_ns=3.0,
            l2_latency_ns=4.0,
            extra_directives={"-enable_ptx_file_line_stats": 0},
        )
        candidate = parse_config_text(rendered)

        self.assertIn("S:4:128:128,L:T:m:L:L,A:384:48,16:0,32 # keep L1", rendered)
        self.assertIn("S:256:128:16,L:B:m:L:P,A:192:32,32:0,32 # keep L2", rendered)
        self.assertEqual(candidate.unified_l1d_size_kib, 384)
        self.assertEqual(candidate.shmem_options_kib, (0, 8, 16, 32, 64, 100, 132, 164))
        self.assertEqual(candidate.shmem_size_bytes, 167936)
        self.assertEqual(candidate.shmem_size_default_bytes, 167936)
        self.assertEqual(candidate.l1_latency_cycles, 40)
        self.assertEqual(candidate.l2_latency_cycles, 148)
        self.assertIn("-enable_ptx_file_line_stats 0", rendered)
        self.assertIn("-gpgpu_unified_l1d_size 192", ORIN_CONFIG)

    def test_rejects_nonintegral_and_unsupported_geometry(self):
        common = dict(
            baseline_l1_capacity_kib=256,
            baseline_l1_associativity=4,
            baseline_l2_capacity_kib=4096,
            baseline_l2_associativity=16,
            l1_associativity=8,
            l2_associativity=16,
        )
        with self.assertRaisesRegex(AccelSimConfigError, "non-integral"):
            generate_candidate_config(
                ORIN_CONFIG,
                **common,
                l1_capacity_kib=320,
                l2_capacity_kib=4096,
            )
        with self.assertRaisesRegex(AccelSimConfigError, "IPOLY"):
            generate_candidate_config(
                ORIN_CONFIG,
                **common,
                l1_capacity_kib=512,
                l2_capacity_kib=16384,
            )
        with self.assertRaisesRegex(AccelSimConfigError, "shared-memory limit"):
            generate_candidate_config(
                ORIN_CONFIG,
                **{**common, "l1_associativity": 4},
                l1_capacity_kib=128,
                l2_capacity_kib=4096,
            )

    def test_latency_delta_rounds_away_from_zero_and_clamps(self):
        self.assertEqual(
            map_relative_delta_latency(
                baseline_cycles=38,
                baseline_latency_ns=0.5,
                candidate_latency_ns=1.0,
                clock_mhz=1300,
            ),
            39,
        )
        self.assertEqual(
            map_relative_delta_latency(
                baseline_cycles=38,
                baseline_latency_ns=1.0,
                candidate_latency_ns=0.5,
                clock_mhz=1300,
            ),
            37,
        )
        self.assertEqual(
            map_relative_delta_latency(
                baseline_cycles=1,
                baseline_latency_ns=100,
                candidate_latency_ns=0,
                clock_mhz=1300,
            ),
            1,
        )

    def test_duplicate_active_directive_is_rejected(self):
        with self.assertRaisesRegex(AccelSimConfigError, "multiple active"):
            parse_config_text(ORIN_CONFIG + "\n-gpgpu_clock_domains 2:2:2:2\n")

    def test_cache_string_round_trip(self):
        raw = "S:128:128:16,L:B:m:L:P,A:192:32,32:0,32"
        self.assertEqual(parse_cache_string(raw).render(), raw)


class RunnerTests(unittest.TestCase):
    def _input_files(self, root):
        trace = root / "kernelslist.g"
        gpu_config = root / "gpgpusim.config"
        trace_config = root / "trace.config"
        for path in (trace, gpu_config, trace_config):
            path.write_text("fixture\n", encoding="utf-8")
        return trace, gpu_config, trace_config

    def _executable(self, root, body):
        path = root / "fake_accelsim.py"
        path.write_text("#!/usr/bin/python3\n" + body, encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_runner_uses_explicit_environment_and_writes_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, gpu_config, trace_config = self._input_files(root)
            binary = self._executable(
                root,
                "import os\n"
                "print('TOKEN=' + os.environ.get('TOKEN', 'missing'))\n"
                f"print({SUCCESS_LOG!r})\n",
            )

            result = run_accelsim(
                binary,
                trace,
                gpu_config,
                trace_config,
                work_dir=root / "candidate-run",
                environment={"TOKEN": "isolated"},
                timeout_s=2,
            )

            self.assertTrue(result.success)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.output.cycles, 100)
            self.assertIn("TOKEN=isolated", result.stdout_log.read_text(encoding="utf-8"))
            self.assertEqual(result.work_dir, (root / "candidate-run").resolve())

    def test_runner_detects_timeout_and_existing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, gpu_config, trace_config = self._input_files(root)
            binary = self._executable(root, "import time\ntime.sleep(2)\n")
            work_dir = root / "timeout-run"

            result = run_accelsim(
                binary,
                trace,
                gpu_config,
                trace_config,
                work_dir=work_dir,
                environment={},
                timeout_s=0.05,
                check=False,
            )
            self.assertTrue(result.timed_out)
            self.assertFalse(result.success)
            with self.assertRaisesRegex(AccelSimRunError, "already exists"):
                run_accelsim(
                    binary,
                    trace,
                    gpu_config,
                    trace_config,
                    work_dir=work_dir,
                    environment={},
                    timeout_s=0.05,
                    check=False,
                )

    def test_build_environment_selects_custom_gpgpusim_lib(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpgpusim_root = root / "accel-sim" / "gpu-simulator" / "gpgpu-sim"
            release = gpgpusim_root / "lib" / "gcc-9" / "cuda-11.0" / "release"
            cuda_root = root / "cuda"
            release.mkdir(parents=True)
            (cuda_root / "bin").mkdir(parents=True)

            env = build_accelsim_environment(
                root / "accel-sim",
                cuda_install_path=cuda_root,
                base_environment={"PATH": "/usr/bin"},
            )

            self.assertEqual(env["GPGPUSIM_ROOT"], str(gpgpusim_root.resolve()))
            self.assertEqual(env["LD_LIBRARY_PATH"], str(release.resolve()))
            self.assertTrue(env["PATH"].startswith(str(cuda_root.resolve() / "bin")))


if __name__ == "__main__":
    unittest.main()
