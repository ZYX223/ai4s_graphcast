import unittest

from graphcast_bench.logical_inventory import group_equations


class LogicalInventoryTests(unittest.TestCase):
    def test_composites_own_their_internal_operations(self):
        rows = [
            dict(id=1, primitive="dot_general", scope="model/mlp/linear_0"),
            dict(id=2, primitive="add", scope="model/mlp/linear_0"),
            dict(id=3, primitive="reduce_sum", scope="model/node_layer_norm/jit(_var)"),
            dict(id=4, primitive="rsqrt", scope="model/node_layer_norm"),
            dict(id=5, primitive="exp", scope="model/mlp/jit(silu)"),
            dict(id=6, primitive="mul", scope="model/mlp/jit(silu)"),
            dict(id=7, primitive="add", scope="model/residual"),
        ]
        groups = group_equations(rows)
        self.assertEqual([g["operator"] for g in groups], ["linear", "layer_norm", "silu", "add"])
        self.assertEqual([g["equation_ids"] for g in groups], [[1, 2], [3, 4], [5, 6], [7]])

    def test_shared_primitive_name_does_not_merge_distinct_invocations(self):
        rows = [dict(id=i, primitive="add", scope="model/residual") for i in (1, 2)]
        self.assertEqual(len(group_equations(rows)), 2)

    def test_rejects_duplicate_equation_identity(self):
        with self.assertRaises(ValueError):
            group_equations([dict(id=1, primitive="add", scope="x")] * 2)

    def test_real_jaxpr_preserves_composites_and_single_step(self):
        import tempfile

        import jax
        import jax.numpy as jnp

        from graphcast_bench.logical_inventory import export_jaxpr

        weights = jnp.eye(4)

        def step(x, _):
            with jax.named_scope("linear_0"):
                x = x @ weights + 0.25
            x = jax.nn.silu(x)
            return x, None

        fn = jax.jit(lambda x: jax.lax.scan(step, x, None, length=1)[0])
        with tempfile.TemporaryDirectory() as out:
            result = export_jaxpr(jax.make_jaxpr(fn)(jnp.ones((2, 4))), out + "/inventory")
        kinds = [g["operator"] for g in result["groups"]]
        self.assertEqual(kinds.count("linear"), 1)
        self.assertEqual(kinds.count("silu"), 1)
        self.assertNotIn("exp", kinds)

    def test_multistep_scan_is_not_silently_counted_once(self):
        import tempfile

        import jax
        import jax.numpy as jnp

        from graphcast_bench.logical_inventory import export_jaxpr

        def fn(x):
            return jax.lax.scan(lambda y, _: (y + 1, None), x, None, length=2)[0]

        with tempfile.TemporaryDirectory() as out, self.assertRaises(ValueError):
            export_jaxpr(jax.make_jaxpr(fn)(jnp.ones(2)), out + "/inventory")

    def test_dead_equations_are_excluded_from_logical_counts(self):
        import tempfile

        import jax
        import jax.numpy as jnp

        from graphcast_bench.logical_inventory import export_jaxpr

        def fn(x):
            _unused = x * 13
            return x + 1

        with tempfile.TemporaryDirectory() as out:
            result = export_jaxpr(jax.make_jaxpr(fn)(jnp.ones(2)), out + "/inventory")
        self.assertEqual([g["operator"] for g in result["groups"]], ["add"])

    def test_scan_sliced_inputs_keep_upstream_dependencies(self):
        import tempfile

        import jax
        import jax.numpy as jnp

        from graphcast_bench.logical_inventory import export_jaxpr

        def fn(x):
            values = x * 3
            return jax.lax.scan(lambda carry, v: (carry, v + 1), None, values, length=1)[1]

        with tempfile.TemporaryDirectory() as out:
            result = export_jaxpr(jax.make_jaxpr(fn)(jnp.ones((1, 4))), out + "/inventory")
        self.assertIn("mul", [g["operator"] for g in result["groups"]])


if __name__ == "__main__":
    unittest.main()
