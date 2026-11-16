import unittest
import weakref

from graphcast_bench.training.microbatches import two_microbatch_grads


class StreamedMicrobatchTests(unittest.TestCase):
    def test_first_device_batch_freed_before_second_transfer_and_same_params(self):
        events = []
        references = []
        params = object()
        hosts = [object(), object()]
        gradient0 = object()
        gradient1 = object()

        class DeviceBatch:
            def __init__(self, number):
                self.number = number

        def transfer(host):
            number = hosts.index(host)
            if number == 1:
                self.assertIsNone(references[0]())
            value = DeviceBatch(number)
            references.append(weakref.ref(value))
            events.append(("upload", number))
            return value

        batches = [transfer(hosts[0])]

        def grad(p, b):
            self.assertIs(p, params)
            events.append(("grad", b.number))
            return b.number + 10.0, [gradient0, gradient1][b.number]

        def block(value):
            events.append(("sync",))

        def select(name):
            events.append(("select", name))

        def memory():
            return {"live": sum(r() is not None for r in references)}

        result = two_microbatch_grads(
            params, batches, hosts[1], grad, transfer, block, select, memory
        )
        self.assertEqual(result[:4], (10.0, gradient0, 11.0, gradient1))
        audit = result[4]
        self.assertEqual(audit["before_release"]["live"], 1)
        self.assertEqual(audit["after_release"]["live"], 0)
        self.assertEqual(audit["after_second_upload"]["live"], 1)
        self.assertEqual([e for e in events if e[0] == "grad"], [("grad", 0), ("grad", 1)])
        self.assertLess(events.index(("grad", 0)), events.index(("upload", 1)))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].number, 1)
        self.assertEqual(len(hosts), 2)
        for key in ["second_microbatch_h2d_seconds", "release_and_diagnostics_seconds"]:
            self.assertGreaterEqual(audit[key], 0.0)

    def test_requires_single_owned_device_batch(self):
        for batches in [[], [object(), object()]]:
            with self.assertRaises(ValueError):
                two_microbatch_grads(None, batches, None, None, None, None, None, None)


if __name__ == "__main__":
    unittest.main()
