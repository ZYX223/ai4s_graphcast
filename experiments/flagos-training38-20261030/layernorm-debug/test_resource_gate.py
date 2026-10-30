import unittest
from resource_gate import require_idle

class ResourceGateTests(unittest.TestCase):
    def test_reserved_memory_rejects_even_with_zero_utilization(self):
        with self.assertRaises(RuntimeError):
            require_idle({'card11':{'HCU use (%)':'0.0','vram Total Used Memory (MiB)':'87912'}},11)
    def test_compute_usage_rejects_low_memory_context(self):
        with self.assertRaises(RuntimeError):
            require_idle({'card11':{'HCU use (%)':'5.0','vram Total Used Memory (MiB)':'14'}},11)
    def test_missing_selected_card_fails_closed(self):
        with self.assertRaises(RuntimeError):
            require_idle({'card10':{'HCU use (%)':'0.0','vram Total Used Memory (MiB)':'14'}},11)
    def test_idle_card_is_accepted(self):
        self.assertTrue(require_idle({'card11':{'HCU use (%)':'0.0','vram Total Used Memory (MiB)':'14'}},11))

if __name__=='__main__':unittest.main()
