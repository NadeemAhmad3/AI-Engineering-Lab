class ChaosController:
    """
    Chaos Degradation Controller for Day 9.
    Injects artificial system failures (slow inference, cache failure, queue overload)
    to verify observability layer detection.
    """
    def __init__(self):
        self.slow_inference_enabled = False
        self.slow_inference_delay_ms = 500.0
        self.cache_failure_enabled = False
        self.queue_overload_enabled = False

    def reset(self):
        self.slow_inference_enabled = False
        self.slow_inference_delay_ms = 500.0
        self.cache_failure_enabled = False
        self.queue_overload_enabled = False

    def configure(self, slow_inf: bool = False, delay_ms: float = 500.0, cache_fail: bool = False, queue_overload: bool = False):
        self.slow_inference_enabled = slow_inf
        self.slow_inference_delay_ms = delay_ms
        self.cache_failure_enabled = cache_fail
        self.queue_overload_enabled = queue_overload

chaos_controller = ChaosController()
