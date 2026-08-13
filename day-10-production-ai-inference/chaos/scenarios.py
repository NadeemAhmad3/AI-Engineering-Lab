class ChaosController:
    """
    Chaos Scenario Controller for Day 10 Capstone.
    Simulates Traffic Spikes, Slow Model Inference, Cache Failures, and Worker/Queue Overload.
    """
    def __init__(self):
        self.slow_model_enabled = False
        self.slow_model_delay_ms = 500.0
        self.cache_failure_enabled = False
        self.queue_overload_enabled = False

    def reset(self):
        self.slow_model_enabled = False
        self.slow_model_delay_ms = 500.0
        self.cache_failure_enabled = False
        self.queue_overload_enabled = False

    def configure(self, slow_model: bool = False, delay_ms: float = 500.0, cache_fail: bool = False, queue_overload: bool = False):
        self.slow_model_enabled = slow_model
        self.slow_model_delay_ms = delay_ms
        self.cache_failure_enabled = cache_fail
        self.queue_overload_enabled = queue_overload

chaos_controller = ChaosController()
