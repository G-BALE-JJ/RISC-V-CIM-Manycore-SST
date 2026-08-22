import unittest

from demand_response_priority_contract import (
    Response,
    ordered_responses,
    replay_responses,
    validate_completion_order,
)


class DemandResponsePriorityContractTest(unittest.TestCase):
    def test_consumer_query_output_prefetch_order_is_deterministic(self):
        requests = [
            Response(40, "prefetch", 2, 1),
            Response(12, "output", 0, 5),
            Response(11, "query", 0, 4),
            Response(10, "consumer", 0, 9),
        ]
        self.assertEqual(
            [request.request_id for request in ordered_responses(requests)],
            [10, 11, 12, 40],
        )

    def test_same_class_uses_issue_sequence_then_request_id(self):
        requests = [
            Response(8, "consumer", 0, 3),
            Response(7, "consumer", 0, 2),
            Response(6, "consumer", 0, 2),
        ]
        self.assertEqual(
            [request.request_id for request in ordered_responses(requests)],
            [6, 7, 8],
        )

    def test_finite_trace_completes_every_request_once(self):
        issued = [
            Response(1, "consumer", 0, 0),
            Response(2, "query", 0, 1),
            Response(3, "prefetch", 2, 2),
        ]
        validate_completion_order(issued, ordered_responses(issued))

    def test_pressure_replay_prioritizes_ready_work_and_drains(self):
        arrivals = [
            (0, Response(1, "prefetch", 1, 0)),
            (0, Response(2, "consumer", 0, 1)),
            (1, Response(3, "query", 0, 2)),
            (1, Response(4, "output", 0, 3)),
        ]
        result = replay_responses(arrivals)
        self.assertEqual([item.response.request_id for item in result.completed], [2, 3, 4, 1])
        self.assertEqual(result.max_queue_depth, 3)
        self.assertEqual(result.completed[-1].wait_ticks, 3)


if __name__ == "__main__":
    unittest.main()
