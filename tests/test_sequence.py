import pickle

from nanovllm.engine.sequence import Sequence

# With tensor parallelism, rank 0 pickles the scheduled sequences to the other ranks.


def test_prefill_sequence_ships_all_token_ids():
    seq = Sequence([1, 2, 3, 4, 5])
    seq.num_cached_tokens, seq.num_scheduled_tokens, seq.block_table = 2, 3, [7]
    worker = pickle.loads(pickle.dumps(seq))
    assert worker.is_prefill
    assert worker.token_ids == [1, 2, 3, 4, 5]
    assert worker[2:5] == [3, 4, 5]
    assert worker.last_token == 5
    assert (len(worker), worker.num_cached_tokens, worker.num_scheduled_tokens, worker.block_table) == (5, 2, 3, [7])


def test_decode_sequence_ships_only_its_last_token():
    seq = Sequence([1, 2, 3, 4, 5])
    seq.append_token(6)
    seq.is_prefill = False
    seq.num_cached_tokens, seq.num_scheduled_tokens, seq.block_table = 5, 1, [7, 8]
    worker = pickle.loads(pickle.dumps(seq))
    assert not worker.is_prefill
    assert worker.token_ids == []
    assert worker.last_token == 6
    assert (len(worker), worker.num_cached_tokens, worker.num_scheduled_tokens, worker.block_table) == (6, 5, 1, [7, 8])
