import random
from collections import Counter
from types import SimpleNamespace

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 4
TOKEN = 7    # what the fake model samples unless a test says otherwise


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", BLOCK_SIZE)


def make_scheduler(num_blocks=64, max_num_seqs=8, max_num_batched_tokens=32, eos=-1):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=eos,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config)


def make_seq(prompt_len, max_tokens, ignore_eos=True, start=1):
    prompt = list(range(start, start + prompt_len))
    return Sequence(prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos))


def check_block_accounting(scheduler):
    bm = scheduler.block_manager
    refs = Counter(block_id for seq in (*scheduler.running, *scheduler.waiting) for block_id in seq.block_table)
    assert set(refs) == bm.used_block_ids
    for block_id, count in refs.items():
        assert bm.blocks[block_id].ref_count == count
    assert len(bm.free_block_ids) + len(bm.used_block_ids) == len(bm.blocks)
    assert not bm.used_block_ids.intersection(bm.free_block_ids)


def assert_all_blocks_free(scheduler):
    bm = scheduler.block_manager
    assert len(bm.free_block_ids) == len(bm.blocks)
    assert not bm.used_block_ids
    assert all(block.ref_count == 0 for block in bm.blocks)


def step(scheduler, sample=lambda seq: TOKEN):
    """Run one engine step with a fake model.

    Returns (seq, num_cached_tokens, num_scheduled_tokens) for each scheduled sequence, as they were
    when the step was scheduled.
    """
    seqs, is_prefill = scheduler.schedule()
    assert 0 < len(seqs) <= scheduler.max_num_seqs
    assert sum(seq.num_scheduled_tokens for seq in seqs) <= scheduler.max_num_batched_tokens
    for seq in seqs:
        assert seq.num_scheduled_tokens > 0
        assert seq.num_cached_tokens + seq.num_scheduled_tokens <= seq.num_tokens
        # every token computed this step needs a KV-cache slot
        assert len(seq.block_table) * BLOCK_SIZE >= seq.num_cached_tokens + seq.num_scheduled_tokens
    batch = [(seq, seq.num_cached_tokens, seq.num_scheduled_tokens) for seq in seqs]
    scheduler.postprocess(seqs, [sample(seq) for seq in seqs])
    check_block_accounting(scheduler)
    return batch


def run(scheduler, seqs=(), sample=lambda seq: TOKEN, max_steps=10_000):
    for seq in seqs:
        scheduler.add(seq)
    steps = []
    while not scheduler.is_finished():
        assert len(steps) < max_steps, "scheduler stopped making progress"
        steps.append(step(scheduler, sample))
    return steps


def test_random_workload_completes_and_frees_every_block():
    rng = random.Random(0)
    # small budget and cache so the run chunks prompts, hits the prefix cache and preempts
    scheduler = make_scheduler(num_blocks=64, max_num_seqs=8, max_num_batched_tokens=16)
    seqs = [make_seq(rng.randint(1, 40), rng.randint(1, 20), start=rng.choice([1, 1000])) for _ in range(40)]
    run(scheduler, seqs)
    for seq in seqs:
        assert seq.is_finished
        assert seq.completion_token_ids == [TOKEN] * seq.max_tokens
    assert_all_blocks_free(scheduler)


def test_long_prompt_is_prefilled_in_budget_sized_chunks():
    scheduler = make_scheduler(max_num_batched_tokens=16)
    seq = make_seq(50, 3)
    steps = run(scheduler, [seq])
    assert [(cached, scheduled) for [(_, cached, scheduled)] in steps] == [
        (0, 16), (16, 16), (32, 16), (48, 2),    # prefill chunks
        (50, 1), (51, 1),                        # decode
    ]
    assert seq.completion_token_ids == [TOKEN] * 3
    assert_all_blocks_free(scheduler)


def test_finished_request_leaves_its_prefix_in_the_cache():
    scheduler = make_scheduler()
    first = make_seq(12, 2)
    scheduler.add(first)
    step(scheduler)
    first_blocks = list(first.block_table)
    run(scheduler)

    second = make_seq(12, 2)
    scheduler.add(second)
    [(seq, cached, scheduled)] = step(scheduler)
    assert seq is second
    assert (cached, scheduled) == (8, 4)    # the last block is always recomputed
    assert second.block_table[:2] == first_blocks[:2]
    run(scheduler)
    assert_all_blocks_free(scheduler)


def test_running_requests_share_prefix_blocks():
    scheduler = make_scheduler()
    first, second = make_seq(12, 10), make_seq(12, 10)
    scheduler.add(first)
    step(scheduler)    # prefills first and hashes its full blocks
    scheduler.add(second)
    batch = step(scheduler)
    assert (second, 8, 4) in batch
    shared = second.block_table[:2]
    assert shared == first.block_table[:2]
    assert all(scheduler.block_manager.blocks[block_id].ref_count == 2 for block_id in shared)
    run(scheduler)
    assert first.completion_token_ids == second.completion_token_ids == [TOKEN] * 10
    assert_all_blocks_free(scheduler)


def test_requests_are_preempted_and_resumed_when_the_kv_cache_runs_out(monkeypatch):
    scheduler = make_scheduler(num_blocks=6)
    preempted = []
    preempt = scheduler.preempt
    monkeypatch.setattr(scheduler, "preempt", lambda seq: (preempted.append(seq), preempt(seq)))
    seqs = [make_seq(4, 12, start=100 * i) for i in range(4)]
    run(scheduler, seqs)
    assert preempted
    for seq in seqs:
        assert seq.completion_token_ids == [TOKEN] * 12
    assert_all_blocks_free(scheduler)


def test_generation_stops_at_eos_unless_ignored():
    scheduler = make_scheduler(eos=9)
    stops = make_seq(5, 10, ignore_eos=False)
    ignores = make_seq(5, 10, ignore_eos=True, start=50)
    run(scheduler, [stops, ignores], sample=lambda seq: 9 if seq.num_completion_tokens == 2 else TOKEN)
    assert stops.completion_token_ids == [TOKEN, TOKEN, 9]
    assert ignores.completion_token_ids == [TOKEN, TOKEN, 9] + [TOKEN] * 7
    assert_all_blocks_free(scheduler)


def test_batches_respect_max_num_seqs():
    scheduler = make_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    seqs = [make_seq(3, 5, start=10 * i) for i in range(20)]
    steps = run(scheduler, seqs)
    assert max(len(batch) for batch in steps) == 4
    assert all(seq.is_finished for seq in seqs)


def test_running_sequences_keep_decoding_while_a_long_prompt_is_prefilled():
    scheduler = make_scheduler(max_num_batched_tokens=16)
    decoding = make_seq(4, 20)
    scheduler.add(decoding)
    step(scheduler)
    long_prompt = make_seq(50, 2, start=100)
    scheduler.add(long_prompt)
    chunks = []
    while long_prompt.num_completion_tokens == 0:
        scheduled = {seq: num_tokens for seq, _, num_tokens in step(scheduler)}
        assert scheduled.keys() == {decoding, long_prompt}
        assert scheduled[decoding] == 1
        chunks.append(scheduled[long_prompt])
    assert chunks == [15, 15, 15, 5]    # the decode token uses 1 of the 16-token budget


def test_mixed_steps_take_the_prefill_path_and_decode_only_steps_do_not():
    scheduler = make_scheduler()
    first, second = make_seq(4, 3), make_seq(6, 3, start=50)
    scheduler.add(first)
    seqs, is_prefill = scheduler.schedule()
    assert seqs == [first] and is_prefill
    scheduler.postprocess(seqs, [TOKEN])
    scheduler.add(second)
    seqs, is_prefill = scheduler.schedule()
    assert seqs == [first, second] and is_prefill    # one forward pass: first decodes, second prefills
    assert not first.is_prefill and second.is_prefill
    scheduler.postprocess(seqs, [TOKEN, TOKEN])
    seqs, is_prefill = scheduler.schedule()
    assert seqs == [first, second] and not is_prefill    # decode only, so the CUDA graph path


def test_every_running_sequence_is_scheduled_every_step():
    rng = random.Random(1)
    max_num_seqs = 6
    scheduler = make_scheduler(num_blocks=200, max_num_seqs=max_num_seqs, max_num_batched_tokens=16)
    for _ in range(30):
        scheduler.add(make_seq(rng.randint(1, 40), rng.randint(1, 20), start=rng.choice([1, 1000])))
    while not scheduler.is_finished():
        running = set(scheduler.running)
        batch = step(scheduler)
        assert running <= {seq for seq, _, _ in batch}    # the cache is big enough that nothing is preempted
        assert len(scheduler.running) <= max_num_seqs
        assert sum(seq.num_cached_tokens < seq.num_tokens - 1 for seq in scheduler.running) <= 1    # unfinished prefills


def test_newest_sequence_is_preempted_first_and_not_readmitted_in_the_same_step():
    scheduler = make_scheduler(num_blocks=6, max_num_batched_tokens=8)
    first, second = make_seq(6, 10), make_seq(16, 2, start=100)
    scheduler.add(first)
    step(scheduler)    # first prefills into 2 of the 6 blocks
    scheduler.add(second)
    step(scheduler)    # first decodes; second takes the other 4 blocks and prefills a chunk
    step(scheduler)    # another chunk; second's prompt still isn't done
    assert 0 < second.num_cached_tokens < second.num_tokens - 1 and not scheduler.block_manager.free_block_ids
    assert len(first) == 9    # first kept decoding while second prefilled; its next token needs a 3rd block
    batch = step(scheduler)    # no block is free, so second, the newest, is preempted mid-prefill
    assert [seq for seq, _, _ in batch] == [first]
    assert list(scheduler.waiting) == [second] and not second.block_table and second.num_cached_tokens == 0
    run(scheduler)
    assert first.completion_token_ids == [TOKEN] * 10 and second.completion_token_ids == [TOKEN] * 2
    assert_all_blocks_free(scheduler)


def test_decode_gets_a_new_block_only_when_its_token_starts_one():
    bm = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)
    seq = make_seq(4, 10)    # the prompt fills block 0 exactly
    bm.allocate(seq, bm.can_allocate(seq))
    seq.append_token(TOKEN)    # token 5 goes in a new block
    assert bm.can_append(seq)
    bm.may_append(seq)
    assert len(seq.block_table) == 2
    seq.append_token(TOKEN)    # token 6 fits in block 1
    assert bm.can_append(seq)
    bm.may_append(seq)
    assert len(seq.block_table) == 2
    seq.append_token(TOKEN), seq.append_token(TOKEN), seq.append_token(TOKEN)    # token 9 needs a third block
    assert not bm.can_append(seq)


def test_chunked_prefill_already_has_all_its_blocks():
    bm = BlockManager(num_blocks=3, block_size=BLOCK_SIZE)
    seq = make_seq(9, 10)    # 9 % BLOCK_SIZE == 1, but allocate() reserved blocks for all 9 tokens
    bm.allocate(seq, bm.can_allocate(seq))
    assert len(seq.block_table) == 3 and not bm.free_block_ids
    assert bm.can_append(seq)
    bm.may_append(seq)
    assert len(seq.block_table) == 3
