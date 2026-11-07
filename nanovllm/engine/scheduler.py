from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        token_budget = self.max_num_batched_tokens
        preempted = False

        # running: each seq decodes one token, or prefills its next chunk if its prompt isn't done yet
        while self.running and token_budget > 0:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                preempted = True
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                num_tokens = seq.num_tokens - seq.num_cached_tokens
                seq.is_prefill = num_tokens > 1
                seq.num_scheduled_tokens = min(num_tokens, token_budget)
                token_budget -= seq.num_scheduled_tokens
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))

        # waiting: prefill new prompts with the leftover budget; the last one may get only a chunk
        while self.waiting and token_budget > 0 and len(self.running) < self.max_num_seqs and not preempted:
            seq = self.waiting[0]
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                break
            self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(seq.num_tokens - seq.num_cached_tokens, token_budget)
            token_budget -= seq.num_scheduled_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        assert scheduled_seqs
        return scheduled_seqs, any(seq.is_prefill for seq in scheduled_seqs)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_tokens:    # chunked prefill not finished yet
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
