from __future__ import annotations

import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from langgraph.checkpoint.base import empty_checkpoint

from agent_plan.agent.client.file_checkpointer import FileCheckpointSaver


WORK = Path(r'D:\yolodo2.0\agent_plan\agent\tests\_tmp_file_checkpointer')


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / 'checkpoint.pkl'
    try:
        saver = FileCheckpointSaver(path)
        checkpoint = empty_checkpoint()
        version = '00000000000000000000000000000001.0000000000000000'
        checkpoint['channel_values'] = {'messages': [{'role': 'user', 'content': 'hello'}]}
        checkpoint['channel_versions'] = {'messages': version}

        config = {'configurable': {'thread_id': 'thread-1', 'checkpoint_ns': ''}}
        saved_config = saver.put(config, checkpoint, {'source': 'unit-test', 'step': 1}, {'messages': version})
        saver.put_writes(saved_config, [('messages', {'content': 'pending-tool'})], task_id='task-1')

        reloaded = FileCheckpointSaver(path)
        latest = reloaded.get_tuple({'configurable': {'thread_id': 'thread-1', 'checkpoint_ns': ''}})
        assert latest is not None
        assert latest.checkpoint['id'] == checkpoint['id']
        assert latest.checkpoint['channel_values']['messages'][0]['content'] == 'hello'
        assert latest.metadata['source'] == 'unit-test'
        assert latest.pending_writes[0][2]['content'] == 'pending-tool'

        exact = reloaded.get_tuple(saved_config)
        assert exact is not None
        assert exact.checkpoint['id'] == checkpoint['id']

        reloaded.delete_thread('thread-1')
        after_delete = FileCheckpointSaver(path)
        assert after_delete.get_tuple({'configurable': {'thread_id': 'thread-1', 'checkpoint_ns': ''}}) is None

        path.write_bytes(b'not-a-pickle')
        corrupted = FileCheckpointSaver(path)
        assert corrupted.get_tuple({'configurable': {'thread_id': 'thread-1', 'checkpoint_ns': ''}}) is None
        assert path.with_suffix('.pkl.corrupt').exists()

        print('file checkpointer smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()