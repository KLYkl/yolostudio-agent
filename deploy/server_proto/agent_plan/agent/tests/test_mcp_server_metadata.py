from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def main() -> None:
    try:
        from yolostudio_agent.agent.server.mcp_server import mcp
    except Exception as exc:
        print(f'mcp server metadata skipped: {exc}')
        return

    manager = getattr(mcp, '_tool_manager', None)
    assert manager is not None, 'missing tool manager'
    tools = getattr(manager, '_tools', None)
    assert isinstance(tools, dict) and tools, 'missing registered tools'

    read_tool = tools['check_training_loop_status']
    assert read_tool.annotations is not None, read_tool
    assert read_tool.annotations.readOnlyHint is True, read_tool.annotations
    assert read_tool.annotations.destructiveHint is False, read_tool.annotations
    assert read_tool.annotations.idempotentHint is True, read_tool.annotations
    assert read_tool.output_schema is not None, read_tool.output_schema

    write_tool = tools['prepare_dataset_for_training']
    assert write_tool.annotations is not None, write_tool
    assert write_tool.annotations.readOnlyHint is False, write_tool.annotations
    assert write_tool.annotations.destructiveHint is True, write_tool.annotations
    assert write_tool.output_schema is not None, write_tool.output_schema

    action_tool = tools['start_training']
    assert action_tool.annotations is not None, action_tool
    assert action_tool.annotations.readOnlyHint is False, action_tool.annotations
    assert action_tool.annotations.destructiveHint is False, action_tool.annotations
    assert action_tool.output_schema is not None, action_tool.output_schema

    knowledge_tool = tools['analyze_training_outcome']
    metrics_schema = knowledge_tool.parameters['properties']['metrics']
    metrics_types = {item.get('type') for item in metrics_schema.get('anyOf', [])}
    assert metrics_types == {'object', 'null'}, metrics_schema
    assert metrics_schema.get('description'), metrics_schema
    assert metrics_schema.get('examples'), metrics_schema

    retrieval_tool = tools['retrieve_training_knowledge']
    signals_schema = retrieval_tool.parameters['properties']['signals']
    signal_types = {item.get('type') for item in signals_schema.get('anyOf', [])}
    assert signal_types == {'array', 'null'}, signals_schema
    assert signals_schema.get('examples'), signals_schema

    loop_tool = tools['start_training_loop']
    allowed_schema = loop_tool.parameters['properties']['allowed_tuning_params']
    allowed_types = {item.get('type') for item in allowed_schema.get('anyOf', [])}
    assert allowed_types == {'array', 'null'}, allowed_schema
    enum_values = set()
    for item in allowed_schema.get('anyOf', []):
        if item.get('type') != 'array':
            continue
        enum_values.update(item.get('items', {}).get('enum') or [])
    assert enum_values == {'lr0', 'batch', 'imgsz', 'epochs', 'optimizer'}, allowed_schema
    assert allowed_schema.get('examples'), allowed_schema

    convert_tool = tools['convert_format']
    convert_classes_schema = convert_tool.parameters['properties']['classes']
    convert_class_types = {item.get('type') for item in convert_classes_schema.get('anyOf', [])}
    assert convert_class_types == {'array', 'null'}, convert_classes_schema
    assert convert_classes_schema.get('description'), convert_classes_schema
    assert convert_classes_schema.get('examples'), convert_classes_schema

    yaml_tool = tools['generate_yaml']
    yaml_classes_schema = yaml_tool.parameters['properties']['classes']
    yaml_class_types = {item.get('type') for item in yaml_classes_schema.get('anyOf', [])}
    assert yaml_class_types == {'array', 'null'}, yaml_classes_schema
    assert yaml_classes_schema.get('description'), yaml_classes_schema
    assert yaml_classes_schema.get('examples'), yaml_classes_schema

    print('mcp server metadata ok')


if __name__ == '__main__':
    main()
