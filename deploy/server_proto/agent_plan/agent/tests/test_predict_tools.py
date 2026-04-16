from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.tools import predict_tools


TMP_ROOT = Path('C:/workspace/yolodo2.0/agent_plan/.tmp_predict_tools')


def _make_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path, format='JPEG')


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)

    source_dir = TMP_ROOT / 'images'
    output_dir = TMP_ROOT / 'predict_out'
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        _make_image(source_dir / 'a.jpg', (32, 32), (255, 0, 0))
        _make_image(source_dir / 'b.jpg', (48, 48), (0, 255, 0))
        _make_image(source_dir / 'c.jpg', (64, 64), (0, 0, 255))
        (source_dir / 'broken.jpg').write_text('not-an-image', encoding='utf-8')

        original_load = predict_tools.service._load_model
        original_run = predict_tools.service._run_batch_inference
        original_draw = predict_tools.service._draw_detections

        predict_tools.service._load_model = lambda model: object()

        def _fake_run(_model, frames, *, conf: float, iou: float):
            outputs = []
            for frame in frames:
                width = int(frame.size[0])
                if width == 32:
                    outputs.append([
                        {'class_id': 0, 'class_name': 'Excavator', 'confidence': 0.91, 'xyxy': [1, 1, 20, 20]},
                    ])
                elif width == 48:
                    outputs.append([])
                else:
                    outputs.append([
                        {'class_id': 1, 'class_name': 'bulldozer', 'confidence': 0.83, 'xyxy': [5, 5, 30, 30]},
                        {'class_id': 1, 'class_name': 'bulldozer', 'confidence': 0.79, 'xyxy': [12, 12, 40, 40]},
                    ])
            return outputs

        predict_tools.service._run_batch_inference = _fake_run
        predict_tools.service._draw_detections = lambda frame, detections: frame

        result = predict_tools.predict_images(
            source_path=str(source_dir),
            model='fake-model.pt',
            output_dir=str(output_dir),
            save_annotated=True,
            save_labels=True,
            save_original=True,
            generate_report=True,
        )
        assert result['ok'] is True, result
        assert result['processed_images'] == 3, result
        assert result['detected_images'] == 2, result
        assert result['empty_images'] == 1, result
        assert len(result['failed_reads']) == 1, result
        assert result['class_counts']['bulldozer'] == 2, result
        assert result['class_counts']['Excavator'] == 1, result
        assert Path(result['annotated_dir']).exists(), result
        assert Path(result['labels_dir']).exists(), result
        assert Path(result['originals_dir']).exists(), result
        assert Path(result['report_path']).exists(), result
        assert len(list(Path(result['labels_dir']).glob('*.txt'))) == 3, result
        assert result['detected_samples'], result
        assert result['empty_samples'], result
        assert result['next_actions'], result
        assert result['prediction_overview']['mode'] == 'images', result
        assert result['prediction_overview']['processed_images'] == 3, result
        assert result['action_candidates'], result

        summary = predict_tools.summarize_prediction_results(report_path=result['report_path'])
        assert summary['ok'] is True, summary
        assert summary['processed_images'] == 3, summary
        assert summary['detected_images'] == 2, summary
        assert summary['empty_images'] == 1, summary
        assert summary['total_detections'] == 3, summary
        assert summary['class_counts']['bulldozer'] == 2, summary
        assert summary['class_counts']['Excavator'] == 1, summary
        assert summary['report_path'] == result['report_path'], summary
        assert summary['detected_samples'], summary
        assert summary['next_actions'], summary
        assert summary['prediction_summary_overview']['mode'] == 'images', summary
        assert summary['action_candidates'], summary

        summary_by_dir = predict_tools.summarize_prediction_results(output_dir=str(output_dir))
        assert summary_by_dir['ok'] is True, summary_by_dir
        assert summary_by_dir['report_path'] == result['report_path'], summary_by_dir

        inspection = predict_tools.inspect_prediction_outputs(output_dir=str(output_dir))
        assert inspection['ok'] is True, inspection
        assert inspection['output_dir'] == result['output_dir'], inspection
        assert inspection['artifact_root_count'] >= 2, inspection
        assert inspection['prediction_output_overview']['artifact_root_count'] >= 2, inspection
        assert inspection['action_candidates'], inspection

        exported_report = predict_tools.export_prediction_report(
            output_dir=str(output_dir),
            export_format='markdown',
        )
        assert exported_report['ok'] is True, exported_report
        assert Path(exported_report['export_path']).exists(), exported_report
        assert exported_report['export_path'].endswith('.md'), exported_report
        assert exported_report['export_overview']['export_format'] == 'markdown', exported_report
        assert exported_report['action_candidates'], exported_report

        path_lists = predict_tools.export_prediction_path_lists(output_dir=str(output_dir))
        assert path_lists['ok'] is True, path_lists
        assert path_lists['detected_count'] == 2, path_lists
        assert path_lists['empty_count'] == 1, path_lists
        assert Path(path_lists['detected_items_path']).exists(), path_lists
        assert Path(path_lists['empty_items_path']).exists(), path_lists
        assert path_lists['path_list_overview']['detected_count'] == 2, path_lists
        assert path_lists['action_candidates'], path_lists

        inspection_after_lists = predict_tools.inspect_prediction_outputs(output_dir=str(output_dir))
        assert inspection_after_lists['ok'] is True, inspection_after_lists
        assert inspection_after_lists['path_list_count'] >= 2, inspection_after_lists

        organized = predict_tools.organize_prediction_results(
            output_dir=str(output_dir),
            organize_by='by_class',
        )
        assert organized['ok'] is True, organized
        assert organized['copied_items'] == 2, organized
        assert organized['bucket_stats']['Excavator'] == 1, organized
        assert organized['bucket_stats']['bulldozer'] == 1, organized
        assert Path(organized['destination_dir']).exists(), organized
        assert organized['organization_overview']['bucket_count'] >= 2, organized
        assert organized['action_candidates'], organized

        missing_model = predict_tools.predict_images(source_path=str(source_dir), model='')
        assert missing_model['ok'] is False, missing_model
        assert '缺少模型参数' in missing_model['summary'], missing_model
        assert missing_model['prediction_overview']['mode'] == 'images', missing_model
        assert missing_model['action_candidates'], missing_model

        missing_report = predict_tools.summarize_prediction_results(output_dir=str(TMP_ROOT / 'missing'))
        assert missing_report['ok'] is False, missing_report
        assert '找不到报告文件' in missing_report['summary'], missing_report
        assert missing_report['prediction_summary_overview']['mode'] == 'summary', missing_report
        assert missing_report['action_candidates'], missing_report

        print('predict tools ok')
    finally:
        predict_tools.service._load_model = original_load
        predict_tools.service._run_batch_inference = original_run
        predict_tools.service._draw_detections = original_draw
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)


if __name__ == '__main__':
    main()
