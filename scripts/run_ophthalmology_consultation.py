import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ophthalmology import OphthalmologyConfig, OphthalmologyConsultationEngine


def main() -> None:
    config = OphthalmologyConfig()
    engine = OphthalmologyConsultationEngine(config)

    sample_patient = {
        "chief_complaint": "右眼疼痛伴畏光3天",
        "history": "有隐形眼镜佩戴史，近期游泳后症状加重",
        "current_findings": "视力下降，流泪，结膜充血",
    }

    result = engine.consult(sample_patient, top_k=5)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
