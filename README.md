# Aerial Human Detection and Search for Multi-UAV Disaster Response

Perception and search modules from a two-drone disaster-response system: a search drone flies a lawnmower coverage pattern, detects people with a fine-tuned YOLO model, and generates GPS targets so a second drone can deliver relief packages.

> Related publication: S. Kshirsagar, H. Kolhe, R. Deshmukh, *et al.*, "A Multi-Agent Drone System for Real-Time Victim Detection and Aid Delivery," *ICAMMS 2026*.
> Full system: [Multiagent Disaster Rescue Drones](https://github.com/HarshalKolhe02/Multiagent_Disaster_Rescue_Drones_NIDAR)

## Overview
| Module | File | Description |
|---|---|---|
| Human detection | `human_detection.py` | YOLO inference on the onboard camera stream; detections mapped to GPS coordinates |
| Search pattern | `LawnMower.py` | Generates lawnmower waypoints to cover a rectangular search area |
| Model | `aerial_model.pt` | YOLO weights fine-tuned on aerial imagery of people |

## Results
- ~90% human-detection accuracy at 8 m altitude in field tests.

## Usage
```bash
pip install ultralytics opencv-python dronekit
python LawnMower.py         # generate search waypoints
python human_detection.py   # run detection on the camera stream
```

## Notes
Selected modules only; the full system integration lives in the team repository linked above.

## License
MIT — see `LICENSE`.
