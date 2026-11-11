# Lab Occupancy Detection v2

This project reads your **existing MediaMTX RTSP streams** and estimates whether at least one person is visible. It does not modify the Raspberry Pi cameras or the MediaMTX configuration.

## Setup (Windows PowerShell)

1. Install Python 3.10+ from python.org, if needed.
2. Extract this ZIP and open PowerShell in the `lab_occupancy_v2` folder.
3. Run:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
   If PowerShell blocks activation, run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and use that Python executable in subsequent commands.
4. Edit `config.json`. Replace `YOUR_SERVER_IP` with the MediaMTX server IP. `cam1` must match a configured MediaMTX path actually publishing video. Add more cameras as needed:
   ```json
   "cameras": {
     "North": "rtsp://SERVER_IP:8554/cam1",
     "South": "rtsp://SERVER_IP:8554/cam2"
   }
   ```
5. Run `python occupancy.py` (or `.\.venv\Scripts\python.exe occupancy.py`). Stop using Ctrl+C.

The first run downloads `yolo26n.pt` from Ultralytics. The model is pretrained for common object detection, with person class 0. No lab-specific model training is needed for the initial test.

## State rules

- **OCCUPIED**: one or more people detected in any fresh camera frame.
- **EMPTY**: every configured camera continuously provided fresh, person-free observations for at least 15 seconds.
- **UNKNOWN**: a camera is disconnected/stale, an inference error occurred, or there is not yet enough person-free observation time.

A camera stream can be running but have an obstructed view. **EMPTY is only an estimate of the monitored camera views**, not proof of an unoccupied lab. This software must not be used as a safety-critical interlock.

Status changes are appended to `occupancy_events.csv` (UTC timestamp, state, details). No images or video are saved.

## Troubleshooting

- Verify the camera is publishing by opening its HLS browser URL as shown in your setup guide: `http://SERVER_IP:8888/cam1`.
- Open `rtsp://SERVER_IP:8554/cam1` in VLC from the computer running the program. Ensure your AWS networking allows that specific computer to reach TCP port 8554.
- On the EC2 server: `sudo systemctl status mediamtx` and `sudo journalctl -fu mediamtx`.
- If the script stays UNKNOWN, check the OFFLINE/STALE messages and network connectivity.
- On a slower computer, increase `inference_interval_seconds` to 2 and optionally lower `imgsz=640` in `occupancy.py` to 416.

## Security

The uploaded camera guide includes default/shared passwords and suggests opening video/SSH ports publicly. Change passwords, restrict inbound ports to authorized hosts or a VPN, and configure MediaMTX authentication before production use. Don't post live camera URLs or credentials publicly. Follow your lab's privacy and data-retention policies.

## Limitations

Detection accuracy depends on lighting, angle, occlusion, and camera coverage. If any person is detected the status becomes OCCUPIED; brief misses are tolerated before EMPTY. Evaluate on real lab footage and tune settings to your needs. Stream capture threads help keep frames current, but severe stream/network stalls may require additional watchdog monitoring for unattended production use.
