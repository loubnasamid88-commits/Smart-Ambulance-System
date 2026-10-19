from pathlib import Path

print((Path(__file__).resolve().parent / "vision_ai" / "best.pt").exists())
