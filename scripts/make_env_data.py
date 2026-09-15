"""生成田间物联网监测终端示例数据（可复现，随机种子固定）。

真实系统里这些数据来自 LoRa 网关 + MQTT 上云；仓库内提供一份结构一致的
样例数据，保证环境查询工具与 Agent 在离线状态下也能被完整验证。

用法：python scripts/make_env_data.py
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "data" / "env" / "field_env.csv"

SITES = [
    ("S1", "大田A区", 0.0),
    ("S2", "大田B区", 0.3),
    ("S3", "大田C区（低洼易涝）", 0.9),
    ("S4", "温室育苗区", -0.4),
    ("S5", "山坡试验田", 0.6),
]
SLOTS = [(8, 0), (12, 0), (16, 0), (20, 0)]
DAYS = 14


def build_rows() -> list[dict[str, object]]:
    rng = random.Random(20260813)
    start = datetime(2026, 8, 1, 0, 0)
    rows: list[dict[str, object]] = []

    for day in range(DAYS):
        rainy = 7 <= day <= 9
        for site_id, site_name, wet_factor in SITES:
            for hour, minute in SLOTS:
                diurnal = math.sin((hour - 8) / 12 * math.pi)
                air_temp = 22 + 7 * diurnal + rng.uniform(-1.0, 1.0) - (1.5 if rainy else 0)
                air_humidity = 62 + 16 * diurnal + wet_factor * 6 + (14 if rainy else 0) + rng.uniform(-3, 3)
                soil_moisture = 56 + wet_factor * 6 + (10 if rainy else 0) + rng.uniform(-3.5, 3.5)
                soil_ph = 6.5 + rng.uniform(-0.35, 0.35)
                light = max(600.0, 6000 + 26000 * diurnal - (9000 if rainy else 0) + rng.uniform(-2000, 2000))
                timestamp = start + timedelta(days=day, hours=hour, minutes=minute)
                rows.append(
                    {
                        "site_id": site_id,
                        "site_name": site_name,
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "air_temp_c": round(air_temp, 1),
                        "air_humidity_pct": round(min(98.0, max(30.0, air_humidity)), 1),
                        "soil_moisture_pct": round(min(88.0, max(35.0, soil_moisture)), 1),
                        "soil_temp_c": round(air_temp - 1.6 + rng.uniform(-0.5, 0.5), 1),
                        "light_lux": round(light, 0),
                        "soil_ph": round(soil_ph, 2),
                    }
                )
    return rows


def main() -> None:
    rows = build_rows()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已生成 {len(rows)} 条环境记录 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
