from typing import Optional
import numpy as np
from jump_portrait.fetch import get_jump_image as jp_get_jump_image


def fetch_jump_image(
    source: str,
    batch: str,
    plate: str,
    well: str,
    channel: str,
    site: int,
    correction: Optional[str] = None
) -> np.ndarray:
    
    try:
        image = jp_get_jump_image(
            source=source,
            batch=batch,
            plate=plate,
            well=well,
            channel=channel,
            site=site,
            correction=correction
        )
        return image
    
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch image: {source}/{batch}/{plate}/{well}/{channel}/site_{site}"
        ) from e


def fetch_all_channels(
    source: str,
    batch: str,
    plate: str,
    well: str,
    site: int,
    channels: list[str],
    correction: Optional[str] = None
) -> dict[str, np.ndarray]:
    
    channel_images = {}
    failed_channels = []
    
    for channel in channels:
        try:
            image = fetch_jump_image(
                source=source,
                batch=batch,
                plate=plate,
                well=well,
                channel=channel,
                site=site,
                correction=correction
            )
            channel_images[channel] = image
        except RuntimeError as e:
            failed_channels.append((channel, str(e)))
    
    if failed_channels:
        failed_str = ", ".join([f"{ch}: {err}" for ch, err in failed_channels])
        raise RuntimeError(f"Failed to fetch channels: {failed_str}")
    
    return channel_images
