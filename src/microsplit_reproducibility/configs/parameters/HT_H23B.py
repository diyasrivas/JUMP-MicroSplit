from ._base import SplittingParameters


def _get_nm_paths(
    nm_path: str, 
) -> list[str]:
    nm_paths = []
    for _ in range(2): # duplicate the noise model for 2 channels
        nm_paths.append(str(nm_path))
    return nm_paths


def get_microsplit_parameters(
    nm_path: str,
    batch_size: int = 32,
    mmse_count: int = 50,
) -> dict:
    nm_paths = _get_nm_paths(nm_path=nm_path)
    return SplittingParameters(
        algorithm="denoisplit",
        img_size=(64, 64),
        grid_size=32,
        target_channels=2,
        multiscale_count=1,
        predict_logvar=None,
        loss_type="denoisplit",
        nm_paths=nm_paths,
        kl_type="kl",
        batch_size=batch_size,
        mmse_count=mmse_count,
    ).model_dump()
