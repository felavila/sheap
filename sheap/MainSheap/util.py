import requests
from pathlib import Path

def ensure_sfd_data(sfd_path: Path = None):
    """
    Ensure the Schlegel, Finkbeiner & Davis (1998) dust maps are available locally.
    Downloads the 4 required FITS files into `sfd_path` if missing.

    Parameters
    ----------
    sfd_path : Path, optional
        Directory where the SFD data should be stored.
        Defaults to `SuportData/sfddata` relative to this file.

    Files
    -----
        - SFD_dust_4096_ngp.fits
        - SFD_dust_4096_sgp.fits
        - SFD_mask_4096_ngp.fits
        - SFD_mask_4096_sgp.fits
    """
    if sfd_path is None:
        sfd_path = Path(__file__).resolve().parent.parent / "SuportData" / "sfddata"

    sfd_path.mkdir(parents=True, exist_ok=True)

    files = [
        "SFD_dust_4096_ngp.fits",
        "SFD_dust_4096_sgp.fits",
        "SFD_mask_4096_ngp.fits",
        "SFD_mask_4096_sgp.fits",
    ]

    base_url = "https://github.com/kbarbary/sfddata/raw/master"

    missing = [fname for fname in files if not (sfd_path / fname).exists()]
    if not missing:
        return
    print(f"For the SFD correction is necessary download a list of files ({missing}) this will be done just ones")
    for fname in missing:
        url = f"{base_url}/{fname}"
        outpath = sfd_path / fname
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(outpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)