"""
pydda.tests.sample_files
==========================

Sample radar files in a number of formats.  Many of these files
are incomplete, they should only be used for testing, not production.

.. autosummary::
    :toctree: generated/

    EXAMPLE_RADAR0
    EXAMPLE_RADAR1
    SOUNDING_PATH
    LTX_GRID
    MHX_GRID
"""

import pooch


sha256_hashes = {
    "cfrad.20110520_081431.542_to_20110520_081813.238_KTLX_SUR.nc": "c43409767d3280113f24c6511505e2bd5775e4e15a3f719d89673eeab272ca9f",
    "cfrad.20110520_081444.871_to_20110520_081914.520_KICT_SUR.nc": "0a41a184d9d551837dbd47b4e4ba673a1d56eabc26453b62dfda997e06255e40",
    "example_grid_radar0.nc": "7b36da5579230078953db31752237b2dead60378c30752c3e2ab0d21f0b41622",
    "example_grid_radar1.nc": "450420c36f3163b30ead18d8fcd789fbe8f4be79cf1dc9a89e4bd5fd2ef055cb",
    "grid0.20171004.095021.nc": "ea239cc9bca120a7ef871f06dd139c0fab2afce9800487964a6502bda06fe88a",
    "grid1.20171004.095021.nc": "8b22b20afa2bd7a643b0a5aae912688c7728bd07185de7478c473872b74b728b",
    "grid1_sydney.nc": "967469882c4dfb536a29ac40b5036261036a4cfcceb9427dafc4e4e5e9947036",
    "grid2_sydney.nc": "4d0bd0ac1cb2cce5ea616280e321c9e74b9ba05f200d15d9d973af6160cec054",
    "grid3_sydney.nc": "6a6fa0a9634ebe07679c0cb289adf4433fb21d677e082f602b0adf0d6354152b",
    "grid4_sydney.nc": "a43a08aca0f065c166631a0cf0fcea1d725ece56187b12365186dd0e30b531c2",
    "grid_ltx.nc": "d848a9cf33cd3b0008f8542680b35019f9aae80940d09327c2b6ded197096ffa",
    "grid_mhx.nc": "b7ccaa9e09fdc03c02c5d254c04dfb1ecf737f170da0cc8510ef061f9adc3752",
    "ruc2anl_130_20110520_0800_001.grb2": "2a09f35d21e4119ed8f9aa1813c4bf44e5ed16404794590cd6ebd1b66e68331e",
    "sgpsondeadjustS2.c1.20110520.083000.cdf": "ad851f08c0857ffd6317f135981c589b82a753cfc83c024c1d40f3024845b1dc",
    "sgpsondewnpnC1.b1.20120520.053800.cdf": "9d487427e6756dcab3150fb16ad69ac72f308568fc3ebdc61c7b99c1379c4fb4",
    "test_era_interim.nc": "d4af3999a622d4ce0bf0427a43c6b258274ccbe79ce4a5b5e8862b8ca38dcfb2",
    "test_sounding.cdf": "828f6acfdc6f12b8df5232c75f2b649891d0796a9cc67318399364d4f14ace5d",
    "twpsondewnpnC3.b1.20060119.050300.cdf": "7bbef7f91de8a4c6e5ab083ce60b7b61d5dbbff05ef2434885b89cbd3ac733ef",
    "twpsondewnpnC3.b1.20060119.112000.cdf": "102b5bf7221a5b80a102803f5a4339ca1614d23e6daf8180c064e134f3aa0830",
    "twpsondewnpnC3.b1.20060119.163300.cdf": "8343687d880fec71a4ad4f07bc56a6dce433ad38885fd91266868fd069f1eaa5",
    "test_coarse0.nc": "692547ded6f8695191706f33b055dbd006341421eac0b1c2a3bf21909508937f",
    "test_coarse1.nc": "127153acd0eca35726c62dd0072661424b765214571c98f3c4aa77171602bdae",
    "test_fine0.nc": "a700d3a33dfe8b66c3fe7a475e2a706e88dc2f96b135ff1c76a02550eeb1b403",
    "test_fine1.nc": "222aa5a1247382d935b2e6a2df94cf58c4a3621dace00a2728e974b955d9c82b",
}

fido = pooch.create(
    path=pooch.os_cache("pydda"),
    base_url="https://github.com/rcjackson/pydda-sample-data/raw/main/pydda-sample-data/",
    registry=sha256_hashes,
)


def get_sample_file(file_name):
    """
    Retrieves a test file from the PyDDA test files GitHub repository.
    The repository is located at:

    https://github.com/rcjackson/pydda-sample-data

    Returns
    -------
    file_name: str
        Location of sample file on local machine
    """
    return fido.fetch(file_name)


EXAMPLE_RADAR0 = fido.fetch("example_grid_radar0.nc")
EXAMPLE_RADAR1 = fido.fetch("example_grid_radar1.nc")
SOUNDING_PATH = fido.fetch("test_sounding.cdf")
ERA_PATH = fido.fetch("test_era_interim.nc")
