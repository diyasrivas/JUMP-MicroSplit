#!/bin/bash
# cpg0029 CP + MitoBrilliant (BR00122246): see ../cpg0029_shared/launchers/noise_models.sh --help
exec "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../cpg0029_shared/launchers/noise_models.sh" --dye_set mitobrilliant "$@"
