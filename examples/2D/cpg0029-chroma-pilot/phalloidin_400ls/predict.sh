#!/bin/bash
# cpg0029 CP + Phenovue phalloidin 400LS (BR00122247): see ../cpg0029_shared/launchers/predict.sh --help
exec "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../cpg0029_shared/launchers/predict.sh" --dye_set phalloidin_400ls "$@"
